//! Galois field GF(p^m) with BigUint elements for orders exceeding u64.

use std::fmt;
use std::sync::Arc;

use num_bigint::BigUint;
use num_traits::{One, Zero};

use crate::error::{GaloisError, Result};
use crate::field::bigint::{
    big_add, big_coeffs_to_int, big_div, big_int_to_coeffs, big_mul, big_mul_ext, big_neg, big_pow,
    big_sub, field_order_big, order_fits_u64, validate_big_element, FieldValue,
};
use crate::poly::PrimePoly;

/// Galois field with arbitrary-precision elements.
#[derive(Clone, PartialEq, Eq)]
pub struct BigGaloisField {
    inner: Arc<BigFieldInner>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BigFieldInner {
    characteristic: u64,
    degree: u32,
    order: BigUint,
    irreducible: PrimePoly,
}

impl BigGaloisField {
    /// Create GF(p^m) using BigUint arithmetic (works when order exceeds u64).
    pub fn new(p: u64, m: u32) -> Result<Self> {
        super::cache::cached_big_field(p, m)
    }

    /// Create GF(p^m) without using the singleton cache.
    pub fn new_uncached(p: u64, m: u32) -> Result<Self> {
        if m == 0 {
            return Err(GaloisError::InvalidDegree(0));
        }
        if !crate::poly::is_prime(p) {
            return Err(GaloisError::InvalidCharacteristic(p));
        }

        let order = field_order_big(p, m);
        let irreducible = if m == 1 {
            PrimePoly::new(p, vec![0, 1])?
        } else {
            crate::databases::irreducible_poly_lookup(p, m).and_then(|(degrees, coeffs)| {
                let dense = crate::conway::sparse_to_dense(&degrees, &coeffs, m);
                PrimePoly::new(p, dense)
            })?
        };

        Ok(Self {
            inner: Arc::new(BigFieldInner {
                characteristic: p,
                degree: m,
                order,
                irreducible,
            }),
        })
    }

    pub fn characteristic(&self) -> u64 {
        self.inner.characteristic
    }

    pub fn degree(&self) -> u32 {
        self.inner.degree
    }

    pub fn order(&self) -> &BigUint {
        &self.inner.order
    }

    pub fn name(&self) -> String {
        if self.degree() == 1 {
            format!("GF({})", self.characteristic())
        } else {
            format!("GF({}^{})", self.characteristic(), self.degree())
        }
    }

    pub fn validate(&self, value: &BigUint) -> Result<BigUint> {
        validate_big_element(value, &self.inner.order)?;
        Ok(value.clone())
    }

    pub fn add(&self, a: &BigUint, b: &BigUint) -> BigUint {
        self.arith(a, b, ArithOp::Add)
    }

    pub fn sub(&self, a: &BigUint, b: &BigUint) -> BigUint {
        self.arith(a, b, ArithOp::Sub)
    }

    pub fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        self.arith(a, b, ArithOp::Mul)
    }

    pub fn div(&self, a: &BigUint, b: &BigUint) -> Result<BigUint> {
        Ok(self.arith(a, b, ArithOp::Div))
    }

    pub fn neg(&self, a: &BigUint) -> BigUint {
        let av = FieldValue::from_big(a.clone());
        if self.degree() == 1 {
            big_neg(&av, self.characteristic()).to_biguint()
        } else {
            ext_neg(&av, self).to_biguint()
        }
    }

    pub fn pow(&self, base: &BigUint, exp: u64) -> BigUint {
        let bv = FieldValue::from_big(base.clone());
        if self.degree() == 1 {
            big_pow(&bv, exp, self.characteristic()).to_biguint()
        } else {
            ext_pow(&bv, exp, self).to_biguint()
        }
    }

    /// Exponentiation with arbitrary-size exponent.
    pub fn pow_big(&self, base: &BigUint, exp: &BigUint) -> BigUint {
        ext_pow_big(&FieldValue::from_big(base.clone()), exp, self).to_biguint()
    }

    pub fn irreducible_poly(&self) -> &PrimePoly {
        &self.inner.irreducible
    }

    /// A primitive element of the multiplicative group.
    pub fn primitive_element(&self) -> Result<BigUint> {
        if self.degree() == 1 {
            let small = crate::field::GaloisField::new(self.characteristic(), 1)?;
            return Ok(BigUint::from(small.primitive_element()));
        }
        if u64::try_from(self.order()).is_ok() {
            let small = crate::field::GaloisField::with_options(
                self.characteristic(),
                self.degree(),
                Some(self.irreducible_poly().clone()),
                None,
                false,
            )?;
            return Ok(BigUint::from(small.primitive_element()));
        }
        self.find_primitive_element_big()
    }

    /// Build a matching `GaloisField` when the order fits in u64.
    pub fn as_small_field(&self) -> Result<crate::field::GaloisField> {
        if self.degree() == 1 {
            return crate::field::GaloisField::new(self.characteristic(), 1);
        }
        if u64::try_from(self.order()).is_ok() {
            return crate::field::GaloisField::with_options(
                self.characteristic(),
                self.degree(),
                Some(self.irreducible_poly().clone()),
                None,
                false,
            );
        }
        Err(GaloisError::InvalidOrder(
            u64::try_from(self.order()).unwrap_or(u64::MAX),
        ))
    }

    /// Discrete logarithm when log/exp tables are available (order fits in u64).
    pub fn discrete_log(&self, value: &BigUint) -> Result<u64> {
        let small = self.as_small_field()?;
        let v = u64::try_from(value).map_err(|_| GaloisError::InvalidElement {
            value: 0,
            characteristic: self.characteristic(),
            degree: self.degree(),
        })?;
        small.discrete_log(small.validate_element(v)?)
    }

    /// α^log where α is a primitive element, when tables are available.
    pub fn exp_from_log(&self, log: u64) -> Result<BigUint> {
        let small = self.as_small_field()?;
        Ok(BigUint::from(small.exp_from_log(log)))
    }

    pub fn element(&self, value: &BigUint) -> Result<super::BigFieldElement> {
        super::BigFieldElement::new(self.clone(), value.clone())
    }

    pub fn element_u64(&self, value: u64) -> Result<super::BigFieldElement> {
        super::BigFieldElement::from_u64(self.clone(), value)
    }

    /// Format a field element for display.
    pub fn format_element(&self, value: &BigUint, repr: super::ElementRepr) -> String {
        use super::bigint::big_int_to_coeffs;
        match repr {
            super::ElementRepr::Int => value.to_string(),
            super::ElementRepr::Poly => {
                let coeffs = big_int_to_coeffs(value, self.characteristic(), self.degree());
                format_big_poly_coeffs(&coeffs, self.characteristic(), "α")
            }
            super::ElementRepr::Power => {
                if value.is_zero() {
                    "0".to_string()
                } else {
                    format!("α^({value})")
                }
            }
        }
    }

    /// Create field, automatically choosing u64 or BigUint representation.
    ///
    /// Tries a `GaloisField` when the order fits in u64, falling back to
    /// `BigGaloisField` when small-field construction fails (e.g. database
    /// irreducibles that fail strict verification).
    pub fn auto(p: u64, m: u32) -> Result<FieldKind> {
        if order_fits_u64(p, m) {
            match crate::field::GaloisField::new(p, m) {
                Ok(f) => return Ok(FieldKind::Small(f)),
                Err(_) => {}
            }
        }
        Ok(FieldKind::Big(Self::new(p, m)?))
    }

    fn arith(&self, a: &BigUint, b: &BigUint, op: ArithOp) -> BigUint {
        let av = FieldValue::from_big(a.clone());
        let bv = FieldValue::from_big(b.clone());
        let p = self.characteristic();
        let result = match (op, self.degree()) {
            (ArithOp::Add, 1) => big_add(&av, &bv, p),
            (ArithOp::Sub, 1) => big_sub(&av, &bv, p),
            (ArithOp::Mul, 1) => big_mul(&av, &bv, p),
            (ArithOp::Div, 1) => big_div(&av, &bv, p),
            (ArithOp::Add, _) => ext_add(&av, &bv, self),
            (ArithOp::Sub, _) => ext_sub(&av, &bv, self),
            (ArithOp::Mul, _) => big_mul_ext(&av, &bv, p, self.degree(), &self.inner.irreducible),
            (ArithOp::Div, _) => ext_div(&av, &bv, self),
        };
        result.to_biguint()
    }
}

enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Either a standard or big Galois field.
#[derive(Clone, Debug)]
pub enum FieldKind {
    Small(crate::field::GaloisField),
    Big(BigGaloisField),
}

impl FieldKind {
    /// Characteristic p of the field.
    pub fn characteristic(&self) -> u64 {
        match self {
            Self::Small(f) => f.characteristic(),
            Self::Big(f) => f.characteristic(),
        }
    }

    /// Extension degree m.
    pub fn degree(&self) -> u32 {
        match self {
            Self::Small(f) => f.degree(),
            Self::Big(f) => f.degree(),
        }
    }

    /// Field order p^m when it fits in u64.
    pub fn order_u64(&self) -> Option<u64> {
        match self {
            Self::Small(f) => Some(f.order()),
            Self::Big(f) => u64::try_from(f.order()).ok(),
        }
    }

    /// Human-readable name like `GF(3^5)`.
    pub fn name(&self) -> String {
        match self {
            Self::Small(f) => f.name(),
            Self::Big(f) => f.name(),
        }
    }

    /// Downcast to `GaloisField` when the order fits in u64.
    pub fn as_small(&self) -> Option<&crate::field::GaloisField> {
        match self {
            Self::Small(f) => Some(f),
            Self::Big(_) => None,
        }
    }

    /// Create a field array from integer values.
    pub fn array<I>(&self, values: I) -> Result<crate::field::GaloisArray>
    where
        I: IntoIterator<Item = u64>,
    {
        crate::field::GaloisArray::from_u64(self, values.into_iter().collect())
    }

    /// Create a zero-filled array of the given shape.
    pub fn zeros(&self, shape: &[usize]) -> crate::field::GaloisArray {
        crate::field::GaloisArray::zeros(self, shape)
    }

    /// Create a one-filled array of the given shape.
    pub fn ones(&self, shape: &[usize]) -> crate::field::GaloisArray {
        crate::field::GaloisArray::ones(self, shape)
    }

    /// Create a field element from an integer representation.
    pub fn element(&self, value: u64) -> Result<crate::field::GaloisElement> {
        crate::field::GaloisElement::from_u64(self, value)
    }

    /// A primitive element of the multiplicative group.
    pub fn primitive_element(&self) -> Result<BigUint> {
        match self {
            Self::Small(f) => Ok(BigUint::from(f.primitive_element())),
            Self::Big(f) => f.primitive_element(),
        }
    }

    /// Discrete logarithm with respect to the primitive element.
    pub fn discrete_log(&self, value: &BigUint) -> Result<u64> {
        match self {
            Self::Small(f) => {
                let v = u64::try_from(value).map_err(|_| GaloisError::InvalidElement {
                    value: 0,
                    characteristic: f.characteristic(),
                    degree: f.degree(),
                })?;
                f.discrete_log(f.validate_element(v)?)
            }
            Self::Big(f) => f.discrete_log(value),
        }
    }

    /// α^log where α is a primitive element.
    pub fn exp_from_log(&self, log: u64) -> Result<BigUint> {
        match self {
            Self::Small(f) => Ok(BigUint::from(f.exp_from_log(log))),
            Self::Big(f) => f.exp_from_log(log),
        }
    }

    /// Create a shaped array from row-major values.
    pub fn array_shape(&self, shape: &[usize], values: Vec<u64>) -> Result<crate::field::GaloisArray> {
        match self {
            Self::Small(f) => {
                Ok(crate::field::GaloisArray::Small(
                    crate::field::FieldArray::from_shape_vec(f.clone(), shape, values)?,
                ))
            }
            Self::Big(f) => Ok(crate::field::GaloisArray::Big(
                crate::field::BigFieldArray::from_shape_vec(
                    f.clone(),
                    shape,
                    values.into_iter().map(BigUint::from).collect(),
                )?,
            )),
        }
    }

    /// Return the small-field representation when available.
    pub fn as_galois_field(&self) -> Option<crate::field::GaloisField> {
        match self {
            Self::Small(f) => Some(f.clone()),
            Self::Big(_) => None,
        }
    }

    /// Return the big-field representation.
    pub fn as_big_field(&self) -> Option<BigGaloisField> {
        match self {
            Self::Small(_) => None,
            Self::Big(f) => Some(f.clone()),
        }
    }

    /// Create a field from its order q = p^m.
    pub fn from_order(order: u64) -> Result<Self> {
        let (p, m) = crate::poly::factor_prime_power(order)?;
        Self::from_prime_power(p, m)
    }

    /// Create GF(p^m), choosing small or big representation as needed.
    pub fn from_prime_power(p: u64, m: u32) -> Result<Self> {
        BigGaloisField::auto(p, m)
    }
}

/// Create GF(p^m), using a `GaloisField` or `BigGaloisField` as needed.
pub fn field(p: u64, m: u32) -> Result<FieldKind> {
    BigGaloisField::auto(p, m)
}

/// Create GF(order), using a `GaloisField` or `BigGaloisField` as needed.
pub fn field_from_order(order: u64) -> Result<FieldKind> {
    FieldKind::from_order(order)
}

/// Create GF(2^m), possibly as a `BigGaloisField`.
#[allow(non_snake_case)]
pub fn GF2(m: u32) -> Result<FieldKind> {
    field(2, m)
}

impl fmt::Debug for BigGaloisField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BigGaloisField")
            .field("name", &self.name())
            .field("order_bits", &self.order().bits())
            .finish()
    }
}

fn ext_add(a: &FieldValue, b: &FieldValue, field: &BigGaloisField) -> FieldValue {
    let coeffs = add_coeffs(a, b, field);
    FieldValue::from_big(big_coeffs_to_int(&coeffs, field.characteristic()))
}

fn ext_sub(a: &FieldValue, b: &FieldValue, field: &BigGaloisField) -> FieldValue {
    let p = field.characteristic();
    let ac = big_int_to_coeffs(&a.to_biguint(), p, field.degree());
    let bc = big_int_to_coeffs(&b.to_biguint(), p, field.degree());
    let diff: Vec<u64> = ac.iter().zip(bc.iter()).map(|(&a, &b)| (a + p - b) % p).collect();
    FieldValue::from_big(big_coeffs_to_int(&diff, p))
}

fn ext_neg(a: &FieldValue, field: &BigGaloisField) -> FieldValue {
    let p = field.characteristic();
    let ac = big_int_to_coeffs(&a.to_biguint(), p, field.degree());
    let neg: Vec<u64> = ac.iter().map(|&c| (p - c) % p).collect();
    FieldValue::from_big(big_coeffs_to_int(&neg, p))
}

fn ext_div(a: &FieldValue, b: &FieldValue, field: &BigGaloisField) -> FieldValue {
    let inv = ext_inv(b, field);
    big_mul_ext(
        a,
        &inv,
        field.characteristic(),
        field.degree(),
        &field.inner.irreducible,
    )
}

fn ext_inv(b: &FieldValue, field: &BigGaloisField) -> FieldValue {
    let exp = field.order() - BigUint::from(2u64);
    ext_pow_big(b, &exp, field)
}

fn ext_pow_big(base: &FieldValue, exp: &BigUint, field: &BigGaloisField) -> FieldValue {
    let mut result = FieldValue::from_u64(1);
    let mut b = base.clone();
    let p = field.characteristic();
    let irr = &field.inner.irreducible;
    let m = field.degree();
    let mut e = exp.clone();
    let two = BigUint::from(2u64);
    let zero = BigUint::zero();
    while e > zero {
        if &e % &two == BigUint::one() {
            result = if m == 1 {
                big_mul(&result, &b, p)
            } else {
                big_mul_ext(&result, &b, p, m, irr)
            };
        }
        b = if m == 1 {
            big_mul(&b, &b, p)
        } else {
            big_mul_ext(&b, &b, p, m, irr)
        };
        e /= &two;
    }
    result
}

fn ext_pow(base: &FieldValue, mut exp: u64, field: &BigGaloisField) -> FieldValue {
    let mut result = FieldValue::from_u64(1);
    let mut b = base.clone();
    let p = field.characteristic();
    let irr = &field.inner.irreducible;
    let m = field.degree();
    while exp > 0 {
        if exp % 2 == 1 {
            result = if m == 1 {
                big_mul(&result, &b, p)
            } else {
                big_mul_ext(&result, &b, p, m, irr)
            };
        }
        b = if m == 1 {
            big_mul(&b, &b, p)
        } else {
            big_mul_ext(&b, &b, p, m, irr)
        };
        exp /= 2;
    }
    result
}

impl BigGaloisField {
    fn find_primitive_element_big(&self) -> Result<BigUint> {
        let q_minus_1 = self.order() - BigUint::one();
        let factors = big_prime_factors(&q_minus_1);

        let mut candidate = BigUint::from(1u64);
        let limit = BigUint::from(50_000u64);
        while candidate < *self.order() && candidate <= limit {
            if self.has_full_multiplicative_order(&candidate, &q_minus_1, &factors)? {
                return Ok(candidate);
            }
            candidate += 1u64;
        }

        Err(GaloisError::NoPrimitiveElement {
            characteristic: self.characteristic(),
            degree: self.degree(),
        })
    }

    fn has_full_multiplicative_order(
        &self,
        value: &BigUint,
        order: &BigUint,
        factors: &[BigUint],
    ) -> Result<bool> {
        let value = self.validate(value)?;
        if value.is_zero() {
            return Ok(false);
        }
        if self.pow_big(&value, order) != BigUint::one() {
            return Ok(false);
        }
        for factor in factors {
            let exp = order / factor;
            if self.pow_big(&value, &exp) == BigUint::one() {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn big_prime_factors(n: &BigUint) -> Vec<BigUint> {
    if let Ok(v) = u64::try_from(n) {
        if let Ok((factors, _)) = crate::prime::factorize(v) {
            return factors.into_iter().map(BigUint::from).collect();
        }
    }

    let mut factors = Vec::new();
    let mut rem = n.clone();
    let two = BigUint::from(2u64);
    while &rem % &two == BigUint::zero() {
        factors.push(two.clone());
        rem /= &two;
    }
    let mut p = 3u64;
    while BigUint::from(p * p) <= rem {
        let bp = BigUint::from(p);
        while &rem % &bp == BigUint::zero() {
            factors.push(bp.clone());
            rem /= &bp;
        }
        p += 2;
        if p > 1_000_003 {
            break;
        }
    }
    if rem > BigUint::one() {
        factors.push(rem);
    }
    factors
}

fn add_coeffs(a: &FieldValue, b: &FieldValue, field: &BigGaloisField) -> Vec<u64> {
    let p = field.characteristic();
    let ac = big_int_to_coeffs(&a.to_biguint(), p, field.degree());
    let bc = big_int_to_coeffs(&b.to_biguint(), p, field.degree());
    ac.iter()
        .zip(bc.iter())
        .map(|(&a, &b)| (a + b) % p)
        .collect()
}

fn format_big_poly_coeffs(coeffs: &[u64], p: u64, var: &str) -> String {
    let mut terms = Vec::new();
    for (power, &coeff) in coeffs.iter().enumerate() {
        let c = coeff % p;
        if c == 0 {
            continue;
        }
        let coeff_str = if c == 1 && power > 0 {
            String::new()
        } else {
            c.to_string()
        };
        let var_str = match power {
            0 => String::new(),
            1 => var.to_string(),
            _ => format!("{var}^{power}"),
        };
        terms.push(match (coeff_str.is_empty(), var_str.is_empty()) {
            (true, true) => "1".to_string(),
            (false, true) => coeff_str,
            (true, false) => var_str,
            (false, false) => format!("{coeff_str}{var_str}"),
        });
    }
    if terms.is_empty() {
        "0".to_string()
    } else {
        terms.join(" + ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn big_field_gf2_100() {
        let f = BigGaloisField::new(2, 100).unwrap();
        let a = BigUint::from(1u64);
        let b = BigUint::from(1u64);
        assert_eq!(f.add(&a, &b), BigUint::zero());
        assert_eq!(f.mul(&a, &b), BigUint::from(1u64));
    }

    #[test]
    fn field_kind_array_small() {
        let fk = field(3, 5).unwrap();
        let arr = fk.array([1, 2, 3]).unwrap();
        assert!(arr.is_small());
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn field_kind_array_big() {
        let fk = GF2(100).unwrap();
        let arr = fk.array([1u64, 0, 1]).unwrap();
        assert!(!arr.is_small());
        let big = arr.as_big().unwrap();
        assert_eq!(big.len(), 3);
    }

    #[test]
    fn big_field_primitive_element() {
        let f = BigGaloisField::new(2, 100).unwrap();
        let alpha = f.primitive_element().unwrap();
        assert!(!alpha.is_zero());
        let q_minus_1 = f.order() - BigUint::one();
        assert_eq!(f.pow_big(&alpha, &q_minus_1), BigUint::one());
    }

    #[test]
    fn big_field_discrete_log_gf2_2() {
        let f = BigGaloisField::new(2, 2).unwrap();
        let alpha = f.primitive_element().unwrap();
        let log0 = f.discrete_log(&alpha).unwrap();
        assert_eq!(f.exp_from_log(log0).unwrap(), alpha);
    }

    #[test]
    fn auto_gf2_2_falls_back_to_big() {
        let fk = BigGaloisField::auto(2, 2).unwrap();
        assert!(fk.as_big_field().is_some());
    }

    #[test]
    fn field_kind_from_order() {
        let fk = FieldKind::from_order(11).unwrap();
        assert!(fk.as_small().is_some());
        let fk4 = FieldKind::from_order(4).unwrap();
        assert!(fk4.as_big_field().is_some());
    }
}
