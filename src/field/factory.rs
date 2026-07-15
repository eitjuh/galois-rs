use std::fmt;
use std::sync::Arc;

use crate::conway::default_irreducible;
use crate::error::{GaloisError, Result};
use crate::poly::PrimePoly;

use super::element::{ElementRepr, FieldElement};
use super::ops::FieldArray;

/// Runtime Galois field GF(p^m), analogous to a `galois.GF()` field class in Python.
#[derive(Clone, PartialEq, Eq)]
pub struct GaloisField {
    inner: Arc<FieldInner>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FieldInner {
    characteristic: u64,
    degree: u32,
    order: u64,
    irreducible: PrimePoly,
    /// Coefficients of x^m mod irreducible: index i is coeff of x^i.
    reduction_coeffs: Vec<u64>,
    /// log table indexed by integer representation; log[0] unused.
    log_table: Vec<u32>,
    /// exp table indexed by discrete log.
    exp_table: Vec<u64>,
    primitive_element: u64,
    default_repr: ElementRepr,
}

impl GaloisField {
    /// Create GF(p^m) from characteristic `p` and extension degree `m`.
    pub fn new(p: u64, m: u32) -> Result<Self> {
        super::cache::cached_field(p, m)
    }

    /// Create GF(p^m) from field order `p^m`.
    pub fn from_order(field_order: u64) -> Result<Self> {
        super::cache::cached_field_from_order(field_order)
    }

    /// Create GF(p^m) without using the singleton cache.
    pub fn new_uncached(p: u64, m: u32) -> Result<Self> {
        Self::with_options(p, m, None, None, true)
    }

    /// Create GF(p^m) with optional irreducible polynomial and primitive element.
    pub fn with_options(
        p: u64,
        m: u32,
        irreducible: Option<PrimePoly>,
        primitive_element: Option<u64>,
        verify: bool,
    ) -> Result<Self> {
        if m == 0 {
            return Err(GaloisError::InvalidDegree(0));
        }
        if !crate::poly::is_prime(p) {
            return Err(GaloisError::InvalidCharacteristic(p));
        }

        let order = p
            .checked_pow(m)
            .ok_or(GaloisError::InvalidOrder(p.pow(m)))?;

        let irreducible = match irreducible {
            Some(poly) => poly,
            None if m == 1 => PrimePoly::new(p, vec![0, 1])?,
            None => default_irreducible(p, m)?,
        };

        if irreducible.characteristic() != p {
            return Err(GaloisError::FieldMismatch);
        }
        if irreducible.degree() as u32 != m {
            return Err(GaloisError::InvalidIrreducibleDegree {
                expected: m,
                actual: irreducible.degree() as usize,
            });
        }
        if m > 1 && !irreducible.is_monic() {
            return Err(GaloisError::NonMonicIrreducible);
        }
        if verify && m > 1 && !irreducible.is_irreducible() {
            return Err(GaloisError::ReduciblePolynomial { characteristic: p });
        }

        let reduction_coeffs = if m == 1 {
            irreducible.coeffs().to_vec()
        } else {
            irreducible.coeffs()[..m as usize].to_vec()
        };
        let field = Self {
            inner: Arc::new(FieldInner {
                characteristic: p,
                degree: m,
                order,
                irreducible,
                reduction_coeffs,
                log_table: Vec::new(),
                exp_table: Vec::new(),
                primitive_element: 0,
                default_repr: ElementRepr::Int,
            }),
        };

        let primitive = match primitive_element {
            Some(value) => field.validate_element(value)?,
            None => field.find_primitive_element()?,
        };

        let (log_table, exp_table) = field.build_log_exp_tables(primitive)?;

        Ok(Self {
            inner: Arc::new(FieldInner {
                characteristic: p,
                degree: m,
                order,
                irreducible: field.inner.irreducible.clone(),
                reduction_coeffs: field.inner.reduction_coeffs.clone(),
                log_table,
                exp_table,
                primitive_element: primitive,
                default_repr: ElementRepr::Int,
            }),
        })
    }

    pub fn characteristic(&self) -> u64 {
        self.inner.characteristic
    }

    pub fn degree(&self) -> u32 {
        self.inner.degree
    }

    pub fn order(&self) -> u64 {
        self.inner.order
    }

    pub fn irreducible_poly(&self) -> &PrimePoly {
        &self.inner.irreducible
    }

    pub fn primitive_element(&self) -> u64 {
        self.inner.primitive_element
    }

    pub fn default_repr(&self) -> ElementRepr {
        self.inner.default_repr
    }

    pub fn set_default_repr(&mut self, repr: ElementRepr) {
        Arc::make_mut(&mut self.inner).default_repr = repr;
    }

    pub fn name(&self) -> String {
        if self.degree() == 1 {
            format!("GF({})", self.characteristic())
        } else {
            format!("GF({}^{})", self.characteristic(), self.degree())
        }
    }

    pub fn element(&self, value: u64) -> Result<FieldElement> {
        let value = self.validate_element(value)?;
        Ok(FieldElement::new(self.clone(), value))
    }

    pub fn array<I>(&self, values: I) -> Result<FieldArray>
    where
        I: IntoIterator<Item = u64>,
    {
        let values = values
            .into_iter()
            .map(|v| self.validate_element(v))
            .collect::<Result<Vec<_>>>()?;
        Ok(FieldArray::new(self.clone(), values))
    }

    /// Create array with explicit shape.
    pub fn array_shape(&self, shape: &[usize], values: Vec<u64>) -> Result<FieldArray> {
        FieldArray::from_shape_vec(self.clone(), shape, values)
    }

    /// Whether this field's order exceeds u64 (requires bigint for full arithmetic).
    pub fn needs_bigint(&self) -> bool {
        crate::field::needs_bigint(self.characteristic(), self.degree())
    }

    /// Convert the irreducible polynomial to a `Poly` over this field.
    pub fn irreducible_as_poly(&self) -> Result<crate::poly::Poly> {
        let coeffs = self.inner.irreducible.coeffs();
        let mut desc: Vec<u64> = coeffs.to_vec();
        desc.reverse();
        crate::poly::Poly::new(desc, self.clone())
    }

    /// Square root in the field (Tonelli-Shanks with fast path).
    pub fn sqrt(&self, a: u64) -> Result<u64> {
        let a = self.validate_element(a)?;
        if a == 0 {
            return Ok(0);
        }
        if self.characteristic() == 2 {
            return self.sqrt_char2(a);
        }
        let exp = (self.order() + 1) / 2;
        let y = self.pow(a, exp);
        if self.mul(y, y) == a {
            return Ok(y);
        }
        self.tonelli_shanks(a)
    }

    /// Multiplicative order of a nonzero element.
    pub fn multiplicative_order(&self, a: u64) -> Result<u64> {
        let a = self.validate_element(a)?;
        if a == 0 {
            return Err(GaloisError::DivisionByZero {
                characteristic: self.characteristic(),
                degree: self.degree(),
            });
        }
        let q = self.order();
        let (factors, _) = crate::prime::factorize(q - 1).unwrap_or((vec![q - 1], vec![1]));
        let mut order = q - 1;
        for &p in &factors {
            while order % p == 0 && self.pow(a, order / p) == 1 {
                order /= p;
            }
        }
        Ok(order)
    }

    fn sqrt_char2(&self, a: u64) -> Result<u64> {
        if self.degree() == 1 {
            return Ok(a);
        }
        let m = self.degree();
        let y = self.pow(a, 1u64 << (m - 1));
        if self.mul(y, y) == a {
            Ok(y)
        } else {
            Err(GaloisError::InvalidElement {
                value: a,
                characteristic: self.characteristic(),
                degree: self.degree(),
            })
        }
    }

    fn tonelli_shanks(&self, a: u64) -> Result<u64> {
        let q = self.order();
        let mut s = 0u32;
        let mut t = q - 1;
        while t % 2 == 0 {
            t /= 2;
            s += 1;
        }
        if t == 1 {
            return Ok(self.pow(a, (t + 1) / 2));
        }

        let mut z = 2u64;
        while self.pow(z, (q - 1) / 2) == 1 {
            z += 1;
        }

        let mut m = s;
        let mut c = self.pow(z, t);
        let mut r = self.pow(a, (t + 1) / 2);
        let mut t_val = self.pow(a, t);
        let mut m_val = t;

        while t_val != 1 {
            let mut i = 1u32;
            let mut temp = self.mul(t_val, t_val);
            while i < m && temp != 1 {
                temp = self.mul(temp, temp);
                i += 1;
            }
            if i == m {
                return Err(GaloisError::InvalidElement {
                    value: a,
                    characteristic: self.characteristic(),
                    degree: self.degree(),
                });
            }
            let b = self.pow(c, m_val / 2u64.pow(i));
            m = i;
            c = self.mul(b, b);
            t_val = self.mul(t_val, c);
            r = self.mul(r, b);
            m_val = t;
        }
        Ok(r)
    }

    pub(crate) fn validate_element(&self, value: u64) -> Result<u64> {
        if value >= self.order() {
            return Err(GaloisError::InvalidElement {
                value,
                characteristic: self.characteristic(),
                degree: self.degree(),
            });
        }
        Ok(value)
    }

    pub(crate) fn add(&self, a: u64, b: u64) -> u64 {
        if self.degree() == 1 {
            (a + b) % self.characteristic()
        } else {
            let mut coeffs = self.int_to_coeffs(a);
            let other = self.int_to_coeffs(b);
            for (dst, src) in coeffs.iter_mut().zip(other.iter()) {
                *dst = (*dst + src) % self.characteristic();
            }
            self.coeffs_to_int(&coeffs)
        }
    }

    pub(crate) fn sub(&self, a: u64, b: u64) -> u64 {
        if self.degree() == 1 {
            (a + self.characteristic() - b % self.characteristic()) % self.characteristic()
        } else {
            let p = self.characteristic();
            let mut coeffs = self.int_to_coeffs(a);
            let other = self.int_to_coeffs(b);
            for (dst, src) in coeffs.iter_mut().zip(other.iter()) {
                *dst = (*dst + p - src) % p;
            }
            self.coeffs_to_int(&coeffs)
        }
    }

    pub(crate) fn mul(&self, a: u64, b: u64) -> u64 {
        if self.degree() == 1 {
            (a * b) % self.characteristic()
        } else if a == 0 || b == 0 {
            0
        } else {
            let product = self.mul_polynomials(&self.int_to_coeffs(a), &self.int_to_coeffs(b));
            self.reduce_polynomial(&product)
        }
    }

    pub(crate) fn div(&self, a: u64, b: u64) -> Result<u64> {
        if b == 0 {
            return Err(GaloisError::DivisionByZero {
                characteristic: self.characteristic(),
                degree: self.degree(),
            });
        }
        if a == 0 {
            return Ok(0);
        }
        if self.inner.exp_table.is_empty() {
            let q = self.order();
            let inv_b = self.pow_by_mul(b, q - 2);
            return Ok(self.mul(a, inv_b));
        }
        let log_a = self.log(a)?;
        let log_b = self.log(b)?;
        let log_len = (self.order() - 1) as usize;
        let exp = (log_a as usize + log_len - log_b as usize % log_len) % log_len;
        Ok(self.inner.exp_table[exp])
    }

    pub(crate) fn neg(&self, a: u64) -> u64 {
        if a == 0 {
            0
        } else if self.degree() == 1 {
            self.characteristic() - a
        } else {
            let p = self.characteristic();
            let coeffs = self
                .int_to_coeffs(a)
                .into_iter()
                .map(|c| (p - c) % p)
                .collect::<Vec<_>>();
            self.coeffs_to_int(&coeffs)
        }
    }

    pub(crate) fn pow(&self, base: u64, exp: u64) -> u64 {
        if exp == 0 {
            return 1;
        }
        if base == 0 {
            return 0;
        }
        if self.inner.exp_table.is_empty() {
            return self.pow_by_mul(base, exp);
        }
        let log_base = self.log(base).unwrap_or(0);
        let log_len = (self.order() - 1) as usize;
        let idx = (log_base as usize * exp as usize) % log_len;
        self.inner.exp_table[idx]
    }

    fn pow_by_mul(&self, mut base: u64, mut exp: u64) -> u64 {
        let mut result = 1u64;
        while exp > 0 {
            if exp % 2 == 1 {
                result = self.mul(result, base);
            }
            base = self.mul(base, base);
            exp /= 2;
        }
        result
    }

    /// Discrete logarithm with respect to the field's primitive element.
    pub fn discrete_log(&self, value: u64) -> Result<u64> {
        Ok(self.log(value)? as u64)
    }

    /// Raise the primitive element to `log`.
    pub fn exp_from_log(&self, log: u64) -> u64 {
        let q = self.order();
        if self.inner.exp_table.is_empty() {
            self.pow(self.primitive_element(), log % (q - 1))
        } else {
            self.inner.exp_table[(log % (q - 1)) as usize]
        }
    }

    pub(crate) fn log(&self, value: u64) -> Result<u32> {
        if value == 0 {
            return Err(GaloisError::DivisionByZero {
                characteristic: self.characteristic(),
                degree: self.degree(),
            });
        }
        Ok(self.inner.log_table[value as usize])
    }

    pub(crate) fn int_to_coeffs(&self, value: u64) -> Vec<u64> {
        let p = self.characteristic();
        let m = self.degree() as usize;
        let mut coeffs = vec![0u64; m];
        let mut v = value;
        for coeff in coeffs.iter_mut() {
            *coeff = v % p;
            v /= p;
        }
        coeffs
    }

    pub(crate) fn coeffs_to_int(&self, coeffs: &[u64]) -> u64 {
        let p = self.characteristic();
        let mut value = 0u64;
        let mut power = 1u64;
        for &c in coeffs {
            value += c * power;
            power *= p;
        }
        value
    }

    pub(crate) fn format_element(&self, value: u64, repr: ElementRepr) -> String {
        match repr {
            ElementRepr::Int => value.to_string(),
            ElementRepr::Poly => {
                let coeffs = self.int_to_coeffs(value);
                format_poly_coeffs(&coeffs, self.characteristic(), "α")
            }
            ElementRepr::Power => {
                if value == 0 {
                    "0".to_string()
                } else {
                    format!("α^{}", self.log(value).unwrap_or(0))
                }
            }
        }
    }

    fn mul_polynomials(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let p = self.characteristic();
        let mut out = vec![0u64; a.len() + b.len() - 1];
        for (i, &av) in a.iter().enumerate() {
            for (j, &bv) in b.iter().enumerate() {
                out[i + j] = (out[i + j] + av * bv) % p;
            }
        }
        out
    }

    fn reduce_polynomial(&self, poly: &[u64]) -> u64 {
        let p = self.characteristic();
        let m = self.degree() as usize;
        let mut work = poly.to_vec();
        let f = self.inner.irreducible.coeffs();

        while work.len() > m {
            let deg = work.len() - 1;
            let coeff = work[deg];
            if coeff != 0 {
                let shift = deg - m;
                for (i, &fi) in f.iter().enumerate() {
                    let idx = shift + i;
                    if idx < work.len() {
                        work[idx] = (work[idx] + p - (coeff * fi) % p) % p;
                    }
                }
            }
            work.pop();
        }

        work.resize(m, 0);
        self.coeffs_to_int(&work)
    }

    fn find_primitive_element(&self) -> Result<u64> {
        let q = self.order();
        if self.degree() == 1 {
            for candidate in 2..self.characteristic() {
                if self.is_primitive_root(candidate) {
                    return Ok(candidate);
                }
            }
            return Ok(1);
        }

        for value in 1..q {
            if self.is_primitive_element(value) {
                return Ok(value);
            }
        }
        Err(GaloisError::ReduciblePolynomial {
            characteristic: self.characteristic(),
        })
    }

    fn is_primitive_root(&self, g: u64) -> bool {
        if g == 0 {
            return false;
        }
        let p = self.characteristic();
        self.has_full_multiplicative_order(g, p - 1)
    }

    fn is_primitive_element(&self, value: u64) -> bool {
        if value == 0 {
            return false;
        }
        self.has_full_multiplicative_order(value, self.order() - 1)
    }

    fn has_full_multiplicative_order(&self, value: u64, order: u64) -> bool {
        if self.pow(value, order) != 1 {
            return false;
        }
        let mut d = 2u64;
        let mut n = order;
        while d * d <= n {
            if n % d == 0 {
                if self.pow(value, order / d) == 1 {
                    return false;
                }
                while n % d == 0 {
                    n /= d;
                }
            }
            d += 1;
        }
        if n > 1 && self.pow(value, order / n) == 1 {
            return false;
        }
        true
    }

    fn build_log_exp_tables(&self, primitive: u64) -> Result<(Vec<u32>, Vec<u64>)> {
        let q = self.order();
        let mut log_table = vec![0u32; q as usize];
        let mut exp_table = vec![0u64; (q - 1) as usize];
        let mut value = 1u64;
        for i in 0..(q - 1) as usize {
            exp_table[i] = value;
            log_table[value as usize] = i as u32;
            value = self.mul(value, primitive);
            if value == 1 && i > 0 && i < (q - 2) as usize {
                return Err(GaloisError::ReduciblePolynomial {
                    characteristic: self.characteristic(),
                });
            }
        }
        Ok((log_table, exp_table))
    }
}

impl fmt::Debug for GaloisField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GaloisField")
            .field("name", &self.name())
            .field("order", &self.order())
            .field("irreducible_poly", &self.irreducible_poly().format_poly("x"))
            .field("primitive_element", &self.primitive_element())
            .finish()
    }
}

fn format_poly_coeffs(coeffs: &[u64], p: u64, var: &str) -> String {
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

/// Alias for `GaloisField::from_order`, mirroring Python's `galois.GF(order)`.
#[allow(non_snake_case)]
pub fn GF(field_order: u64) -> Result<GaloisField> {
    GaloisField::from_order(field_order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gf3_5_properties() {
        let field = GaloisField::new(3, 5).unwrap();
        assert_eq!(field.characteristic(), 3);
        assert_eq!(field.degree(), 5);
        assert_eq!(field.order(), 243);
        assert_eq!(
            field.irreducible_poly().format_poly("x"),
            "1 + 2x + x^5"
        );
    }

    #[test]
    fn gf3_5_arithmetic_matches_python() {
        let gf = GaloisField::new(3, 5).unwrap();
        let x = gf.array([236, 87, 38, 112]).unwrap();
        let y = gf.array([109, 17, 108, 224]).unwrap();

        assert_eq!(x.add(&y).unwrap().values(), &[18, 95, 146, 0]);
        assert_eq!(x.sub(&y).unwrap().values(), &[127, 100, 173, 224]);
        assert_eq!(x.mul(&y).unwrap().values(), &[21, 241, 179, 82]);
        assert_eq!(x.div(&y).unwrap().values(), &[67, 47, 192, 2]);
    }
}
