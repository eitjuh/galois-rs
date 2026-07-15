use crate::error::{GaloisError, Result};
use crate::field::GaloisField;
use crate::field::FieldArray;

/// A univariate polynomial over GF(p^m), analogous to Python `galois.Poly`.
#[derive(Clone, PartialEq, Eq)]
pub struct Poly {
    field: GaloisField,
    /// Coefficients in degree-descending order: [a_d, a_{d-1}, ..., a_0]
    coeffs: Vec<u64>,
}

impl Poly {
    /// Create a polynomial from coefficients in degree-descending order (Python default).
    pub fn new(coeffs: Vec<u64>, field: GaloisField) -> Result<Self> {
        let coeffs = coeffs
            .into_iter()
            .map(|c| field.validate_element(c))
            .collect::<Result<Vec<_>>>()?;
        let mut poly = Self { field, coeffs };
        poly.strip_leading_zeros();
        Ok(poly)
    }

    /// Create from ascending-order coefficients.
    pub fn new_asc(coeffs: Vec<u64>, field: GaloisField) -> Result<Self> {
        let mut desc = coeffs;
        desc.reverse();
        Self::new(desc, field)
    }

    pub fn zero(field: GaloisField) -> Result<Self> {
        Self::new(vec![0], field)
    }

    pub fn one(field: GaloisField) -> Result<Self> {
        Self::new(vec![1], field)
    }

    pub fn x(field: GaloisField) -> Result<Self> {
        Self::new(vec![1, 0], field)
    }

    pub fn field(&self) -> &GaloisField {
        &self.field
    }

    pub fn degree(&self) -> isize {
        if self.coeffs.len() <= 1 && self.coeffs[0] == 0 {
            -1
        } else {
            self.coeffs.len() as isize - 1
        }
    }

    pub fn is_zero(&self) -> bool {
        self.degree() < 0
    }

    pub fn is_one(&self) -> bool {
        self.degree() == 0 && self.coeffs.last().copied() == Some(1)
    }

    pub fn is_monic(&self) -> bool {
        !self.is_zero() && self.coeffs[0] == 1
    }

    /// Coefficients in degree-descending order.
    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
    }

    /// Coefficients in degree-ascending order.
    pub fn coeffs_asc(&self) -> Vec<u64> {
        let mut c = self.coeffs.clone();
        c.reverse();
        c
    }

    pub fn leading_coeff(&self) -> u64 {
        if self.is_zero() {
            0
        } else {
            self.coeffs[0]
        }
    }

    fn strip_leading_zeros(&mut self) {
        while self.coeffs.len() > 1 && self.coeffs[0] == 0 {
            self.coeffs.remove(0);
        }
        if self.coeffs.is_empty() {
            self.coeffs.push(0);
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        ensure_same_field(&self.field, &other.field)?;
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut out = vec![0u64; len];
        for i in 0..len {
            let a = self.coeffs.get(i).copied().unwrap_or(0);
            let b = other.coeffs.get(i).copied().unwrap_or(0);
            out[i] = self.field.add(a, b);
        }
        Self::new(out, self.field.clone())
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        ensure_same_field(&self.field, &other.field)?;
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut out = vec![0u64; len];
        for i in 0..len {
            let a = self.coeffs.get(i).copied().unwrap_or(0);
            let b = other.coeffs.get(i).copied().unwrap_or(0);
            out[i] = self.field.sub(a, b);
        }
        Self::new(out, self.field.clone())
    }

    pub fn neg(&self) -> Result<Self> {
        let coeffs = self.coeffs.iter().map(|&c| self.field.neg(c)).collect();
        Self::new(coeffs, self.field.clone())
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        ensure_same_field(&self.field, &other.field)?;
        if self.is_zero() || other.is_zero() {
            return Self::zero(self.field.clone());
        }
        let mut out = vec![0u64; self.coeffs.len() + other.coeffs.len() - 1];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                out[i + j] = self.field.add(out[i + j], self.field.mul(a, b));
            }
        }
        Self::new(out, self.field.clone())
    }

    pub fn mul_scalar(&self, scalar: u64) -> Result<Self> {
        let scalar = self.field.validate_element(scalar)?;
        if scalar == 0 || self.is_zero() {
            return Self::zero(self.field.clone());
        }
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| self.field.mul(c, scalar))
            .collect();
        Self::new(coeffs, self.field.clone())
    }

    pub fn divmod(&self, divisor: &Self) -> Result<(Self, Self)> {
        ensure_same_field(&self.field, &divisor.field)?;
        if divisor.is_zero() {
            return Err(GaloisError::PolynomialDivisionByZero);
        }

        let mut remainder = self.clone();
        let mut quotient_coeffs = vec![0u64];

        let divisor_deg = divisor.degree() as usize;
        let inv_lead = self.field.div(1, divisor.leading_coeff())?;

        while remainder.degree() >= divisor.degree() && !remainder.is_zero() {
            let rem_deg = remainder.degree() as usize;
            let scale = self.field.mul(remainder.leading_coeff(), inv_lead);
            let shift = rem_deg - divisor_deg;

            if quotient_coeffs.len() <= shift {
                quotient_coeffs.resize(shift + 1, 0);
            }
            quotient_coeffs[shift] = self.field.add(quotient_coeffs[shift], scale);

            let term = divisor.mul_scalar(scale)?;
            let shifted = term.shift(shift)?;
            remainder = remainder.sub(&shifted)?;
        }

        Self::new(quotient_coeffs, self.field.clone()).map(|q| (q, remainder))
    }

    pub fn shift(&self, amount: usize) -> Result<Self> {
        if self.is_zero() {
            return Self::zero(self.field.clone());
        }
        let mut coeffs = vec![0u64; amount];
        coeffs.extend_from_slice(&self.coeffs);
        Self::new(coeffs, self.field.clone())
    }

    pub fn rem(&self, divisor: &Self) -> Result<Self> {
        let (_, r) = self.divmod(divisor)?;
        Ok(r)
    }

    /// Raise polynomial to `exp` modulo `modulus`.
    pub fn mod_pow(&self, mut exp: usize, modulus: &Self) -> Result<Self> {
        let field = self.field().clone();
        let mut result = Poly::one(field.clone())?;
        let mut base = self.rem(modulus)?;
        while exp > 0 {
            if exp % 2 == 1 {
                result = result.mul(&base)?.rem(modulus)?;
            }
            base = base.mul(&base)?.rem(modulus)?;
            exp /= 2;
        }
        Ok(result)
    }

    pub fn div(&self, divisor: &Self) -> Result<Self> {
        let (q, _) = self.divmod(divisor)?;
        Ok(q)
    }

    pub fn evaluate(&self, x: u64) -> Result<u64> {
        let x = self.field.validate_element(x)?;
        let mut result = 0u64;
        for &coeff in &self.coeffs {
            result = self.field.add(self.field.mul(result, x), coeff);
        }
        Ok(result)
    }

    pub fn derivative(&self, k: u32) -> Result<Self> {
        if k == 0 {
            return Ok(self.clone());
        }
        if self.degree() < k as isize {
            return Self::zero(self.field.clone());
        }
        let mut poly = self.clone();
        for _ in 0..k {
            let d = poly.degree();
            if d <= 0 {
                return Self::zero(poly.field.clone());
            }
            let mut new_coeffs = Vec::with_capacity(d as usize);
            for (i, &c) in poly.coeffs.iter().enumerate() {
                let power = (d as usize - i) as u64;
                new_coeffs.push(poly.field.mul(c, power));
            }
            poly = Self::new(new_coeffs, poly.field.clone())?;
        }
        Ok(poly)
    }

    pub fn make_monic(&self) -> Result<Self> {
        if self.is_zero() || self.is_monic() {
            return Ok(self.clone());
        }
        let inv = self.field.div(1, self.leading_coeff())?;
        self.mul_scalar(inv)
    }

    pub fn format_poly(&self, var: &str) -> String {
        if self.is_zero() {
            return "0".to_string();
        }
        let mut terms = Vec::new();
        let d = self.degree() as usize;
        for (i, &coeff) in self.coeffs.iter().enumerate() {
            if coeff == 0 {
                continue;
            }
            let power = d - i;
            let coeff_str = if coeff == 1 && power > 0 {
                String::new()
            } else {
                self.field.format_element(coeff, crate::field::ElementRepr::Int)
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
        terms.join(" + ")
    }
}

impl std::fmt::Debug for Poly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Poly({}, {})",
            self.format_poly("x"),
            self.field.name()
        )
    }
}

impl std::fmt::Display for Poly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Poly({})", self.format_poly("x"))
    }
}

fn ensure_same_field(a: &GaloisField, b: &GaloisField) -> Result<()> {
    if a.order() != b.order()
        || a.characteristic() != b.characteristic()
        || a.degree() != b.degree()
    {
        return Err(GaloisError::PolynomialFieldMismatch);
    }
    Ok(())
}

/// Raise `poly` to q = p^m modulo `modulus` (one application of the Frobenius at field order).
pub fn raise_to_field_order(poly: &Poly, modulus: &Poly) -> Result<Poly> {
    let steps = poly.field().degree().max(1) as usize;
    let mut h = poly.rem(modulus)?;
    for _ in 0..steps {
        h = frobenius_step(&h, modulus)?;
    }
    Ok(h)
}

/// Frobenius map: `poly` ↦ `poly^p` mod `modulus`.
pub fn frobenius_step(h: &Poly, modulus: &Poly) -> Result<Poly> {
    let field = h.field();
    let p = field.characteristic();
    let mut result = Poly::zero(field.clone())?;
    for (i, &coeff) in h.coeffs_asc().iter().enumerate() {
        let term = Poly::new(vec![coeff], field.clone())?;
        let power = term.mod_pow(p as usize, modulus)?;
        let shifted = power.shift(i)?;
        result = result.add(&shifted)?;
    }
    result.rem(modulus)
}

/// Multiplicative inverse of `a` modulo monic `m`.
pub fn mod_inverse_poly(a: &Poly, modulus: &Poly) -> Result<Poly> {
    let field = a.field().clone();
    let mut old_r = modulus.clone();
    let mut r = a.rem(modulus)?;
    let mut old_t = Poly::zero(field.clone())?;
    let mut t = Poly::one(field.clone())?;

    while !r.is_zero() {
        let (q, rem) = old_r.divmod(&r)?;
        old_r = r;
        r = rem;
        let qt = q.mul(&t)?;
        let new_t = old_t.sub(&qt)?;
        old_t = t;
        t = new_t;
    }

    if old_r.degree() != 0 {
        return Err(GaloisError::PolynomialDivisionByZero);
    }
    let lead = old_r.coeffs()[0];
    if lead == 0 {
        return Err(GaloisError::PolynomialDivisionByZero);
    }
    t.mul_scalar(field.div(1, lead)?)
}

/// Greatest common divisor of two polynomials.
pub fn poly_gcd(a: &Poly, b: &Poly) -> Result<Poly> {
    let mut r0 = a.clone();
    let mut r1 = b.clone();
    while !r1.is_zero() {
        let (_, rem) = r0.divmod(&r1)?;
        r0 = r1;
        r1 = rem;
    }
    r0.make_monic()
}

/// Construct polynomial from roots.
pub fn poly_from_roots(roots: &[u64], field: GaloisField) -> Result<Poly> {
    let mut poly = Poly::one(field)?;
    for &root in roots {
        let root = poly.field().validate_element(root)?;
        let linear = Poly::new(vec![1, poly.field().neg(root)], poly.field().clone())?;
        poly = poly.mul(&linear)?;
    }
    Ok(poly)
}

/// Lagrange interpolating polynomial.
pub fn lagrange_poly(x: &FieldArray, y: &FieldArray) -> Result<Poly> {
    if x.len() != y.len() {
        return Err(GaloisError::LengthMismatch);
    }
    let field = x.field().clone();
    let n = x.len();
    let mut result = Poly::zero(field.clone())?;

    for i in 0..n {
        let xi = x.values()[i];
        let yi = y.values()[i];
        let mut basis = Poly::one(field.clone())?;
        let mut denom = 1u64;
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = x.values()[j];
            let linear = Poly::new(vec![1, field.neg(xj)], field.clone())?;
            basis = basis.mul(&linear)?;
            denom = field.mul(denom, field.sub(xi, xj));
        }
        let scale = field.div(yi, denom)?;
        let term = basis.mul_scalar(scale)?;
        result = result.add(&term)?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GaloisField;

    #[test]
    fn poly_over_gf7() {
        let gf = GaloisField::new(7, 1).unwrap();
        let a = Poly::new(vec![1, 0, 1, 1], gf.clone()).unwrap();
        assert_eq!(a.format_poly("x"), "x^3 + x + 1");
        let b = Poly::new(vec![1, 2], gf).unwrap();
        let prod = a.mul(&b).unwrap();
        assert!(!prod.is_zero());
    }

    #[test]
    fn poly_gf3_5() {
        let gf = GaloisField::new(3, 5).unwrap();
        let p = Poly::new(vec![124, 0, 223, 0, 0, 15], gf).unwrap();
        assert_eq!(p.degree(), 5);
    }
}
