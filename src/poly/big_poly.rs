//! Polynomials over `BigGaloisField`.

use std::fmt;

use num_bigint::BigUint;
use num_traits::{One, Zero};

use crate::error::{GaloisError, Result};
use crate::field::BigGaloisField;

/// Univariate polynomial over a `BigGaloisField`.
#[derive(Clone, PartialEq, Eq)]
pub struct BigPoly {
    field: BigGaloisField,
    /// Coefficients in degree-descending order.
    coeffs: Vec<BigUint>,
}

impl BigPoly {
    pub fn new(coeffs: Vec<BigUint>, field: BigGaloisField) -> Result<Self> {
        for c in &coeffs {
            field.validate(c)?;
        }
        let mut poly = Self { field, coeffs };
        poly.strip_leading_zeros();
        Ok(poly)
    }

    pub fn from_u64(coeffs: Vec<u64>, field: BigGaloisField) -> Result<Self> {
        Self::new(coeffs.into_iter().map(BigUint::from).collect(), field)
    }

    /// Create from ascending-order coefficients.
    pub fn new_asc(coeffs: Vec<BigUint>, field: BigGaloisField) -> Result<Self> {
        let mut desc = coeffs;
        desc.reverse();
        Self::new(desc, field)
    }

    pub fn from_u64_asc(coeffs: Vec<u64>, field: BigGaloisField) -> Result<Self> {
        Self::new_asc(coeffs.into_iter().map(BigUint::from).collect(), field)
    }

    pub fn zero(field: BigGaloisField) -> Result<Self> {
        Self::new(vec![BigUint::zero()], field)
    }

    pub fn one(field: BigGaloisField) -> Result<Self> {
        Self::new(vec![BigUint::one()], field)
    }

    pub fn x(field: BigGaloisField) -> Result<Self> {
        Self::new(vec![BigUint::one(), BigUint::zero()], field)
    }

    pub fn field(&self) -> &BigGaloisField {
        &self.field
    }

    pub fn degree(&self) -> isize {
        if self.coeffs.len() <= 1 && self.coeffs[0].is_zero() {
            -1
        } else {
            self.coeffs.len() as isize - 1
        }
    }

    pub fn is_zero(&self) -> bool {
        self.degree() < 0
    }

    pub fn is_one(&self) -> bool {
        self.degree() == 0 && self.coeffs.last() == Some(&BigUint::one())
    }

    pub fn is_monic(&self) -> bool {
        !self.is_zero() && self.coeffs[0] == BigUint::one()
    }

    pub fn coeffs(&self) -> &[BigUint] {
        &self.coeffs
    }

    /// Coefficients in degree-ascending order.
    pub fn coeffs_asc(&self) -> Vec<BigUint> {
        let mut c = self.coeffs.clone();
        c.reverse();
        c
    }

    pub fn leading_coeff(&self) -> &BigUint {
        &self.coeffs[0]
    }

    fn strip_leading_zeros(&mut self) {
        while self.coeffs.len() > 1 && self.coeffs[0].is_zero() {
            self.coeffs.remove(0);
        }
        if self.coeffs.is_empty() {
            self.coeffs.push(BigUint::zero());
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        ensure_same_field(&self.field, &other.field)?;
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(BigUint::zero);
            let b = other.coeffs.get(i).cloned().unwrap_or_else(BigUint::zero);
            out.push(self.field.add(&a, &b));
        }
        Self::new(out, self.field.clone())
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        ensure_same_field(&self.field, &other.field)?;
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).cloned().unwrap_or_else(BigUint::zero);
            let b = other.coeffs.get(i).cloned().unwrap_or_else(BigUint::zero);
            out.push(self.field.sub(&a, &b));
        }
        Self::new(out, self.field.clone())
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        ensure_same_field(&self.field, &other.field)?;
        if self.is_zero() || other.is_zero() {
            return Self::zero(self.field.clone());
        }
        let mut out = vec![BigUint::zero(); self.coeffs.len() + other.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in other.coeffs.iter().enumerate() {
                let prod = self.field.mul(a, b);
                out[i + j] = self.field.add(&out[i + j], &prod);
            }
        }
        Self::new(out, self.field.clone())
    }

    pub fn mul_scalar(&self, scalar: &BigUint) -> Result<Self> {
        let scalar = self.field.validate(scalar)?;
        if scalar.is_zero() || self.is_zero() {
            return Self::zero(self.field.clone());
        }
        let coeffs = self
            .coeffs
            .iter()
            .map(|c| self.field.mul(c, &scalar))
            .collect();
        Self::new(coeffs, self.field.clone())
    }

    pub fn divmod(&self, divisor: &Self) -> Result<(Self, Self)> {
        ensure_same_field(&self.field, &divisor.field)?;
        if divisor.is_zero() {
            return Err(GaloisError::PolynomialDivisionByZero);
        }

        let mut remainder = self.clone();
        let mut quotient_coeffs = vec![BigUint::zero()];

        let divisor_deg = divisor.degree() as usize;
        let inv_lead = self
            .field
            .div(&BigUint::one(), divisor.leading_coeff())?;

        while remainder.degree() >= divisor.degree() && !remainder.is_zero() {
            let rem_deg = remainder.degree() as usize;
            let scale = self.field.mul(remainder.leading_coeff(), &inv_lead);
            let shift = rem_deg - divisor_deg;

            if quotient_coeffs.len() <= shift {
                quotient_coeffs.resize(shift + 1, BigUint::zero());
            }
            quotient_coeffs[shift] = self.field.add(&quotient_coeffs[shift], &scale);

            let term = divisor.mul_scalar(&scale)?;
            let shifted = term.shift(shift)?;
            remainder = remainder.sub(&shifted)?;
        }

        Self::new(quotient_coeffs, self.field.clone()).map(|q| (q, remainder))
    }

    pub fn shift(&self, amount: usize) -> Result<Self> {
        if self.is_zero() {
            return Self::zero(self.field.clone());
        }
        let mut coeffs = vec![BigUint::zero(); amount];
        coeffs.extend_from_slice(&self.coeffs);
        Self::new(coeffs, self.field.clone())
    }

    pub fn rem(&self, divisor: &Self) -> Result<Self> {
        let (_, r) = self.divmod(divisor)?;
        Ok(r)
    }

    pub fn div(&self, divisor: &Self) -> Result<Self> {
        let (q, _) = self.divmod(divisor)?;
        Ok(q)
    }

    pub fn neg(&self) -> Result<Self> {
        let coeffs = self
            .coeffs
            .iter()
            .map(|c| self.field.neg(c))
            .collect();
        Self::new(coeffs, self.field.clone())
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
            for (i, c) in poly.coeffs.iter().enumerate() {
                let power = (d as usize - i) as u64;
                let scaled = poly.field.mul(c, &BigUint::from(power));
                new_coeffs.push(scaled);
            }
            poly = Self::new(new_coeffs, poly.field.clone())?;
        }
        Ok(poly)
    }

    pub fn make_monic(&self) -> Result<Self> {
        if self.is_zero() || self.is_monic() {
            return Ok(self.clone());
        }
        let inv = self
            .field
            .div(&BigUint::one(), self.leading_coeff())?;
        self.mul_scalar(&inv)
    }

    /// Raise polynomial to `exp` modulo `modulus`.
    pub fn mod_pow(&self, mut exp: usize, modulus: &Self) -> Result<Self> {
        let field = self.field().clone();
        let mut result = Self::one(field.clone())?;
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

    pub fn evaluate(&self, x: &BigUint) -> Result<BigUint> {
        let x = self.field.validate(x)?;
        let mut result = BigUint::zero();
        for coeff in &self.coeffs {
            result = self.field.add(&self.field.mul(&result, &x), coeff);
        }
        Ok(result)
    }

    pub fn format_poly(&self, var: &str) -> String {
        if self.is_zero() {
            return "0".to_string();
        }
        let d = self.degree() as usize;
        let mut terms = Vec::new();
        for (i, coeff) in self.coeffs.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }
            let power = d - i;
            let coeff_str = if *coeff == BigUint::one() && power > 0 {
                String::new()
            } else {
                coeff.to_string()
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

fn ensure_same_field(a: &BigGaloisField, b: &BigGaloisField) -> Result<()> {
    if a.characteristic() != b.characteristic() || a.degree() != b.degree() {
        return Err(GaloisError::PolynomialFieldMismatch);
    }
    Ok(())
}

impl fmt::Display for BigPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BigPoly({})", self.format_poly("x"))
    }
}

impl fmt::Debug for BigPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BigPoly({}, {})",
            self.format_poly("x"),
            self.field.name()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn big_poly_over_gf2_100() {
        let field = BigGaloisField::new(2, 100).unwrap();
        let a = BigPoly::from_u64(vec![1, 0, 1], field.clone()).unwrap();
        let b = BigPoly::from_u64(vec![1, 1], field).unwrap();
        let prod = a.mul(&b).unwrap();
        assert!(!prod.is_zero());
        assert!(prod.degree() >= 2);
    }
}
