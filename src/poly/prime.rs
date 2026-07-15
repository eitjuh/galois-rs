use crate::error::{GaloisError, Result};
use crate::poly::integer::IntegerPoly;

/// Polynomial over GF(p), stored with ascending coefficients mod p.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimePoly {
    pub(crate) p: u64,
    coeffs: Vec<u64>,
}

impl PrimePoly {
    pub fn new(p: u64, coeffs: Vec<u64>) -> Result<Self> {
        if !crate::poly::integer::is_prime(p) {
            return Err(GaloisError::InvalidCharacteristic(p));
        }
        let coeffs = coeffs.into_iter().map(|c| c % p).collect();
        let mut poly = Self { p, coeffs };
        poly.strip();
        Ok(poly)
    }

    pub fn from_integer(p: u64, poly: &IntegerPoly) -> Result<Self> {
        Self::new(p, poly.mod_coeffs(p))
    }

    pub fn zero(p: u64) -> Result<Self> {
        Self::new(p, vec![0])
    }

    pub fn one(p: u64) -> Result<Self> {
        Self::new(p, vec![1])
    }

    pub fn x(p: u64) -> Result<Self> {
        Self::new(p, vec![0, 1])
    }

    pub fn characteristic(&self) -> u64 {
        self.p
    }

    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
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

    pub fn is_monic(&self) -> bool {
        !self.is_zero() && self.coeffs.last().copied() == Some(1)
    }

    pub fn is_one(&self) -> bool {
        self.coeffs == [1]
    }

    fn strip(&mut self) {
        while self.coeffs.len() > 1 && self.coeffs.last().copied() == Some(0) {
            self.coeffs.pop();
        }
        if self.coeffs.is_empty() {
            self.coeffs.push(0);
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        self.ensure_same_field(other)?;
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).copied().unwrap_or(0);
            let b = other.coeffs.get(i).copied().unwrap_or(0);
            coeffs.push((a + b) % self.p);
        }
        Self::new(self.p, coeffs)
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.ensure_same_field(other)?;
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeffs.get(i).copied().unwrap_or(0);
            let b = other.coeffs.get(i).copied().unwrap_or(0);
            coeffs.push((a + self.p - b) % self.p);
        }
        Self::new(self.p, coeffs)
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        self.ensure_same_field(other)?;
        if self.is_zero() || other.is_zero() {
            return Self::zero(self.p);
        }
        let mut out = vec![0u64; self.coeffs.len() + other.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in other.coeffs.iter().enumerate() {
                out[i + j] = (out[i + j] + a * b) % self.p;
            }
        }
        Self::new(self.p, out)
    }

    pub fn mul_scalar(&self, scalar: u64) -> Result<Self> {
        let scalar = scalar % self.p;
        if scalar == 0 || self.is_zero() {
            return Self::zero(self.p);
        }
        Self::new(
            self.p,
            self.coeffs.iter().map(|c| (c * scalar) % self.p).collect(),
        )
    }

    pub fn divmod(&self, divisor: &Self) -> Result<(Self, Self)> {
        self.ensure_same_field(divisor)?;
        if divisor.is_zero() {
            return Err(GaloisError::DivisionByZero {
                characteristic: self.p,
                degree: 1,
            });
        }

        let mut remainder = self.clone();
        let mut quotient = Self::zero(self.p)?;

        let divisor_deg = divisor.degree() as usize;
        let inv_lead = mod_inverse(divisor.coeffs[divisor_deg], self.p)?;

        while remainder.degree() >= divisor.degree() {
            let rem_deg = remainder.degree() as usize;
            let scale = (remainder.coeffs[rem_deg] * inv_lead) % self.p;
            let shift = rem_deg - divisor_deg;
            let mut term = divisor.mul_scalar(scale)?;
            term.coeffs.resize(shift + term.coeffs.len(), 0);
            term.coeffs.rotate_right(shift);
            quotient = quotient.add(&term)?;
            remainder = remainder.sub(&term)?;
        }

        Ok((quotient, remainder))
    }

    pub fn gcd(&self, other: &Self) -> Result<Self> {
        self.ensure_same_field(other)?;
        let mut a = self.clone();
        let mut b = other.clone();
        while !b.is_zero() {
            let (_, r) = a.divmod(&b)?;
            a = b;
            b = r;
        }
        if a.is_zero() {
            return Self::zero(self.p);
        }
        let inv = mod_inverse(*a.coeffs.last().unwrap(), self.p)?;
        a.mul_scalar(inv)
    }

    /// Test irreducibility of a monic polynomial over GF(p).
    pub fn is_irreducible(&self) -> bool {
        if self.degree() <= 0 || !self.is_monic() {
            return false;
        }
        let m = self.degree() as u64;
        let p = self.p;
        let mut power = PrimePoly::x(p).unwrap();
        for _ in 1..=m / 2 {
            for _ in 0..p {
                power = power.mul(&power).unwrap();
                let (_, rem) = power.divmod(self).unwrap();
                power = rem;
            }
            let diff = power.sub(&PrimePoly::x(p).unwrap()).unwrap();
            let gcd = self.gcd(&diff).unwrap();
            if !gcd.is_one() {
                return false;
            }
        }
        true
    }

    pub fn format_poly(&self, var: &str) -> String {
        IntegerPoly::new(self.coeffs.clone()).format_mod(self.p, var)
    }

    fn ensure_same_field(&self, other: &Self) -> Result<()> {
        if self.p != other.p {
            return Err(GaloisError::FieldMismatch);
        }
        Ok(())
    }
}

pub fn mod_inverse(a: u64, p: u64) -> Result<u64> {
    if a == 0 {
        return Err(GaloisError::DivisionByZero {
            characteristic: p,
            degree: 1,
        });
    }
    let mut t = 0i64;
    let mut new_t = 1i64;
    let mut r = p as i64;
    let mut new_r = (a % p) as i64;
    while new_r != 0 {
        let quotient = r / new_r;
        (t, new_t) = (new_t, t - quotient * new_t);
        (r, new_r) = (new_r, r - quotient * new_r);
    }
    if r != 1 {
        return Err(GaloisError::DivisionByZero {
            characteristic: p,
            degree: 1,
        });
    }
    Ok(((t % p as i64) + p as i64) as u64 % p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplication_mod_3() {
        let a = PrimePoly::new(3, vec![1, 2]).unwrap();
        let b = PrimePoly::new(3, vec![2, 1]).unwrap();
        let c = a.mul(&b).unwrap();
        assert_eq!(c.coeffs(), &[2, 2, 2]);
    }
}
