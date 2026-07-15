//! Linear-feedback shift registers (Fibonacci and Galois LFSR).

use num_bigint::BigUint;
use num_traits::{One, Zero};

use crate::error::{GaloisError, Result};
use crate::field::{FieldArray, FieldKind, GaloisArray, GaloisField};
use crate::poly::{FieldPoly, Poly};

/// Fibonacci LFSR over a Galois field.
#[derive(Clone, Debug)]
pub struct Flfsr {
    field: GaloisField,
    state: Vec<u64>,
    taps: Vec<usize>,
}

impl Flfsr {
    pub fn new(field: GaloisField, state: Vec<u64>, taps: Vec<usize>) -> Result<Self> {
        for &s in &state {
            field.validate_element(s)?;
        }
        Ok(Self { field, state, taps })
    }

    pub fn step(&mut self) -> u64 {
        let feedback = self
            .taps
            .iter()
            .map(|&i| self.state[i])
            .fold(0u64, |acc, v| self.field.add(acc, v));
        let out = self.state.last().copied().unwrap_or(0);
        self.state.remove(0);
        self.state.push(feedback);
        out
    }

    pub fn generate(&mut self, n: usize) -> FieldArray {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.step());
        }
        FieldArray::new(self.field.clone(), out)
    }

    pub fn state(&self) -> &[u64] {
        &self.state
    }
}

/// Galois LFSR over a Galois field.
#[derive(Clone, Debug)]
pub struct Glfsr {
    field: GaloisField,
    state: Vec<u64>,
    feedback_poly: Vec<u64>,
}

impl Glfsr {
    pub fn new(field: GaloisField, state: Vec<u64>, feedback_poly: Vec<u64>) -> Result<Self> {
        for &s in &state {
            field.validate_element(s)?;
        }
        Ok(Self {
            field,
            state,
            feedback_poly,
        })
    }

    pub fn step(&mut self) -> u64 {
        let out = self.state[0];
        let feedback = self
            .feedback_poly
            .iter()
            .zip(self.state.iter())
            .map(|(&c, &s)| self.field.mul(c, s))
            .fold(0u64, |acc, v| self.field.add(acc, v));
        self.state.remove(0);
        self.state.push(feedback);
        out
    }

    pub fn generate(&mut self, n: usize) -> FieldArray {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.step());
        }
        FieldArray::new(self.field.clone(), out)
    }
}

/// Berlekamp-Massey algorithm: find error locator polynomial from syndromes.
pub fn berlekamp_massey(sequence: &FieldArray) -> Result<Poly> {
    let field = sequence.field().clone();
    let s = sequence.values();
    let n = s.len();
    let mut c = vec![0u64; n + 1];
    let mut b = vec![0u64; n + 1];
    c[0] = 1;
    b[0] = 1;
    let mut l = 0usize;
    let mut m = 1usize;
    let mut disc = 1u64;

    for i in 0..n {
        let mut d = s[i];
        for j in 1..=l {
            d = field.add(d, field.mul(c[j], s[i - j]));
        }

        if d == 0 {
            m += 1;
            continue;
        }

        let t = c.clone();
        let scale = field.div(d, disc)?;

        for j in 0..b.len() {
            if j + m < c.len() {
                c[j + m] = field.sub(c[j + m], field.mul(scale, b[j]));
            }
        }

        if 2 * l <= i {
            l = i + 1 - l;
            b = t;
            disc = d;
            m = 1;
        } else {
            m += 1;
        }
    }

    let coeffs: Vec<u64> = (0..=l).rev().map(|i| c[i]).collect();
    Poly::new(coeffs, field)
}

/// Berlekamp-Massey over a unified `GaloisArray`, returning a `FieldPoly`.
pub fn berlekamp_massey_array(sequence: &crate::field::GaloisArray) -> Result<crate::poly::FieldPoly> {
    match sequence {
        crate::field::GaloisArray::Small(a) => {
            Ok(crate::poly::FieldPoly::Small(berlekamp_massey(a)?))
        }
        crate::field::GaloisArray::Big(a) => berlekamp_massey_big(a),
    }
}

fn berlekamp_massey_big(sequence: &crate::field::BigFieldArray) -> Result<crate::poly::FieldPoly> {
    let field = sequence.field().clone();
    let s = sequence.values();
    let n = s.len();
    let mut c = vec![BigUint::zero(); n + 1];
    let mut b = vec![BigUint::zero(); n + 1];
    c[0] = BigUint::one();
    b[0] = BigUint::one();
    let mut l = 0usize;
    let mut m = 1usize;
    let mut disc = BigUint::one();

    for i in 0..n {
        let mut d = s[i].clone();
        for j in 1..=l {
            d = field.add(&d, &field.mul(&c[j], &s[i - j]));
        }

        if d.is_zero() {
            m += 1;
            continue;
        }

        let t = c.clone();
        let scale = field.div(&d, &disc)?;

        for j in 0..b.len() {
            if j + m < c.len() {
                c[j + m] = field.sub(&c[j + m], &field.mul(&scale, &b[j]));
            }
        }

        if 2 * l <= i {
            l = i + 1 - l;
            b = t;
            disc = d;
            m = 1;
        } else {
            m += 1;
        }
    }

    let coeffs: Vec<BigUint> = (0..=l).rev().map(|i| c[i].clone()).collect();
    Ok(crate::poly::FieldPoly::Big(crate::poly::BigPoly::new(coeffs, field)?))
}

/// Fibonacci LFSR over any `FieldKind`.
#[derive(Clone, Debug)]
pub struct FieldFlfsr {
    field: FieldKind,
    state: Vec<BigUint>,
    taps: Vec<usize>,
}

impl FieldFlfsr {
    pub fn new(field: FieldKind, state: Vec<u64>, taps: Vec<usize>) -> Result<Self> {
        let state = match &field {
            FieldKind::Small(f) => {
                for &s in &state {
                    f.validate_element(s)?;
                }
                state.into_iter().map(BigUint::from).collect()
            }
            FieldKind::Big(f) => {
                let values: Vec<BigUint> = state.into_iter().map(BigUint::from).collect();
                for v in &values {
                    f.validate(v)?;
                }
                values
            }
        };
        Ok(Self { field, state, taps })
    }

    pub fn field(&self) -> &FieldKind {
        &self.field
    }

    pub fn step(&mut self) -> BigUint {
        let feedback = self
            .taps
            .iter()
            .map(|&i| self.state[i].clone())
            .fold(BigUint::zero(), |acc, v| self.add(&acc, &v));
        let out = self.state.last().cloned().unwrap_or_else(BigUint::zero);
        self.state.remove(0);
        self.state.push(feedback);
        out
    }

    pub fn generate(&mut self, n: usize) -> Result<GaloisArray> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.step());
        }
        match &self.field {
            FieldKind::Small(f) => {
                let values: Vec<u64> = out
                    .iter()
                    .map(|v| {
                        u64::try_from(v).map_err(|_| GaloisError::InvalidElement {
                            value: 0,
                            characteristic: f.characteristic(),
                            degree: f.degree(),
                        })
                    })
                    .collect::<Result<_>>()?;
                Ok(GaloisArray::Small(crate::field::FieldArray::new(f.clone(), values)))
            }
            FieldKind::Big(f) => Ok(GaloisArray::Big(crate::field::BigFieldArray::new(
                f.clone(),
                out,
            )?)),
        }
    }

    pub fn state(&self) -> Result<GaloisArray> {
        match &self.field {
            FieldKind::Small(f) => {
                let values: Vec<u64> = self
                    .state
                    .iter()
                    .map(|v| {
                        u64::try_from(v).map_err(|_| GaloisError::InvalidElement {
                            value: 0,
                            characteristic: f.characteristic(),
                            degree: f.degree(),
                        })
                    })
                    .collect::<Result<_>>()?;
                Ok(GaloisArray::Small(crate::field::FieldArray::new(f.clone(), values)))
            }
            FieldKind::Big(f) => Ok(GaloisArray::Big(crate::field::BigFieldArray::new(
                f.clone(),
                self.state.clone(),
            )?)),
        }
    }

    fn add(&self, a: &BigUint, b: &BigUint) -> BigUint {
        match &self.field {
            FieldKind::Small(f) => BigUint::from(f.add(
                u64::try_from(a).unwrap_or(0),
                u64::try_from(b).unwrap_or(0),
            )),
            FieldKind::Big(f) => f.add(a, b),
        }
    }
}

/// Galois LFSR over any `FieldKind`.
#[derive(Clone, Debug)]
pub struct FieldGlfsr {
    field: FieldKind,
    state: Vec<BigUint>,
    feedback_poly: Vec<BigUint>,
}

impl FieldGlfsr {
    pub fn new(field: FieldKind, state: Vec<u64>, feedback_poly: Vec<u64>) -> Result<Self> {
        let state = match &field {
            FieldKind::Small(f) => {
                for &s in &state {
                    f.validate_element(s)?;
                }
                state.into_iter().map(BigUint::from).collect()
            }
            FieldKind::Big(f) => {
                let values: Vec<BigUint> = state.into_iter().map(BigUint::from).collect();
                for v in &values {
                    f.validate(v)?;
                }
                values
            }
        };
        let feedback_poly = feedback_poly.into_iter().map(BigUint::from).collect();
        Ok(Self {
            field,
            state,
            feedback_poly,
        })
    }

    pub fn from_field_poly(field: FieldKind, state: Vec<u64>, poly: &FieldPoly) -> Result<Self> {
        let coeffs = match poly {
            FieldPoly::Small(p) => p.coeffs_asc(),
            FieldPoly::Big(p) => p
                .coeffs_asc()
                .into_iter()
                .map(|c| {
                    u64::try_from(&c).map_err(|_| GaloisError::InvalidElement {
                        value: 0,
                        characteristic: field.characteristic(),
                        degree: field.degree(),
                    })
                })
                .collect::<Result<_>>()?,
        };
        Self::new(field, state, coeffs)
    }

    pub fn step(&mut self) -> BigUint {
        let out = self.state[0].clone();
        let feedback = self
            .feedback_poly
            .iter()
            .zip(self.state.iter())
            .map(|(c, s)| self.mul(c, s))
            .fold(BigUint::zero(), |acc, v| self.add(&acc, &v));
        self.state.remove(0);
        self.state.push(feedback);
        out
    }

    pub fn generate(&mut self, n: usize) -> Result<GaloisArray> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.step());
        }
        match &self.field {
            FieldKind::Small(f) => {
                let values: Vec<u64> = out
                    .iter()
                    .map(|v| {
                        u64::try_from(v).map_err(|_| GaloisError::InvalidElement {
                            value: 0,
                            characteristic: f.characteristic(),
                            degree: f.degree(),
                        })
                    })
                    .collect::<Result<_>>()?;
                Ok(GaloisArray::Small(crate::field::FieldArray::new(f.clone(), values)))
            }
            FieldKind::Big(f) => Ok(GaloisArray::Big(crate::field::BigFieldArray::new(
                f.clone(),
                out,
            )?)),
        }
    }

    fn add(&self, a: &BigUint, b: &BigUint) -> BigUint {
        match &self.field {
            FieldKind::Small(f) => BigUint::from(f.add(
                u64::try_from(a).unwrap_or(0),
                u64::try_from(b).unwrap_or(0),
            )),
            FieldKind::Big(f) => f.add(a, b),
        }
    }

    fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        match &self.field {
            FieldKind::Small(f) => BigUint::from(f.mul(
                u64::try_from(a).unwrap_or(0),
                u64::try_from(b).unwrap_or(0),
            )),
            FieldKind::Big(f) => f.mul(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GaloisField;

    #[test]
    fn berlekamp_massey_gf2() {
        let gf = GaloisField::new(2, 1).unwrap();
        // Fibonacci sequence mod 2: 1,1,0,1,1,0,...
        let seq = gf.array([1, 1, 0, 1, 1, 0]).unwrap();
        let poly = berlekamp_massey(&seq).unwrap();
        assert!(!poly.is_zero());
    }

    #[test]
    fn field_flfsr_gf2() {
        let fk = FieldKind::Small(GaloisField::new(2, 1).unwrap());
        let mut lfsr = FieldFlfsr::new(fk, vec![1, 0, 1], vec![0, 2]).unwrap();
        let out = lfsr.generate(6).unwrap();
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn field_glfsr_big_gf2_2() {
        let fk = FieldKind::Big(crate::field::BigGaloisField::new(2, 2).unwrap());
        let mut lfsr = FieldGlfsr::new(fk, vec![1, 1], vec![1, 1]).unwrap();
        let out = lfsr.generate(4).unwrap();
        assert_eq!(out.len(), 4);
    }
}
