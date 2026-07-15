//! Forward error correction codes: BCH and Reed-Solomon.

mod decode;
mod field_code;
mod field_decode;

use crate::error::{GaloisError, Result};
use crate::field::{FieldArray, GaloisField};
use crate::poly::{poly_from_roots, Poly};

pub use field_code::{FieldBch, FieldReedSolomon};

/// Reed-Solomon code over GF(q).
#[derive(Clone, Debug)]
pub struct ReedSolomon {
    pub(crate) field: GaloisField,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) generator: Poly,
}

impl ReedSolomon {
    /// Create RS(n, k) code over the given field.
    pub fn new(field: GaloisField, n: usize, k: usize) -> Result<Self> {
        if k >= n {
            return Err(GaloisError::InvalidDegree(k as u64));
        }
        let d = n - k + 1;
        let alpha = field.primitive_element();
        let mut roots = Vec::with_capacity(d - 1);
        for i in 1..d {
            roots.push(field.pow(alpha, i as u64));
        }
        let generator = poly_from_roots(&roots, field.clone())?;
        Ok(Self {
            field,
            n,
            k,
            generator,
        })
    }

    pub fn encode(&self, message: &FieldArray) -> Result<FieldArray> {
        if message.len() != self.k {
            return Err(GaloisError::LengthMismatch);
        }
        let mut codeword = vec![0u64; self.n];
        for (i, &symbol) in message.values().iter().enumerate() {
            codeword[self.n - self.k + i] = symbol;
        }
        let msg_poly = Poly::new_asc(codeword.clone(), self.field.clone())?;
        let (_, remainder) = msg_poly.divmod(&self.generator)?;
        for (i, &c) in remainder.coeffs_asc().iter().enumerate() {
            if i < self.n - self.k {
                codeword[i] = self.field.sub(0, c);
            }
        }
        Ok(FieldArray::new(self.field.clone(), codeword))
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }
}

/// BCH code over GF(q).
#[derive(Clone, Debug)]
pub struct Bch {
    pub(crate) field: GaloisField,
    pub(crate) n: usize,
    pub(crate) k: usize,
    t: usize,
    pub(crate) generator: Poly,
}

impl Bch {
    /// Create a BCH code with designed distance 2t+1.
    pub fn new(field: GaloisField, n: usize, k: usize, t: usize) -> Result<Self> {
        let alpha = field.primitive_element();
        let mut roots = Vec::new();
        for i in 1..=2 * t {
            roots.push(field.pow(alpha, i as u64));
        }
        let generator = poly_from_roots(&roots, field.clone())?;
        Ok(Self {
            field,
            n,
            k,
            t,
            generator,
        })
    }

    pub fn encode(&self, message: &FieldArray) -> Result<FieldArray> {
        if message.len() != self.k {
            return Err(GaloisError::LengthMismatch);
        }
        let rs = ReedSolomon {
            field: self.field.clone(),
            n: self.n,
            k: self.k,
            generator: self.generator.clone(),
        };
        rs.encode(message)
    }

    pub fn t(&self) -> usize {
        self.t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GaloisField;

    #[test]
    fn rs_encode_length() {
        let gf = GaloisField::new(7, 1).unwrap();
        let rs = ReedSolomon::new(gf.clone(), 6, 4).unwrap();
        let msg = gf.array([1, 2, 3, 4]).unwrap();
        let codeword = rs.encode(&msg).unwrap();
        assert_eq!(codeword.len(), 6);
        assert_eq!(&codeword.values()[2..], msg.values());
    }
}
