//! Reed-Solomon and BCH codes over any `FieldKind`.

use num_bigint::BigUint;
use num_traits::Zero;

use crate::error::{GaloisError, Result};
use crate::field::{FieldKind, GaloisArray};
use crate::poly::{poly_from_roots_values, FieldPoly};

use super::ReedSolomon;
use super::field_decode;

/// Reed-Solomon code over small or big Galois fields.
#[derive(Clone, Debug)]
pub struct FieldReedSolomon {
    pub(crate) field: FieldKind,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) generator: FieldPoly,
}

impl FieldReedSolomon {
    /// Create RS(n, k) over the given field kind.
    pub fn new(field: FieldKind, n: usize, k: usize) -> Result<Self> {
        if k >= n {
            return Err(GaloisError::InvalidDegree(k as u64));
        }
        let d = n - k + 1;
        let alpha = field.primitive_element()?;
        let mut roots = Vec::with_capacity(d - 1);
        match &field {
            FieldKind::Small(f) => {
                let a = u64::try_from(&alpha).unwrap_or_else(|_| f.primitive_element());
                for i in 1..d {
                    roots.push(BigUint::from(f.pow(a, i as u64)));
                }
            }
            FieldKind::Big(f) => {
                for i in 1..d {
                    roots.push(f.pow(&alpha, i as u64));
                }
            }
        }
        let generator = poly_from_roots_values(&roots, &field)?;
        Ok(Self {
            field,
            n,
            k,
            generator,
        })
    }

    pub fn field(&self) -> &FieldKind {
        &self.field
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn generator(&self) -> &FieldPoly {
        &self.generator
    }

    /// Encode a message into a codeword.
    pub fn encode(&self, message: &GaloisArray) -> Result<GaloisArray> {
        if message.len() != self.k {
            return Err(GaloisError::LengthMismatch);
        }
        match (&self.field, message) {
            (FieldKind::Small(f), GaloisArray::Small(m)) => {
                let rs = ReedSolomon {
                    field: f.clone(),
                    n: self.n,
                    k: self.k,
                    generator: self.generator.as_small().unwrap().clone(),
                };
                Ok(GaloisArray::Small(rs.encode(m)?))
            }
            (FieldKind::Big(f), GaloisArray::Big(m)) => {
                let mut codeword: Vec<BigUint> = vec![BigUint::zero(); self.n];
                for (i, v) in m.values().iter().enumerate() {
                    codeword[self.n - self.k + i] = v.clone();
                }
                let msg_poly = big_poly_from_asc(&self.field, &codeword)?;
                let (_, remainder) = msg_poly.divmod(&self.generator)?;
                for (i, c) in remainder_coeffs_asc(&remainder)?.iter().enumerate() {
                    if i < self.n - self.k {
                        codeword[i] = f.sub(&BigUint::zero(), c);
                    }
                }
                Ok(GaloisArray::Big(crate::field::BigFieldArray::from_shape_vec(
                    f.clone(),
                    &[self.n],
                    codeword,
                )?))
            }
            _ => Err(GaloisError::FieldMismatch),
        }
    }

    /// Decode a received codeword.
    pub fn decode(&self, received: &GaloisArray) -> Result<GaloisArray> {
        field_decode::decode(self, received)
    }

    /// Compute syndromes of a received word.
    pub fn syndromes(&self, received: &GaloisArray) -> Result<GaloisArray> {
        field_decode::syndromes(self, received)
    }
}

/// BCH code over small or big Galois fields.
#[derive(Clone, Debug)]
pub struct FieldBch {
    pub(crate) inner: FieldReedSolomon,
    t: usize,
}

impl FieldBch {
    /// Create a BCH code with designed distance 2t+1.
    pub fn new(field: FieldKind, n: usize, k: usize, t: usize) -> Result<Self> {
        let alpha = field.primitive_element()?;
        let mut roots = Vec::with_capacity(2 * t);
        match &field {
            FieldKind::Small(f) => {
                let a = u64::try_from(&alpha).unwrap_or_else(|_| f.primitive_element());
                for i in 1..=(2 * t) {
                    roots.push(BigUint::from(f.pow(a, i as u64)));
                }
            }
            FieldKind::Big(f) => {
                for i in 1..=(2 * t) {
                    roots.push(f.pow(&alpha, i as u64));
                }
            }
        }
        let generator = poly_from_roots_values(&roots, &field)?;
        Ok(Self {
            inner: FieldReedSolomon {
                field,
                n,
                k,
                generator,
            },
            t,
        })
    }

    pub fn encode(&self, message: &GaloisArray) -> Result<GaloisArray> {
        self.inner.encode(message)
    }

    pub fn decode(&self, received: &GaloisArray) -> Result<GaloisArray> {
        self.inner.decode(received)
    }

    pub fn t(&self) -> usize {
        self.t
    }

    pub fn field(&self) -> &FieldKind {
        &self.inner.field
    }

    pub fn generator(&self) -> &FieldPoly {
        &self.inner.generator
    }

    pub fn n(&self) -> usize {
        self.inner.n
    }

    pub fn k(&self) -> usize {
        self.inner.k
    }
}

fn big_poly_from_asc(field: &FieldKind, coeffs: &[BigUint]) -> Result<FieldPoly> {
    match field {
        FieldKind::Big(f) => {
            let mut desc = coeffs.to_vec();
            desc.reverse();
            Ok(FieldPoly::Big(crate::poly::BigPoly::new(desc, f.clone())?))
        }
        FieldKind::Small(f) => {
            let values: Vec<u64> = coeffs
                .iter()
                .map(|c| {
                    u64::try_from(c).map_err(|_| GaloisError::InvalidElement {
                        value: 0,
                        characteristic: f.characteristic(),
                        degree: f.degree(),
                    })
                })
                .collect::<Result<_>>()?;
            Ok(FieldPoly::Small(crate::poly::Poly::new_asc(values, f.clone())?))
        }
    }
}

fn remainder_coeffs_asc(poly: &FieldPoly) -> Result<Vec<BigUint>> {
    match poly {
        FieldPoly::Small(p) => Ok(p.coeffs_asc().into_iter().map(BigUint::from).collect()),
        FieldPoly::Big(p) => Ok(p.coeffs_asc()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{BigGaloisField, GaloisField};

    #[test]
    fn field_rs_small_roundtrip() {
        let fk = FieldKind::Small(GaloisField::new(7, 1).unwrap());
        let rs = FieldReedSolomon::new(fk.clone(), 6, 4).unwrap();
        let msg = fk.array([1, 2, 3, 4]).unwrap();
        let codeword = rs.encode(&msg).unwrap();
        let decoded = rs.decode(&codeword).unwrap();
        assert_eq!(decoded.as_small().unwrap().values(), msg.as_small().unwrap().values());
    }

    #[test]
    fn field_rs_big_encode_decode() {
        let field = BigGaloisField::new(2, 2).unwrap();
        let fk = FieldKind::Big(field);
        let rs = FieldReedSolomon::new(fk.clone(), 3, 1).unwrap();
        let msg = fk.array([1u64]).unwrap();
        let codeword = rs.encode(&msg).unwrap();
        let decoded = rs.decode(&codeword).unwrap();
        assert_eq!(
            decoded.as_big().unwrap().values(),
            msg.as_big().unwrap().values()
        );
    }

    #[test]
    fn field_rs_big_corrects_single_error() {
        let field = BigGaloisField::new(2, 2).unwrap();
        let fk = FieldKind::Big(field.clone());
        let rs = FieldReedSolomon::new(fk.clone(), 3, 1).unwrap();
        let msg = fk.array([1u64]).unwrap();
        let mut codeword = rs.encode(&msg).unwrap();
        let values = codeword.as_big().unwrap().values();
        let err = field.add(&values[1], &BigUint::from(1u64));
        codeword = GaloisArray::Big(
            crate::field::BigFieldArray::from_shape_vec(field.clone(), &[3], {
                let mut v = values.clone();
                v[1] = err;
                v
            })
            .unwrap(),
        );
        let decoded = rs.decode(&codeword).unwrap();
        assert_eq!(
            decoded.as_big().unwrap().values(),
            msg.as_big().unwrap().values()
        );
    }

    #[test]
    fn field_bch_small_roundtrip() {
        let fk = FieldKind::Small(GaloisField::new(11, 1).unwrap());
        let bch = FieldBch::new(fk.clone(), 10, 6, 2).unwrap();
        let msg = fk.array([1, 2, 3, 4, 5, 6]).unwrap();
        let codeword = bch.encode(&msg).unwrap();
        let decoded = bch.decode(&codeword).unwrap();
        assert_eq!(
            decoded.as_small().unwrap().values(),
            msg.as_small().unwrap().values()
        );
    }

    #[test]
    fn field_bch_small_corrects_single_error() {
        let fk = FieldKind::Small(GaloisField::new(11, 1).unwrap());
        let bch = FieldBch::new(fk.clone(), 10, 6, 2).unwrap();
        let msg = fk.array([1, 2, 3, 4, 5, 6]).unwrap();
        let codeword = bch.encode(&msg).unwrap();
        let mut values = codeword.as_small().unwrap().values().to_vec();
        values[3] = fk.as_small().unwrap().add(values[3], 1);
        let received = fk.array(values).unwrap();
        let decoded = bch.decode(&received).unwrap();
        assert_eq!(
            decoded.as_small().unwrap().values(),
            msg.as_small().unwrap().values()
        );
    }

    #[test]
    fn field_bch_big_encode_decode() {
        let field = BigGaloisField::new(2, 2).unwrap();
        let fk = FieldKind::Big(field);
        let bch = FieldBch::new(fk.clone(), 3, 1, 1).unwrap();
        let msg = fk.array([1u64]).unwrap();
        let codeword = bch.encode(&msg).unwrap();
        let decoded = bch.decode(&codeword).unwrap();
        assert_eq!(
            decoded.as_big().unwrap().values(),
            msg.as_big().unwrap().values()
        );
    }

    #[test]
    fn field_bch_big_corrects_single_error() {
        let field = BigGaloisField::new(2, 2).unwrap();
        let fk = FieldKind::Big(field.clone());
        let bch = FieldBch::new(fk.clone(), 3, 1, 1).unwrap();
        let msg = fk.array([1u64]).unwrap();
        let codeword = bch.encode(&msg).unwrap();
        let values = codeword.as_big().unwrap().values();
        let err = field.add(&values[0], &BigUint::from(1u64));
        let received = GaloisArray::Big(
            crate::field::BigFieldArray::from_shape_vec(field, &[3], {
                let mut v = values.clone();
                v[0] = err;
                v
            })
            .unwrap(),
        );
        let decoded = bch.decode(&received).unwrap();
        assert_eq!(
            decoded.as_big().unwrap().values(),
            msg.as_big().unwrap().values()
        );
    }
}
