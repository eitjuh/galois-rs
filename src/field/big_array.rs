//! Multi-dimensional arrays over `BigGaloisField`.

use std::fmt;

use ndarray::{Array, ArrayD, IxDyn};
use num_bigint::BigUint;

use crate::error::{GaloisError, Result};

use super::big_field::BigGaloisField;
use super::broadcast::{broadcast_big, broadcast_big_result, total_size};

/// Array of elements over a `BigGaloisField`.
#[derive(Clone, PartialEq, Eq)]
pub struct BigFieldArray {
    field: BigGaloisField,
    data: ArrayD<BigUint>,
}

impl BigFieldArray {
    pub fn new(field: BigGaloisField, values: Vec<BigUint>) -> Result<Self> {
        for v in &values {
            field.validate(v)?;
        }
        let len = values.len();
        Ok(Self {
            field,
            data: Array::from_shape_vec(IxDyn(&[len]), values)
                .map_err(|_| GaloisError::ShapeMismatch {
                    expected: vec![len],
                    actual: vec![0],
                })?
                .into_dyn(),
        })
    }

    pub fn from_u64(field: BigGaloisField, values: Vec<u64>) -> Result<Self> {
        Self::new(
            field,
            values.into_iter().map(BigUint::from).collect(),
        )
    }

    pub fn from_shape_vec(
        field: BigGaloisField,
        shape: &[usize],
        values: Vec<BigUint>,
    ) -> Result<Self> {
        if total_size(shape) != values.len() {
            return Err(GaloisError::ShapeMismatch {
                expected: shape.to_vec(),
                actual: vec![values.len()],
            });
        }
        let len = values.len();
        for v in &values {
            field.validate(v)?;
        }
        Ok(Self {
            field,
            data: Array::from_shape_vec(IxDyn(shape), values)
                .map_err(|_| GaloisError::ShapeMismatch {
                    expected: shape.to_vec(),
                    actual: vec![len],
                })?
                .into_dyn(),
        })
    }

    pub fn zeros(field: BigGaloisField, shape: &[usize]) -> Self {
        Self {
            field,
            data: ArrayD::zeros(IxDyn(shape)),
        }
    }

    pub fn ones(field: BigGaloisField, shape: &[usize]) -> Self {
        Self {
            field,
            data: ArrayD::from_elem(IxDyn(shape), BigUint::from(1u64)),
        }
    }

    pub fn field(&self) -> &BigGaloisField {
        &self.field
    }

    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn values(&self) -> Vec<BigUint> {
        self.data.iter().cloned().collect()
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        if total_size(shape) != self.len() {
            return Err(GaloisError::ShapeMismatch {
                expected: shape.to_vec(),
                actual: self.shape(),
            });
        }
        Ok(Self {
            field: self.field.clone(),
            data: self
                .data
                .clone()
                .into_shape_with_order(IxDyn(shape))
                .map_err(|_| GaloisError::ShapeMismatch {
                    expected: shape.to_vec(),
                    actual: self.shape(),
                })?,
        })
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        ensure_same_field(self, other)?;
        let field = self.field.clone();
        let (shape, values) = broadcast_big(
            &self.shape(),
            &self.values(),
            &other.shape(),
            &other.values(),
            |a, b| field.add(a, b),
        )?;
        Self::from_shape_vec(field, &shape, values)
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        ensure_same_field(self, other)?;
        let field = self.field.clone();
        let (shape, values) = broadcast_big(
            &self.shape(),
            &self.values(),
            &other.shape(),
            &other.values(),
            |a, b| field.sub(a, b),
        )?;
        Self::from_shape_vec(field, &shape, values)
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        ensure_same_field(self, other)?;
        let field = self.field.clone();
        let (shape, values) = broadcast_big(
            &self.shape(),
            &self.values(),
            &other.shape(),
            &other.values(),
            |a, b| field.mul(a, b),
        )?;
        Self::from_shape_vec(field, &shape, values)
    }

    pub fn div(&self, other: &Self) -> Result<Self> {
        ensure_same_field(self, other)?;
        let field = self.field.clone();
        let (shape, values) = broadcast_big_result(
            &self.shape(),
            &self.values(),
            &other.shape(),
            &other.values(),
            |a, b| field.div(a, b),
        )?;
        Self::from_shape_vec(field, &shape, values)
    }

    pub fn neg(&self) -> Self {
        let out: Vec<BigUint> = self.data.iter().map(|v| self.field.neg(v)).collect();
        Self::from_shape_vec(self.field.clone(), &self.shape(), out).unwrap()
    }
}

fn ensure_same_field(a: &BigFieldArray, b: &BigFieldArray) -> Result<()> {
    if a.field.characteristic() != b.field.characteristic()
        || a.field.degree() != b.field.degree()
    {
        return Err(GaloisError::FieldMismatch);
    }
    Ok(())
}

impl fmt::Debug for BigFieldArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BigFieldArray")
            .field("field", &self.field.name())
            .field("shape", &self.shape())
            .field("len", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};

    use super::*;

    #[test]
    fn big_array_arithmetic_gf2_100() {
        let field = BigGaloisField::new(2, 100).unwrap();
        let a = BigFieldArray::from_u64(field.clone(), vec![1, 0, 1]).unwrap();
        let b = BigFieldArray::from_u64(field.clone(), vec![1, 1, 0]).unwrap();
        let sum = a.add(&b).unwrap();
        assert_eq!(sum.values(), vec![BigUint::zero(), BigUint::one(), BigUint::one()]);
        let prod = a.mul(&b).unwrap();
        assert_eq!(prod.values()[0], BigUint::one());
    }

    #[test]
    fn big_array_broadcast_scalar() {
        let field = BigGaloisField::new(2, 100).unwrap();
        let a = BigFieldArray::from_u64(field.clone(), vec![1, 0, 1, 0]).unwrap();
        let s = BigFieldArray::from_u64(field.clone(), vec![1]).unwrap();
        let sum = a.add(&s).unwrap();
        assert_eq!(sum.shape(), vec![4]);
        assert_eq!(sum.values()[0], BigUint::zero());
    }
}
