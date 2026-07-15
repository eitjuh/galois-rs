//! Unified field arrays for small and big Galois fields.

use num_bigint::BigUint;

use crate::error::{GaloisError, Result};
use crate::field::{BigFieldArray, BigGaloisField, FieldArray, FieldKind};

/// Array over either a `GaloisField` or `BigGaloisField`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GaloisArray {
    Small(FieldArray),
    Big(BigFieldArray),
}

impl GaloisArray {
    pub fn from_u64(field: &FieldKind, values: Vec<u64>) -> Result<Self> {
        match field {
            FieldKind::Small(f) => Ok(Self::Small(FieldArray::new(f.clone(), values))),
            FieldKind::Big(f) => Ok(Self::Big(BigFieldArray::from_u64(f.clone(), values)?)),
        }
    }

    pub fn from_big(field: &BigGaloisField, values: Vec<BigUint>) -> Result<Self> {
        Ok(Self::Big(BigFieldArray::new(field.clone(), values)?))
    }

    pub fn zeros(field: &FieldKind, shape: &[usize]) -> Self {
        match field {
            FieldKind::Small(f) => Self::Small(FieldArray::zeros(f.clone(), shape)),
            FieldKind::Big(f) => Self::Big(BigFieldArray::zeros(f.clone(), shape)),
        }
    }

    pub fn ones(field: &FieldKind, shape: &[usize]) -> Self {
        match field {
            FieldKind::Small(f) => Self::Small(FieldArray::ones(f.clone(), shape)),
            FieldKind::Big(f) => Self::Big(BigFieldArray::ones(f.clone(), shape)),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Small(a) => a.len(),
            Self::Big(a) => a.len(),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            Self::Small(a) => a.shape(),
            Self::Big(a) => a.shape(),
        }
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        match self {
            Self::Small(a) => Ok(Self::Small(a.reshape(shape)?)),
            Self::Big(a) => Ok(Self::Big(a.reshape(shape)?)),
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => Ok(Self::Small(a.add(b)?)),
            (Self::Big(a), Self::Big(b)) => Ok(Self::Big(a.add(b)?)),
            _ => Err(GaloisError::FieldMismatch),
        }
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => Ok(Self::Small(a.sub(b)?)),
            (Self::Big(a), Self::Big(b)) => Ok(Self::Big(a.sub(b)?)),
            _ => Err(GaloisError::FieldMismatch),
        }
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => Ok(Self::Small(a.mul(b)?)),
            (Self::Big(a), Self::Big(b)) => Ok(Self::Big(a.mul(b)?)),
            _ => Err(GaloisError::FieldMismatch),
        }
    }

    pub fn div(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => Ok(Self::Small(a.div(b)?)),
            (Self::Big(a), Self::Big(b)) => Ok(Self::Big(a.div(b)?)),
            _ => Err(GaloisError::FieldMismatch),
        }
    }

    pub fn is_small(&self) -> bool {
        matches!(self, Self::Small(_))
    }

    pub fn as_small(&self) -> Option<&FieldArray> {
        match self {
            Self::Small(a) => Some(a),
            Self::Big(_) => None,
        }
    }

    pub fn as_big(&self) -> Option<&BigFieldArray> {
        match self {
            Self::Small(_) => None,
            Self::Big(a) => Some(a),
        }
    }

    pub fn field(&self) -> FieldKind {
        match self {
            Self::Small(a) => FieldKind::Small(a.field().clone()),
            Self::Big(a) => FieldKind::Big(a.field().clone()),
        }
    }
}
