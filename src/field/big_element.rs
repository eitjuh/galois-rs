//! Field elements over `BigGaloisField`.

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use num_bigint::BigUint;

use crate::error::{GaloisError, Result};

use super::big_field::BigGaloisField;
use super::element::{ElementRepr, FieldElement};

/// A single element of a `BigGaloisField`.
#[derive(Clone, PartialEq, Eq)]
pub struct BigFieldElement {
    pub(crate) field: BigGaloisField,
    pub(crate) value: BigUint,
}

impl BigFieldElement {
    pub fn new(field: BigGaloisField, value: BigUint) -> Result<Self> {
        let value = field.validate(&value)?;
        Ok(Self { field, value })
    }

    pub fn from_u64(field: BigGaloisField, value: u64) -> Result<Self> {
        Self::new(field, BigUint::from(value))
    }

    pub fn field(&self) -> &BigGaloisField {
        &self.field
    }

    pub fn value(&self) -> &BigUint {
        &self.value
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        ensure_same_field(&self.field, &other.field)?;
        Ok(Self {
            field: self.field.clone(),
            value: self.field.add(&self.value, &other.value),
        })
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        ensure_same_field(&self.field, &other.field)?;
        Ok(Self {
            field: self.field.clone(),
            value: self.field.sub(&self.value, &other.value),
        })
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        ensure_same_field(&self.field, &other.field)?;
        Ok(Self {
            field: self.field.clone(),
            value: self.field.mul(&self.value, &other.value),
        })
    }

    pub fn div(&self, other: &Self) -> Result<Self> {
        ensure_same_field(&self.field, &other.field)?;
        Ok(Self {
            field: self.field.clone(),
            value: self.field.div(&self.value, &other.value)?,
        })
    }

    pub fn pow(&self, exp: u64) -> Self {
        Self {
            field: self.field.clone(),
            value: self.field.pow(&self.value, exp),
        }
    }

    pub fn inverse(&self) -> Result<Self> {
        self.div(&Self::new(self.field.clone(), BigUint::from(1u64))?)
    }

    pub fn format_with(&self, repr: ElementRepr) -> String {
        self.field.format_element(&self.value, repr)
    }
}

impl fmt::Debug for BigFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BigGF({}^{})({})",
            self.field.characteristic(),
            self.field.degree(),
            self.format_with(ElementRepr::Int)
        )
    }
}

impl fmt::Display for BigFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_with(ElementRepr::Int))
    }
}

impl Add for &BigFieldElement {
    type Output = Result<BigFieldElement>;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl Sub for &BigFieldElement {
    type Output = Result<BigFieldElement>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(rhs)
    }
}

impl Mul for &BigFieldElement {
    type Output = Result<BigFieldElement>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

impl Div for &BigFieldElement {
    type Output = Result<BigFieldElement>;
    fn div(self, rhs: Self) -> Self::Output {
        self.div(rhs)
    }
}

impl Neg for &BigFieldElement {
    type Output = BigFieldElement;
    fn neg(self) -> Self::Output {
        BigFieldElement {
            field: self.field.clone(),
            value: self.field.neg(&self.value),
        }
    }
}

fn ensure_same_field(a: &BigGaloisField, b: &BigGaloisField) -> Result<()> {
    if a.characteristic() != b.characteristic() || a.degree() != b.degree() {
        return Err(GaloisError::FieldMismatch);
    }
    Ok(())
}

/// Field element over either a `GaloisField` or `BigGaloisField`.
#[derive(Clone, PartialEq, Eq)]
pub enum GaloisElement {
    Small(FieldElement),
    Big(BigFieldElement),
}

impl GaloisElement {
    pub fn from_u64(field: &super::FieldKind, value: u64) -> Result<Self> {
        match field {
            super::FieldKind::Small(f) => Ok(Self::Small(f.element(value)?)),
            super::FieldKind::Big(f) => Ok(Self::Big(BigFieldElement::from_u64(f.clone(), value)?)),
        }
    }

    pub fn from_big(field: &BigGaloisField, value: BigUint) -> Result<Self> {
        Ok(Self::Big(BigFieldElement::new(field.clone(), value)?))
    }

    pub fn is_small(&self) -> bool {
        matches!(self, Self::Small(_))
    }

    pub fn as_small(&self) -> Option<&FieldElement> {
        match self {
            Self::Small(e) => Some(e),
            Self::Big(_) => None,
        }
    }

    pub fn as_big(&self) -> Option<&BigFieldElement> {
        match self {
            Self::Small(_) => None,
            Self::Big(e) => Some(e),
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
}

impl fmt::Debug for GaloisElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Small(e) => write!(f, "{e:?}"),
            Self::Big(e) => write!(f, "{e:?}"),
        }
    }
}

impl fmt::Display for GaloisElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Small(e) => write!(f, "{e}"),
            Self::Big(e) => write!(f, "{e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::GF2;

    #[test]
    fn big_element_gf2_100() {
        let field = BigGaloisField::new(2, 100).unwrap();
        let a = BigFieldElement::from_u64(field.clone(), 1).unwrap();
        let b = BigFieldElement::from_u64(field, 1).unwrap();
        let sum = a.add(&b).unwrap();
        assert_eq!(sum.value(), &BigUint::from(0u64));
    }

    #[test]
    fn galois_element_from_field_kind() {
        let fk = GF2(100).unwrap();
        let e = GaloisElement::from_u64(&fk, 1).unwrap();
        assert!(!e.is_small());
    }
}
