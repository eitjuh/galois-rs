use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::error::{GaloisError, Result};

use super::factory::GaloisField;

/// Element display representation, matching Python galois `repr` options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ElementRepr {
    #[default]
    Int,
    Poly,
    Power,
}

/// A single element of a Galois field.
#[derive(Clone, PartialEq, Eq)]
pub struct FieldElement {
    pub(crate) field: GaloisField,
    pub(crate) value: u64,
}

impl FieldElement {
    pub(crate) fn new(field: GaloisField, value: u64) -> Self {
        Self { field, value }
    }

    pub fn field(&self) -> &GaloisField {
        &self.field
    }

    pub fn value(&self) -> u64 {
        self.value
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        ensure_same_field(&self.field, &other.field)?;
        Ok(Self::new(
            self.field.clone(),
            self.field.add(self.value, other.value),
        ))
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        ensure_same_field(&self.field, &other.field)?;
        Ok(Self::new(
            self.field.clone(),
            self.field.sub(self.value, other.value),
        ))
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        ensure_same_field(&self.field, &other.field)?;
        Ok(Self::new(
            self.field.clone(),
            self.field.mul(self.value, other.value),
        ))
    }

    pub fn div(&self, other: &Self) -> Result<Self> {
        ensure_same_field(&self.field, &other.field)?;
        Ok(Self::new(
            self.field.clone(),
            self.field.div(self.value, other.value)?,
        ))
    }

    pub fn pow(&self, exp: u64) -> Self {
        Self::new(self.field.clone(), self.field.pow(self.value, exp))
    }

    pub fn inverse(&self) -> Result<Self> {
        self.div(&Self::new(self.field.clone(), 1))
    }

    pub fn format_with(&self, repr: ElementRepr) -> String {
        self.field.format_element(self.value, repr)
    }
}

impl fmt::Debug for FieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GF({}^{})({})",
            self.field.characteristic(),
            self.field.degree(),
            self.field
                .format_element(self.value, self.field.default_repr())
        )
    }
}

impl fmt::Display for FieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.field
                .format_element(self.value, self.field.default_repr())
        )
    }
}

impl Add for &FieldElement {
    type Output = Result<FieldElement>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl Sub for &FieldElement {
    type Output = Result<FieldElement>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(rhs)
    }
}

impl Mul for &FieldElement {
    type Output = Result<FieldElement>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

impl Div for &FieldElement {
    type Output = Result<FieldElement>;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(rhs)
    }
}

impl Neg for &FieldElement {
    type Output = FieldElement;

    fn neg(self) -> Self::Output {
        FieldElement::new(self.field.clone(), self.field.neg(self.value))
    }
}

fn ensure_same_field(a: &GaloisField, b: &GaloisField) -> Result<()> {
    if a.order() != b.order()
        || a.characteristic() != b.characteristic()
        || a.degree() != b.degree()
    {
        return Err(GaloisError::FieldMismatch);
    }
    Ok(())
}
