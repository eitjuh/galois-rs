use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use ndarray::{Array, ArrayD, IxDyn};

use crate::error::{GaloisError, Result};

use super::broadcast::{broadcast_binary, broadcast_binary_result, broadcast_shapes, total_size};
use super::element::FieldElement;
use super::factory::GaloisField;

/// Multi-dimensional array of Galois field elements (NumPy-style).
#[derive(Clone, PartialEq, Eq)]
pub struct FieldArray {
    field: GaloisField,
    data: ArrayD<u64>,
}

impl FieldArray {
    pub(crate) fn new(field: GaloisField, values: Vec<u64>) -> Self {
        let len = values.len();
        Self {
            field,
            data: Array::from_shape_vec(IxDyn(&[len]), values)
                .unwrap_or_else(|_| ArrayD::zeros(IxDyn(&[0])))
                .into_dyn(),
        }
    }

    /// Create from explicit shape and values (row-major).
    pub fn from_shape_vec(field: GaloisField, shape: &[usize], values: Vec<u64>) -> Result<Self> {
        let len = values.len();
        if total_size(shape) != len {
            return Err(GaloisError::ShapeMismatch {
                expected: shape.to_vec(),
                actual: vec![len],
            });
        }
        for &v in &values {
            field.validate_element(v)?;
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

    /// Create a zero-filled array of given shape.
    pub fn zeros(field: GaloisField, shape: &[usize]) -> Self {
        Self {
            field,
            data: ArrayD::zeros(IxDyn(shape)),
        }
    }

    /// Create a one-filled array of given shape.
    pub fn ones(field: GaloisField, shape: &[usize]) -> Self {
        Self {
            field,
            data: ArrayD::from_elem(IxDyn(shape), 1),
        }
    }

    pub fn field(&self) -> &GaloisField {
        &self.field
    }

    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Flat slice of values (row-major).
    pub fn values(&self) -> &[u64] {
        self.data.as_slice().unwrap_or(&[])
    }

    pub fn data(&self) -> &ArrayD<u64> {
        &self.data
    }

    pub fn elements(&self) -> Vec<FieldElement> {
        self.values()
            .iter()
            .map(|&v| FieldElement::new(self.field.clone(), v))
            .collect()
    }

    pub fn get(&self, index: usize) -> Option<FieldElement> {
        self.values()
            .get(index)
            .map(|&v| FieldElement::new(self.field.clone(), v))
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
        let (shape, values) = broadcast_binary(
            &self.shape(),
            self.values(),
            &other.shape(),
            other.values(),
            |a, b| field.add(a, b),
        )?;
        Self::from_shape_vec(field, &shape, values)
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        ensure_same_field(self, other)?;
        let field = self.field.clone();
        let (shape, values) = broadcast_binary(
            &self.shape(),
            self.values(),
            &other.shape(),
            other.values(),
            |a, b| field.sub(a, b),
        )?;
        Self::from_shape_vec(field, &shape, values)
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        ensure_same_field(self, other)?;
        let field = self.field.clone();
        let (shape, values) = broadcast_binary(
            &self.shape(),
            self.values(),
            &other.shape(),
            other.values(),
            |a, b| field.mul(a, b),
        )?;
        Self::from_shape_vec(field, &shape, values)
    }

    pub fn div(&self, other: &Self) -> Result<Self> {
        ensure_same_field(self, other)?;
        let field = self.field.clone();
        let (shape, values) = broadcast_binary_result(
            &self.shape(),
            self.values(),
            &other.shape(),
            other.values(),
            |a, b| field.div(a, b),
        )?;
        Self::from_shape_vec(field, &shape, values)
    }

    pub fn neg(&self) -> Self {
        let values: Vec<u64> = self.values().iter().map(|&v| self.field.neg(v)).collect();
        Self::new(self.field.clone(), values)
    }

    pub fn pow(&self, exp: u64) -> Self {
        let values: Vec<u64> = self
            .values()
            .iter()
            .map(|&v| self.field.pow(v, exp))
            .collect();
        Self::new(self.field.clone(), values)
    }

    pub fn sqrt(&self) -> Result<Self> {
        let values: Vec<u64> = self
            .values()
            .iter()
            .map(|&v| self.field.sqrt(v))
            .collect::<Result<_>>()?;
        Ok(Self::new(self.field.clone(), values))
    }

    /// Discrete log base the field's primitive element (returns integers, not field elements).
    pub fn log(&self) -> Result<Vec<u64>> {
        self.values()
            .iter()
            .map(|&v| {
                if v == 0 {
                    return Err(GaloisError::DivisionByZero {
                        characteristic: self.field.characteristic(),
                        degree: self.field.degree(),
                    });
                }
                Ok(self.field.log(v)? as u64)
            })
            .collect()
    }

    pub fn multiplicative_order(&self) -> Result<Vec<u64>> {
        self.values()
            .iter()
            .map(|&v| self.field.multiplicative_order(v))
            .collect()
    }

    pub fn scalar_add(&self, scalar: u64) -> Result<Self> {
        let scalar = self.field.validate_element(scalar)?;
        let values: Vec<u64> = self
            .values()
            .iter()
            .map(|&a| self.field.add(a, scalar))
            .collect();
        Ok(Self::new(self.field.clone(), values))
    }

    pub fn scalar_mul(&self, scalar: u64) -> Result<Self> {
        let scalar = self.field.validate_element(scalar)?;
        let values: Vec<u64> = self
            .values()
            .iter()
            .map(|&a| self.field.mul(a, scalar))
            .collect();
        Ok(Self::new(self.field.clone(), values))
    }

    /// Add with broadcasting against a scalar.
    pub fn add_scalar(&self, scalar: u64) -> Result<Self> {
        let scalar_arr = Self::new(self.field.clone(), vec![self.field.validate_element(scalar)?]);
        self.add(&scalar_arr)
    }

    pub fn mul_scalar_broadcast(&self, scalar: u64) -> Result<Self> {
        let scalar_arr = Self::new(self.field.clone(), vec![self.field.validate_element(scalar)?]);
        self.mul(&scalar_arr)
    }
}

impl fmt::Debug for FieldArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GF({}^{}) shape={:?} {:?}",
            self.field.characteristic(),
            self.field.degree(),
            self.shape(),
            self.values()
        )
    }
}

impl Add for &FieldArray {
    type Output = Result<FieldArray>;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl Sub for &FieldArray {
    type Output = Result<FieldArray>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(rhs)
    }
}

impl Mul for &FieldArray {
    type Output = Result<FieldArray>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

impl Div for &FieldArray {
    type Output = Result<FieldArray>;
    fn div(self, rhs: Self) -> Self::Output {
        self.div(rhs)
    }
}

impl Neg for &FieldArray {
    type Output = FieldArray;
    fn neg(self) -> Self::Output {
        self.neg()
    }
}

fn ensure_same_field(a: &FieldArray, b: &FieldArray) -> Result<()> {
    if a.field.order() != b.field.order()
        || a.field.characteristic() != b.field.characteristic()
        || a.field.degree() != b.field.degree()
    {
        return Err(GaloisError::FieldMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GaloisField;

    #[test]
    fn broadcast_add_scalar() {
        let gf = GaloisField::new(3, 5).unwrap();
        let x = gf.array([236, 87, 38, 112]).unwrap();
        let two = gf.array([2]).unwrap();
        let result = x.add(&two).unwrap();
        assert_eq!(result.values(), &[235, 89, 37, 111]);
    }

    #[test]
    fn reshape_2d() {
        let gf = GaloisField::new(5, 1).unwrap();
        let a = FieldArray::from_shape_vec(gf, &[2, 2], vec![1, 2, 3, 4]).unwrap();
        assert_eq!(a.shape(), vec![2, 2]);
        let flat = a.reshape(&[4]).unwrap();
        assert_eq!(flat.shape(), vec![4]);
    }

    #[test]
    fn sqrt_gf3_5() {
        let gf = GaloisField::new(3, 5).unwrap();
        let x = gf.array([236, 87, 38, 112]).unwrap();
        let roots = x.sqrt().unwrap();
        // verify: roots^2 == x
        let check = roots.mul(&roots).unwrap();
        assert_eq!(check.values(), x.values());
    }
}
