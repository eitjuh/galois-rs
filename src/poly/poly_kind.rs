//! Unified polynomials over small and big Galois fields.

use num_bigint::BigUint;
use num_traits::One;

use crate::error::{GaloisError, Result};
use crate::field::FieldKind;

use super::factor::factors as small_poly_factors;
use super::{BigPoly, Poly};

/// Polynomial over either a `GaloisField` or `BigGaloisField`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldPoly {
    Small(Poly),
    Big(BigPoly),
}

impl FieldPoly {
    pub fn from_u64(field: &FieldKind, coeffs: Vec<u64>) -> Result<Self> {
        match field {
            FieldKind::Small(f) => Ok(Self::Small(Poly::new(coeffs, f.clone())?)),
            FieldKind::Big(f) => Ok(Self::Big(BigPoly::from_u64(coeffs, f.clone())?)),
        }
    }

    pub fn from_u64_asc(field: &FieldKind, coeffs: Vec<u64>) -> Result<Self> {
        match field {
            FieldKind::Small(f) => Ok(Self::Small(Poly::new_asc(coeffs, f.clone())?)),
            FieldKind::Big(f) => Ok(Self::Big(BigPoly::from_u64_asc(coeffs, f.clone())?)),
        }
    }

    pub fn zero(field: &FieldKind) -> Result<Self> {
        match field {
            FieldKind::Small(f) => Ok(Self::Small(Poly::zero(f.clone())?)),
            FieldKind::Big(f) => Ok(Self::Big(BigPoly::zero(f.clone())?)),
        }
    }

    pub fn one(field: &FieldKind) -> Result<Self> {
        match field {
            FieldKind::Small(f) => Ok(Self::Small(Poly::one(f.clone())?)),
            FieldKind::Big(f) => Ok(Self::Big(BigPoly::one(f.clone())?)),
        }
    }

    pub fn x(field: &FieldKind) -> Result<Self> {
        match field {
            FieldKind::Small(f) => Ok(Self::Small(Poly::x(f.clone())?)),
            FieldKind::Big(f) => Ok(Self::Big(BigPoly::x(f.clone())?)),
        }
    }

    pub fn degree(&self) -> isize {
        match self {
            Self::Small(p) => p.degree(),
            Self::Big(p) => p.degree(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.degree() < 0
    }

    pub fn is_small(&self) -> bool {
        matches!(self, Self::Small(_))
    }

    pub fn as_small(&self) -> Option<&Poly> {
        match self {
            Self::Small(p) => Some(p),
            Self::Big(_) => None,
        }
    }

    pub fn as_big(&self) -> Option<&BigPoly> {
        match self {
            Self::Small(_) => None,
            Self::Big(p) => Some(p),
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => Ok(Self::Small(a.add(b)?)),
            (Self::Big(a), Self::Big(b)) => Ok(Self::Big(a.add(b)?)),
            _ => Err(GaloisError::PolynomialFieldMismatch),
        }
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => Ok(Self::Small(a.sub(b)?)),
            (Self::Big(a), Self::Big(b)) => Ok(Self::Big(a.sub(b)?)),
            _ => Err(GaloisError::PolynomialFieldMismatch),
        }
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => Ok(Self::Small(a.mul(b)?)),
            (Self::Big(a), Self::Big(b)) => Ok(Self::Big(a.mul(b)?)),
            _ => Err(GaloisError::PolynomialFieldMismatch),
        }
    }

    pub fn divmod(&self, divisor: &Self) -> Result<(Self, Self)> {
        match (self, divisor) {
            (Self::Small(a), Self::Small(b)) => {
                let (q, r) = a.divmod(b)?;
                Ok((Self::Small(q), Self::Small(r)))
            }
            (Self::Big(a), Self::Big(b)) => {
                let (q, r) = a.divmod(b)?;
                Ok((Self::Big(q), Self::Big(r)))
            }
            _ => Err(GaloisError::PolynomialFieldMismatch),
        }
    }

    pub fn rem(&self, divisor: &Self) -> Result<Self> {
        Ok(self.divmod(divisor)?.1)
    }

    pub fn div(&self, divisor: &Self) -> Result<Self> {
        Ok(self.divmod(divisor)?.0)
    }

    pub fn neg(&self) -> Result<Self> {
        match self {
            Self::Small(p) => Ok(Self::Small(p.neg()?)),
            Self::Big(p) => Ok(Self::Big(p.neg()?)),
        }
    }

    pub fn derivative(&self, k: u32) -> Result<Self> {
        match self {
            Self::Small(p) => Ok(Self::Small(p.derivative(k)?)),
            Self::Big(p) => Ok(Self::Big(p.derivative(k)?)),
        }
    }

    pub fn evaluate_u64(&self, x: u64) -> Result<u64> {
        match self {
            Self::Small(p) => p.evaluate(x),
            Self::Big(p) => {
                let result = p.evaluate(&BigUint::from(x))?;
                u64::try_from(result).map_err(|_| GaloisError::InvalidElement {
                    value: 0,
                    characteristic: p.field().characteristic(),
                    degree: p.field().degree(),
                })
            }
        }
    }

    pub fn format_poly(&self, var: &str) -> String {
        match self {
            Self::Small(p) => p.format_poly(var),
            Self::Big(p) => p.format_poly(var),
        }
    }
}

/// Full factorization into irreducibles for either polynomial kind.
pub fn field_poly_factors(poly: &FieldPoly) -> Result<Vec<FieldPoly>> {
    match poly {
        FieldPoly::Small(p) => small_poly_factors(p)
            .map(|facs| facs.into_iter().map(FieldPoly::Small).collect()),
        FieldPoly::Big(p) => crate::poly::big_factors(p)
            .map(|facs| facs.into_iter().map(FieldPoly::Big).collect()),
    }
}

/// Roots of a field polynomial.
pub fn field_poly_roots(poly: &FieldPoly) -> Result<crate::field::GaloisArray> {
    match poly {
        FieldPoly::Small(p) => {
            let roots = crate::poly::poly_roots(p)?;
            Ok(crate::field::GaloisArray::Small(crate::field::FieldArray::new(
                p.field().clone(),
                roots,
            )))
        }
        FieldPoly::Big(p) => {
            let roots = crate::poly::big_roots(p)?;
            Ok(crate::field::GaloisArray::Big(crate::field::BigFieldArray::new(
                p.field().clone(),
                roots,
            )?))
        }
    }
}

/// Construct polynomial ∏(x - root) over the given field kind.
pub fn poly_from_roots_values(roots: &[BigUint], field: &FieldKind) -> Result<FieldPoly> {
    match field {
        FieldKind::Small(f) => {
            let mut poly = Poly::one(f.clone())?;
            for root in roots {
                let root_u64 = u64::try_from(root).map_err(|_| GaloisError::InvalidElement {
                    value: 0,
                    characteristic: f.characteristic(),
                    degree: f.degree(),
                })?;
                let root = f.validate_element(root_u64)?;
                let linear = Poly::new(vec![1, f.neg(root)], f.clone())?;
                poly = poly.mul(&linear)?;
            }
            Ok(FieldPoly::Small(poly))
        }
        FieldKind::Big(f) => {
            let mut poly = BigPoly::one(f.clone())?;
            for root in roots {
                let root = f.validate(root)?;
                let neg = f.neg(&root);
                let linear = BigPoly::new(vec![BigUint::from(1u64), neg], f.clone())?;
                poly = poly.mul(&linear)?;
            }
            Ok(FieldPoly::Big(poly))
        }
    }
}

/// Construct polynomial ∏(x - root) over the given field kind (u64 roots).
pub fn poly_from_roots_kind(roots: &[u64], field: &FieldKind) -> Result<FieldPoly> {
    poly_from_roots_values(
        &roots.iter().map(|&r| BigUint::from(r)).collect::<Vec<_>>(),
        field,
    )
}

/// Lagrange interpolating polynomial over small or big field arrays.
pub fn lagrange_poly_array(x: &crate::field::GaloisArray, y: &crate::field::GaloisArray) -> Result<FieldPoly> {
    if x.len() != y.len() {
        return Err(GaloisError::LengthMismatch);
    }
    match (x, y) {
        (crate::field::GaloisArray::Small(xs), crate::field::GaloisArray::Small(ys)) => {
            Ok(FieldPoly::Small(super::field_poly::lagrange_poly(xs, ys)?))
        }
        (crate::field::GaloisArray::Big(xs), crate::field::GaloisArray::Big(ys)) => {
            lagrange_poly_big(xs, ys)
        }
        _ => Err(GaloisError::FieldMismatch),
    }
}

fn lagrange_poly_big(
    x: &crate::field::BigFieldArray,
    y: &crate::field::BigFieldArray,
) -> Result<FieldPoly> {
    let field = x.field().clone();
    let n = x.len();
    let mut result = BigPoly::zero(field.clone())?;

    for i in 0..n {
        let xi = &x.values()[i];
        let yi = &y.values()[i];
        let mut basis = BigPoly::one(field.clone())?;
        let mut denom = BigUint::one();
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = &x.values()[j];
            let linear = BigPoly::new(
                vec![BigUint::one(), field.neg(xj)],
                field.clone(),
            )?;
            basis = basis.mul(&linear)?;
            denom = field.mul(&denom, &field.sub(xi, xj));
        }
        let scale = field.div(yi, &denom)?;
        let term = basis.mul_scalar(&scale)?;
        result = result.add(&term)?;
    }
    Ok(FieldPoly::Big(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_poly_factors_gf2_2() {
        let field = crate::field::BigGaloisField::new(2, 2).unwrap();
        let fk = crate::field::FieldKind::Big(field);
        let p = FieldPoly::from_u64(&fk, vec![1, 1, 1]).unwrap();
        let facs = field_poly_factors(&p).unwrap();
        assert!(!facs.is_empty());
    }

    #[test]
    fn lagrange_poly_array_gf7() {
        let fk = crate::field::FieldKind::Small(crate::field::GaloisField::new(7, 1).unwrap());
        let x = fk.array([1, 2, 4]).unwrap();
        let y = fk.array([1, 0, 1]).unwrap();
        let p = lagrange_poly_array(&x, &y).unwrap();
        assert_eq!(p.degree(), 2);
        assert_eq!(p.evaluate_u64(1).unwrap(), 1);
        assert_eq!(p.evaluate_u64(2).unwrap(), 0);
        assert_eq!(p.evaluate_u64(4).unwrap(), 1);
    }
}
