//! Normal elements of extension fields.

use crate::error::{GaloisError, Result};
use crate::field::GaloisField;
use crate::poly::{poly_gcd, Poly};
use crate::prime::factorize;

/// Find a normal element of GF(q^m).
pub fn normal_element(field: &GaloisField) -> Result<u64> {
    let q = field.order();
    let m = field.degree();
    let p = field.characteristic();

    for candidate in 1..q {
        if is_normal_element(candidate, field)? {
            return Ok(candidate);
        }
        // avoid infinite loop on large fields - cap search for huge q
        if candidate > 10_000 && m > 1 {
            break;
        }
    }
    Err(GaloisError::NoPrimitiveElement {
        characteristic: p,
        degree: m,
    })
}

/// Whether `element` is a normal element of the extension field.
pub fn is_normal_element(element: u64, field: &GaloisField) -> Result<bool> {
    let m = field.degree();
    if m == 1 {
        return Ok(element != 0);
    }

    let q = field.order();
    let (factors, _) = factorize(q - 1).unwrap_or((vec![q - 1], vec![1]));

    let x = Poly::x(field.clone())?;
    let g = Poly::new(vec![field.validate_element(element)?], field.clone())?;

    for &f in &factors {
        let exp = (q - 1) / f;
        let term = g.pow_poly(exp)?;
        let diff = term.sub(&x)?;
        let f = field.irreducible_as_poly()?;
        let gcd = poly_gcd(&diff, &f)?;
        if !gcd.is_one() {
            return Ok(false);
        }
    }
    Ok(true)
}

trait FieldPolyPow {
    fn pow_poly(&self, exp: u64) -> Result<Poly>;
}

impl FieldPolyPow for Poly {
    fn pow_poly(&self, mut exp: u64) -> Result<Poly> {
        let mut result = Poly::one(self.field().clone())?;
        let mut base = self.clone();
        while exp > 0 {
            if exp % 2 == 1 {
                result = result.mul(&base)?;
            }
            base = base.mul(&base)?;
            exp /= 2;
        }
        Ok(result)
    }
}
