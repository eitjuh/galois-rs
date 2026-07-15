//! Default irreducible polynomials using the Conway polynomial database.

use crate::databases::conway_poly_lookup;
use crate::error::{GaloisError, Result};
use crate::poly::PrimePoly;

/// Returns the default irreducible polynomial for GF(p^m).
///
/// Uses Frank Luebeck's Conway polynomial database (same as Python `galois`).
pub fn default_irreducible(p: u64, m: u32) -> Result<PrimePoly> {
    if m == 1 {
        return PrimePoly::new(p, vec![0, 1]);
    }

    match conway_poly_lookup(p, m) {
        Ok((degrees, coeffs)) => {
            let dense = sparse_to_dense(&degrees, &coeffs, m);
            let poly = PrimePoly::new(p, dense)?;
            if !poly.is_irreducible() {
                return Err(GaloisError::ReduciblePolynomial { characteristic: p });
            }
            Ok(poly)
        }
        Err(GaloisError::ConwayPolyNotFound { .. }) => {
            // Fall back to irreducible poly database
            crate::databases::irreducible_poly_lookup(p, m).and_then(|(degrees, coeffs)| {
                let dense = sparse_to_dense(&degrees, &coeffs, m);
                PrimePoly::new(p, dense)
            })
        }
        Err(e) => Err(e),
    }
}

/// Returns the Conway polynomial C_{p,m}(x) over GF(p).
pub fn conway_poly(p: u64, m: u32) -> Result<PrimePoly> {
    let (degrees, coeffs) = conway_poly_lookup(p, m)?;
    PrimePoly::new(p, sparse_to_dense(&degrees, &coeffs, m))
}

/// Returns a monic irreducible polynomial over GF(p) of degree m.
pub fn irreducible_poly(p: u64, m: u32) -> Result<PrimePoly> {
    if m == 0 {
        return Err(GaloisError::InvalidDegree(0));
    }
    if m == 1 {
        return PrimePoly::new(p, vec![0, 1]);
    }
    let (degrees, coeffs) = crate::databases::irreducible_poly_lookup(p, m)?;
    PrimePoly::new(p, sparse_to_dense(&degrees, &coeffs, m))
}

/// Convert sparse (degree, coeff) pairs to dense ascending coefficients including monic term.
pub(crate) fn sparse_to_dense(degrees: &[u32], coeffs: &[u64], m: u32) -> Vec<u64> {
    let mut dense = vec![0u64; m as usize + 1];
    for (&deg, &coeff) in degrees.iter().zip(coeffs.iter()) {
        if deg as usize <= m as usize {
            dense[deg as usize] = coeff;
        }
    }
    if dense[m as usize] == 0 {
        dense[m as usize] = 1;
    }
    dense
}
