//! Primitive polynomial generation and lookup.

use crate::conway::sparse_to_dense;
use crate::databases::irreducible_poly_lookup;
use crate::error::{GaloisError, Result};
use crate::poly::{Poly, PrimePoly};
use crate::field::GaloisField;

/// Returns a monic primitive polynomial over GF(p) of degree m.
///
/// Searches lexicographically-minimal primitive polynomials (same strategy as Python galois
/// when not in the Wolfram database).
pub fn primitive_poly(p: u64, m: u32) -> Result<PrimePoly> {
    if m == 0 {
        return Err(GaloisError::InvalidDegree(0));
    }
    if m == 1 {
        return PrimePoly::new(p, vec![0, 1]);
    }

    // Try database first (many primitive polys are stored as irreducible)
    if let Ok((degrees, coeffs)) = irreducible_poly_lookup(p, m) {
        let dense = sparse_to_dense(&degrees, &coeffs, m);
        let poly = PrimePoly::new(p, dense)?;
        if poly.is_irreducible() {
            let gf = GaloisField::with_options(p, m, Some(poly.clone()), None, false)?;
            if is_primitive_prime_poly(&poly, &gf) {
                return Ok(poly);
            }
        }
    }

    // Search lexicographically
    search_primitive_poly(p, m)
}

fn is_primitive_prime_poly(poly: &PrimePoly, field: &GaloisField) -> bool {
    let q = field.order();
    let (factors, _) = crate::prime::factorize(q - 1).unwrap_or((vec![q - 1], vec![1]));
    let alpha = field.primitive_element();
    for &f in &factors {
        let exp = (q - 1) / f;
        let gf = GaloisField::with_options(
            field.characteristic(),
            field.degree(),
            Some(poly.clone()),
            Some(alpha),
            false,
        )
        .unwrap_or_else(|_| field.clone());
        if gf.pow(alpha, exp) == 1 {
            return false;
        }
    }
    true
}

fn search_primitive_poly(p: u64, m: u32) -> Result<PrimePoly> {
    let size = m as usize + 1;
    let mut coeffs = vec![0u64; size];
    coeffs[m as usize] = 1;

    loop {
        if coeffs[m as usize] != 1 {
            return Err(GaloisError::ReduciblePolynomial { characteristic: p });
        }
        if let Ok(poly) = PrimePoly::new(p, coeffs.clone()) {
            if poly.is_irreducible() {
                if let Ok(gf) = GaloisField::with_options(p, m, Some(poly.clone()), None, false) {
                    if is_primitive_prime_poly(&poly, &gf) {
                        return Ok(poly);
                    }
                }
            }
        }
        if !increment_lexicographic(&mut coeffs, p, m as usize) {
            return Err(GaloisError::ReduciblePolynomial { characteristic: p });
        }
    }
}

fn increment_lexicographic(coeffs: &mut [u64], p: u64, max_idx: usize) -> bool {
    for i in 0..max_idx {
        coeffs[i] += 1;
        if coeffs[i] < p {
            return true;
        }
        coeffs[i] = 0;
    }
    false
}

/// Whether a polynomial over GF(q) is primitive.
pub fn is_primitive(poly: &Poly) -> bool {
    let field = poly.field();
    if poly.degree() <= 0 {
        return false;
    }
    let q = field.order();
    let (factors, _) = crate::prime::factorize(q - 1).unwrap_or((vec![q - 1], vec![1]));
    let x = Poly::x(field.clone()).unwrap();
    for &f in &factors {
        let exp = (q - 1) / f;
        if poly.pow(exp).map(|p| p.rem(&x).ok()).ok().flatten().map(|r| r.is_zero()).unwrap_or(false) {
            return false;
        }
    }
    true
}

trait PolyPow {
    fn pow(&self, exp: u64) -> Result<Poly>;
}

impl PolyPow for Poly {
    fn pow(&self, mut exp: u64) -> Result<Poly> {
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
