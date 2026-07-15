//! Polynomial algorithms over `BigGaloisField`.

use num_bigint::BigUint;
use num_traits::One;

use num_traits::Zero;

use crate::error::{GaloisError, Result};

use super::BigPoly;

/// Greatest common divisor of two `BigPoly` polynomials.
pub fn big_poly_gcd(a: &BigPoly, b: &BigPoly) -> Result<BigPoly> {
    let mut r0 = a.clone();
    let mut r1 = b.clone();
    while !r1.is_zero() {
        let (_, rem) = r0.divmod(&r1)?;
        r0 = r1;
        r1 = rem;
    }
    r0.make_monic()
}

/// Multiplicative inverse of `a` modulo monic `modulus`.
pub fn big_mod_inverse_poly(a: &BigPoly, modulus: &BigPoly) -> Result<BigPoly> {
    let field = a.field().clone();
    let mut old_r = modulus.clone();
    let mut r = a.rem(modulus)?;
    let mut old_t = BigPoly::zero(field.clone())?;
    let mut t = BigPoly::one(field.clone())?;

    while !r.is_zero() {
        let (q, rem) = old_r.divmod(&r)?;
        old_r = r;
        r = rem;
        let qt = q.mul(&t)?;
        let new_t = old_t.sub(&qt)?;
        old_t = t;
        t = new_t;
    }

    if old_r.degree() != 0 {
        return Err(GaloisError::PolynomialDivisionByZero);
    }
    let lead = old_r.leading_coeff();
    if lead.is_zero() {
        return Err(GaloisError::PolynomialDivisionByZero);
    }
    let inv = field.div(&BigUint::one(), lead)?;
    t.mul_scalar(&inv)
}

/// Frobenius map: `poly` ↦ `poly^p` mod `modulus`.
pub fn big_frobenius_step(h: &BigPoly, modulus: &BigPoly) -> Result<BigPoly> {
    let field = h.field();
    let p = field.characteristic();
    let mut result = BigPoly::zero(field.clone())?;
    for (i, coeff) in h.coeffs_asc().iter().enumerate() {
        let term = BigPoly::new(vec![coeff.clone()], field.clone())?;
        let power = term.mod_pow(p as usize, modulus)?;
        let shifted = power.shift(i)?;
        result = result.add(&shifted)?;
    }
    result.rem(modulus)
}

/// Raise `poly` to q = p^m modulo `modulus`.
pub fn big_raise_to_field_order(poly: &BigPoly, modulus: &BigPoly) -> Result<BigPoly> {
    let steps = poly.field().degree().max(1) as usize;
    let mut h = poly.rem(modulus)?;
    for _ in 0..steps {
        h = big_frobenius_step(&h, modulus)?;
    }
    Ok(h)
}
