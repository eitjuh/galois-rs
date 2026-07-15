//! Decode for `FieldReedSolomon` over small and big fields.

use num_bigint::BigUint;
use num_traits::{One, Zero};

use crate::error::{GaloisError, Result};
use crate::field::{BigFieldArray, BigGaloisField, FieldArray, FieldKind, GaloisArray};
use crate::lfsr::berlekamp_massey_array;
use crate::poly::{BigPoly, FieldPoly, Poly};

use super::field_code::FieldReedSolomon;
use super::ReedSolomon;

impl FieldReedSolomon {
    pub(crate) fn decode_inner(&self, received: &GaloisArray) -> Result<GaloisArray> {
        if received.len() != self.n {
            return Err(GaloisError::LengthMismatch);
        }
        match (&self.field, received) {
            (FieldKind::Small(f), GaloisArray::Small(r)) => {
                let rs = ReedSolomon {
                    field: f.clone(),
                    n: self.n,
                    k: self.k,
                    generator: self.generator.as_small().unwrap().clone(),
                };
                Ok(GaloisArray::Small(rs.decode(r)?))
            }
            (FieldKind::Big(f), GaloisArray::Big(r)) => decode_big(self, f, r),
            _ => Err(GaloisError::FieldMismatch),
        }
    }
}

pub fn decode(rs: &FieldReedSolomon, received: &GaloisArray) -> Result<GaloisArray> {
    rs.decode_inner(received)
}

pub fn syndromes(rs: &FieldReedSolomon, received: &GaloisArray) -> Result<GaloisArray> {
    if received.len() != rs.n {
        return Err(GaloisError::LengthMismatch);
    }
    match (&rs.field, received) {
        (FieldKind::Small(f), GaloisArray::Small(r)) => {
            let inner = ReedSolomon {
                field: f.clone(),
                n: rs.n,
                k: rs.k,
                generator: rs.generator.as_small().unwrap().clone(),
            };
            Ok(GaloisArray::Small(FieldArray::new(
                f.clone(),
                inner_syndromes_small(&inner, r)?,
            )))
        }
        (FieldKind::Big(f), GaloisArray::Big(r)) => {
            let values = syndromes_big(f, rs.n, rs.k, r)?;
            Ok(GaloisArray::Big(BigFieldArray::from_shape_vec(
                f.clone(),
                &[values.len()],
                values,
            )?))
        }
        _ => Err(GaloisError::FieldMismatch),
    }
}

fn inner_syndromes_small(rs: &ReedSolomon, received: &FieldArray) -> Result<Vec<u64>> {
    let field = &rs.field;
    let d = rs.n - rs.k + 1;
    let alpha = field.primitive_element();
    let mut s = Vec::with_capacity(d - 1);
    for i in 1..d {
        let mut sum = 0u64;
        for (j, &r) in received.values().iter().enumerate() {
            let exp = field.pow(alpha, (i * j) as u64);
            sum = field.add(sum, field.mul(r, exp));
        }
        s.push(sum);
    }
    Ok(s)
}

fn syndromes_big(
    field: &BigGaloisField,
    n: usize,
    k: usize,
    received: &BigFieldArray,
) -> Result<Vec<BigUint>> {
    let d = n - k + 1;
    let alpha = field.primitive_element()?;
    let mut s = Vec::with_capacity(d - 1);
    for i in 1..d {
        let mut sum = BigUint::zero();
        for (j, r) in received.values().iter().enumerate() {
            let exp = field.pow(&alpha, (i * j) as u64);
            sum = field.add(&sum, &field.mul(r, &exp));
        }
        s.push(sum);
    }
    Ok(s)
}

fn decode_big(
    rs: &FieldReedSolomon,
    field: &BigGaloisField,
    received: &BigFieldArray,
) -> Result<GaloisArray> {
    let syndrome_values = syndromes_big(field, rs.n, rs.k, received)?;
    if syndrome_values.iter().all(|s| s.is_zero()) {
        return Ok(GaloisArray::Big(BigFieldArray::from_shape_vec(
            field.clone(),
            &[rs.k],
            received.values()[rs.n - rs.k..].to_vec(),
        )?));
    }

    let syndrome_array = GaloisArray::Big(BigFieldArray::from_shape_vec(
        field.clone(),
        &[syndrome_values.len()],
        syndrome_values,
    )?);

    let t = (rs.n - rs.k) / 2;
    let locator = berlekamp_massey_array(&syndrome_array)?;
    if locator.degree() as usize > t {
        return Err(GaloisError::LengthMismatch);
    }

    let positions = chien_search_field(&locator, rs.n)?;
    if positions.len() != locator.degree() as usize {
        return Err(GaloisError::LengthMismatch);
    }

    let error_values = forney_errors_field(&syndrome_array, &locator, &positions)?;

    if let Some(decoded) = try_correction_big(field, received, rs.k, &positions, &error_values)? {
        return Ok(decoded);
    }

    let negated: Vec<BigUint> = error_values.iter().map(|v| field.neg(v)).collect();
    if let Some(decoded) = try_correction_big(field, received, rs.k, &positions, &negated)? {
        return Ok(decoded);
    }

    Err(GaloisError::LengthMismatch)
}

fn try_correction_big(
    field: &BigGaloisField,
    received: &BigFieldArray,
    k: usize,
    positions: &[usize],
    values: &[BigUint],
) -> Result<Option<GaloisArray>> {
    let mut corrected = received.values().to_vec();
    for (&pos, val) in positions.iter().zip(values.iter()) {
        corrected[pos] = field.sub(&corrected[pos], val);
    }
    let check = BigFieldArray::from_shape_vec(field.clone(), &[corrected.len()], corrected.clone())?;
    let n = corrected.len();
    let syndromes = syndromes_big(field, n, n - k, &check)?;
    if syndromes.iter().all(|s| s.is_zero()) {
        return Ok(Some(GaloisArray::Big(BigFieldArray::from_shape_vec(
            field.clone(),
            &[k],
            corrected[n - k..].to_vec(),
        )?)));
    }
    Ok(None)
}

pub fn chien_search_field(locator: &FieldPoly, n: usize) -> Result<Vec<usize>> {
    match locator {
        FieldPoly::Small(p) => chien_search_small(p, n),
        FieldPoly::Big(p) => chien_search_big(p, n),
    }
}

fn chien_search_small(locator: &Poly, n: usize) -> Result<Vec<usize>> {
    let field = locator.field();
    let alpha = field.primitive_element();
    let mut positions = Vec::new();
    for i in 0..n {
        let x_inv = field.div(1, field.pow(alpha, i as u64))?;
        if locator.evaluate(x_inv)? == 0 {
            positions.push(i);
        }
    }
    Ok(positions)
}

fn chien_search_big(locator: &BigPoly, n: usize) -> Result<Vec<usize>> {
    let field = locator.field();
    let alpha = field.primitive_element()?;
    let mut positions = Vec::new();
    for i in 0..n {
        let x = field.pow(&alpha, i as u64);
        let x_inv = field.div(&BigUint::one(), &x)?;
        if locator.evaluate(&x_inv)? == BigUint::zero() {
            positions.push(i);
        }
    }
    Ok(positions)
}

pub fn forney_errors_field(
    syndromes: &GaloisArray,
    locator: &FieldPoly,
    positions: &[usize],
) -> Result<Vec<BigUint>> {
    match (syndromes, locator) {
        (GaloisArray::Small(s), FieldPoly::Small(loc)) => {
            let values = forney_small(s, loc, positions)?;
            Ok(values.into_iter().map(BigUint::from).collect())
        }
        (GaloisArray::Big(s), FieldPoly::Big(loc)) => forney_big(s, loc, positions),
        _ => Err(GaloisError::FieldMismatch),
    }
}

fn forney_small(syndromes: &FieldArray, locator: &Poly, positions: &[usize]) -> Result<Vec<u64>> {
    let field = syndromes.field();
    let alpha = field.primitive_element();
    let coeffs: Vec<u64> = syndromes.values().iter().rev().copied().collect();
    let s_poly = Poly::new(coeffs, field.clone())?;
    let mut mod_coeffs = vec![0u64; syndromes.len() + 1];
    mod_coeffs[0] = 1;
    let modulus = Poly::new(mod_coeffs, field.clone())?;
    let omega = s_poly.mul(locator)?.rem(&modulus)?;

    let mut values = Vec::with_capacity(positions.len());
    for &pos in positions {
        let x = field.pow(alpha, pos as u64);
        let x_inv = field.div(1, x)?;
        let num = omega.evaluate(x_inv)?;
        let denom = locator_derivative_small(locator, x_inv)?;
        if denom == 0 {
            return Err(GaloisError::DivisionByZero {
                characteristic: field.characteristic(),
                degree: field.degree(),
            });
        }
        values.push(field.neg(field.div(num, denom)?));
    }
    Ok(values)
}

fn forney_big(
    syndromes: &BigFieldArray,
    locator: &BigPoly,
    positions: &[usize],
) -> Result<Vec<BigUint>> {
    let field = syndromes.field();
    let alpha = field.primitive_element()?;
    let coeffs: Vec<BigUint> = syndromes.values().iter().rev().cloned().collect();
    let s_poly = BigPoly::new(coeffs, field.clone())?;
    let mut mod_coeffs = vec![BigUint::zero(); syndromes.len() + 1];
    mod_coeffs[0] = BigUint::one();
    let modulus = BigPoly::new(mod_coeffs, field.clone())?;
    let omega = s_poly.mul(locator)?.rem(&modulus)?;

    let mut values = Vec::with_capacity(positions.len());
    for &pos in positions {
        let x = field.pow(&alpha, pos as u64);
        let x_inv = field.div(&BigUint::one(), &x)?;
        let num = omega.evaluate(&x_inv)?;
        let denom = locator_derivative_big(locator, &x_inv)?;
        if denom.is_zero() {
            return Err(GaloisError::DivisionByZero {
                characteristic: field.characteristic(),
                degree: field.degree(),
            });
        }
        let val = field.div(&num, &denom)?;
        values.push(field.neg(&val));
    }
    Ok(values)
}

fn locator_derivative_small(locator: &Poly, x: u64) -> Result<u64> {
    let field = locator.field();
    if locator.degree() <= 0 {
        return Ok(0);
    }
    let mut result = 0u64;
    let d = locator.degree() as usize;
    for (i, &coeff) in locator.coeffs().iter().enumerate() {
        let power = d - i;
        if power == 0 {
            continue;
        }
        let term = field.mul(
            field.mul(coeff, power as u64),
            field.pow(x, (power - 1) as u64),
        );
        result = field.add(result, term);
    }
    Ok(result)
}

fn locator_derivative_big(locator: &BigPoly, x: &BigUint) -> Result<BigUint> {
    let field = locator.field();
    if locator.degree() <= 0 {
        return Ok(BigUint::zero());
    }
    let mut result = BigUint::zero();
    let d = locator.degree() as usize;
    for (i, coeff) in locator.coeffs().iter().enumerate() {
        let power = d - i;
        if power == 0 {
            continue;
        }
        let term = field.mul(
            &field.mul(coeff, &BigUint::from(power as u64)),
            &field.pow(x, (power - 1) as u64),
        );
        result = field.add(&result, &term);
    }
    Ok(result)
}
