//! Reed-Solomon and BCH decoding with Berlekamp-Massey, Chien search, and Forney correction.

use crate::error::{GaloisError, Result};
use crate::field::{FieldArray, GaloisField};
use crate::lfsr::berlekamp_massey;
use crate::poly::Poly;

use super::{Bch, ReedSolomon};

impl ReedSolomon {
    /// Decode a received word using syndrome decoding.
    pub fn decode(&self, received: &FieldArray) -> Result<FieldArray> {
        if received.len() != self.n {
            return Err(GaloisError::LengthMismatch);
        }
        let syndromes = self.syndromes(received)?;
        if syndromes.iter().all(|&s| s == 0) {
            return Ok(FieldArray::new(
                self.field.clone(),
                received.values()[self.n - self.k..].to_vec(),
            ));
        }

        let t = (self.n - self.k) / 2;
        let syndrome_array = FieldArray::new(self.field.clone(), syndromes.clone());

        if t == 1 {
            if let Some(decoded) = decode_single_error(self, received, &syndromes)? {
                return Ok(decoded);
            }
        }

        if t == 2 && syndromes.len() >= 4 {
            if let Some(decoded) = decode_two_errors(self, received, &syndromes)? {
                return Ok(decoded);
            }
        }

        let locator = berlekamp_massey(&syndrome_array)?;

        if locator.degree() as usize > t {
            return Err(GaloisError::LengthMismatch);
        }

        let error_positions = chien_search(&locator, self.n)?;
        if error_positions.len() != locator.degree() as usize {
            return Err(GaloisError::LengthMismatch);
        }
        let error_values = forney_errors(
            &syndrome_array,
            &locator,
            &error_positions,
            self.n,
        )?;

        if let Some(decoded) = try_correction(self, received, &error_positions, &error_values)? {
            return Ok(decoded);
        }

        let negated: Vec<u64> = error_values
            .iter()
            .map(|&v| self.field.neg(v))
            .collect();
        if let Some(decoded) = try_correction(self, received, &error_positions, &negated)? {
            return Ok(decoded);
        }

        Err(GaloisError::LengthMismatch)
    }

    fn syndromes(&self, received: &FieldArray) -> Result<Vec<u64>> {
        let field = &self.field;
        let d = self.n - self.k + 1;
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
}

impl Bch {
    /// Decode a received BCH codeword.
    pub fn decode(&self, received: &FieldArray) -> Result<FieldArray> {
        let rs = ReedSolomon {
            field: self.field.clone(),
            n: self.n,
            k: self.k,
            generator: self.generator.clone(),
        };
        rs.decode(received)
    }
}

fn try_correction(
    rs: &ReedSolomon,
    received: &FieldArray,
    positions: &[usize],
    values: &[u64],
) -> Result<Option<FieldArray>> {
    let mut corrected = received.values().to_vec();
    for (&pos, &val) in positions.iter().zip(values.iter()) {
        corrected[pos] = rs.field.sub(corrected[pos], val);
    }
    let check = FieldArray::new(rs.field.clone(), corrected.clone());
    if rs.syndromes(&check)?.iter().all(|&s| s == 0) {
        return Ok(Some(FieldArray::new(
            rs.field.clone(),
            corrected[rs.n - rs.k..].to_vec(),
        )));
    }
    Ok(None)
}

/// Find error positions via Chien search on the error locator polynomial.
fn chien_search(locator: &Poly, n: usize) -> Result<Vec<usize>> {
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

/// Compute error magnitudes via Forney's algorithm.
fn forney_errors(
    syndromes: &FieldArray,
    locator: &Poly,
    positions: &[usize],
    _n: usize,
) -> Result<Vec<u64>> {
    let field = syndromes.field();
    let alpha = field.primitive_element();
    let s_poly = syndrome_poly_from_syndromes(syndromes)?;
    let modulus = omega_modulus(field, syndromes.len())?;
    let omega = s_poly.mul(locator)?.rem(&modulus)?;

    let mut values = Vec::with_capacity(positions.len());
    for &pos in positions {
        let x = field.pow(alpha, pos as u64);
        let x_inv = field.div(1, x)?;
        let num = omega.evaluate(x_inv)?;
        let denom = locator_derivative_at(locator, x_inv)?;
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

/// Fast decoder for t=1: one error at position l, magnitude e.
fn decode_single_error(
    rs: &ReedSolomon,
    received: &FieldArray,
    syndromes: &[u64],
) -> Result<Option<FieldArray>> {
    if syndromes.len() < 2 || syndromes[1] == 0 {
        return Ok(None);
    }
    let field = &rs.field;
    let alpha = field.primitive_element();
    let ratio = field.div(syndromes[0], syndromes[1])?;

    let mut position = None;
    for i in 0..rs.n {
        let x_inv = field.div(1, field.pow(alpha, i as u64))?;
        if ratio == x_inv {
            position = Some(i);
            break;
        }
    }
    let pos = match position {
        Some(p) => p,
        None => return Ok(None),
    };

    let x_inv = field.div(1, field.pow(alpha, pos as u64))?;
    let magnitude = field.mul(syndromes[0], x_inv);

    let mut corrected = received.values().to_vec();
    corrected[pos] = field.sub(corrected[pos], magnitude);

    let check = FieldArray::new(field.clone(), corrected.clone());
    if !rs.syndromes(&check)?.iter().all(|&s| s == 0) {
        return Ok(None);
    }

    Ok(Some(FieldArray::new(
        field.clone(),
        corrected[rs.n - rs.k..].to_vec(),
    )))
}

/// Brute-force two-error solver for t = 2.
fn decode_two_errors(
    rs: &ReedSolomon,
    received: &FieldArray,
    syndromes: &[u64],
) -> Result<Option<FieldArray>> {
    let field = &rs.field;
    let alpha = field.primitive_element();

    for p in 0..rs.n {
        for q in (p + 1)..rs.n {
            let xp = field.pow(alpha, p as u64);
            let xq = field.pow(alpha, q as u64);
            let x2p = field.mul(xp, xp);
            let x2q = field.mul(xq, xq);
            let det = field.sub(field.mul(xp, x2q), field.mul(xq, x2p));
            if det == 0 {
                continue;
            }
            let e1 = field.div(
                field.sub(field.mul(syndromes[0], x2q), field.mul(syndromes[1], xq)),
                det,
            )?;
            let e2 = field.div(
                field.sub(field.mul(syndromes[1], xp), field.mul(syndromes[0], x2p)),
                det,
            )?;

            let mut ok = true;
            for (j, &sj) in syndromes.iter().enumerate().take(4) {
                let power = (j + 1) as u64;
                let term = field.add(
                    field.mul(e1, field.pow(xp, power)),
                    field.mul(e2, field.pow(xq, power)),
                );
                if term != sj {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }

            let mut corrected = received.values().to_vec();
            corrected[p] = field.sub(corrected[p], e1);
            corrected[q] = field.sub(corrected[q], e2);
            let check = FieldArray::new(field.clone(), corrected.clone());
            if rs.syndromes(&check)?.iter().all(|&v| v == 0) {
                return Ok(Some(FieldArray::new(
                    field.clone(),
                    corrected[rs.n - rs.k..].to_vec(),
                )));
            }
        }
    }
    Ok(None)
}

fn omega_modulus(field: &GaloisField, syndrome_len: usize) -> Result<Poly> {
    let mut coeffs = vec![0u64; syndrome_len + 1];
    coeffs[0] = 1;
    Poly::new(coeffs, field.clone())
}

fn syndrome_poly_from_syndromes(syndromes: &FieldArray) -> Result<Poly> {
    let coeffs: Vec<u64> = syndromes.values().iter().rev().copied().collect();
    Poly::new(coeffs, syndromes.field().clone())
}

fn locator_derivative_at(locator: &Poly, x: u64) -> Result<u64> {
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

/// Create syndrome polynomial from syndromes.
pub fn syndrome_poly(syndromes: &[u64], field: GaloisField) -> Result<Poly> {
    let coeffs: Vec<u64> = syndromes.iter().rev().copied().collect();
    Poly::new(coeffs, field)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GaloisField;

    #[test]
    fn rs_roundtrip_no_errors() {
        let gf = GaloisField::new(7, 1).unwrap();
        let rs = ReedSolomon::new(gf.clone(), 6, 4).unwrap();
        let msg = gf.array([1, 2, 3, 4]).unwrap();
        let codeword = rs.encode(&msg).unwrap();
        let decoded = rs.decode(&codeword).unwrap();
        assert_eq!(decoded.values(), msg.values());
    }

    #[test]
    fn rs_corrects_single_error() {
        let gf = GaloisField::new(7, 1).unwrap();
        let rs = ReedSolomon::new(gf.clone(), 6, 4).unwrap();
        let msg = gf.array([1, 2, 3, 4]).unwrap();
        let mut codeword = rs.encode(&msg).unwrap().values().to_vec();
        codeword[0] = gf.add(codeword[0], 3);
        let received = gf.array(codeword).unwrap();
        let decoded = rs.decode(&received).unwrap();
        assert_eq!(decoded.values(), msg.values());
    }

    #[test]
    fn rs_corrects_message_error() {
        let gf = GaloisField::new(7, 1).unwrap();
        let rs = ReedSolomon::new(gf.clone(), 6, 4).unwrap();
        let msg = gf.array([1, 2, 3, 4]).unwrap();
        let mut codeword = rs.encode(&msg).unwrap().values().to_vec();
        codeword[5] = gf.add(codeword[5], 2);
        let received = gf.array(codeword).unwrap();
        let decoded = rs.decode(&received).unwrap();
        assert_eq!(decoded.values(), msg.values());
    }

    #[test]
    fn rs_corrects_two_errors() {
        let gf = GaloisField::new(7, 1).unwrap();
        let rs = ReedSolomon::new(gf.clone(), 7, 3).unwrap();
        let msg = gf.array([1, 2, 3]).unwrap();
        let mut codeword = rs.encode(&msg).unwrap().values().to_vec();
        codeword[0] = gf.add(codeword[0], 2);
        codeword[3] = gf.add(codeword[3], 4);
        let received = gf.array(codeword).unwrap();
        let decoded = rs.decode(&received).unwrap();
        assert_eq!(decoded.values(), msg.values());
    }

    #[test]
    fn forney_gf11_magnitudes() {
        let gf = GaloisField::new(11, 1).unwrap();
        let rs = ReedSolomon::new(gf.clone(), 9, 3).unwrap();
        let msg = gf.array([1, 2, 3]).unwrap();
        let mut codeword = rs.encode(&msg).unwrap().values().to_vec();
        codeword[0] = gf.add(codeword[0], 2);
        codeword[2] = gf.add(codeword[2], 3);
        codeword[5] = gf.add(codeword[5], 1);
        let received = gf.array(codeword).unwrap();
        let syndromes = rs.syndromes(&received).unwrap();
        let syndrome_array = FieldArray::new(gf.clone(), syndromes);
        let locator = berlekamp_massey(&syndrome_array).unwrap();
        let positions = chien_search(&locator, rs.n).unwrap();
        let values = forney_errors(&syndrome_array, &locator, &positions, rs.n).unwrap();
        assert_eq!(values, vec![2, 3, 1]);
    }

    #[test]
    fn rs_corrects_three_errors() {
        let gf = GaloisField::new(11, 1).unwrap();
        let rs = ReedSolomon::new(gf.clone(), 9, 3).unwrap();
        let msg = gf.array([1, 2, 3]).unwrap();
        let mut codeword = rs.encode(&msg).unwrap().values().to_vec();
        codeword[0] = gf.add(codeword[0], 2);
        codeword[2] = gf.add(codeword[2], 3);
        codeword[5] = gf.add(codeword[5], 1);
        let received = gf.array(codeword).unwrap();
        let decoded = rs.decode(&received).unwrap();
        assert_eq!(decoded.values(), msg.values());
    }
}