//! Parse polynomial strings like "x^3 + x + 1".

use crate::error::{GaloisError, Result};
use crate::field::GaloisField;
use crate::poly::Poly;

/// Parse a polynomial string over the given field.
///
/// Supports forms like `"x^3 + 2x + 1"` or `"x + 1"`.
pub fn parse_poly(s: &str, field: GaloisField) -> Result<Poly> {
    let coeffs = parse_coeffs_asc(s, field.characteristic())?;
    Poly::new_asc(coeffs, field)
}

/// Parse a polynomial string over any `FieldKind`.
pub fn parse_field_poly(s: &str, field: &crate::field::FieldKind) -> Result<crate::poly::FieldPoly> {
    let p = field.characteristic();
    let coeffs = parse_coeffs_asc(s, p)?;
    match field {
        crate::field::FieldKind::Small(f) => {
            Ok(crate::poly::FieldPoly::Small(Poly::new_asc(coeffs, f.clone())?))
        }
        crate::field::FieldKind::Big(f) => Ok(crate::poly::FieldPoly::Big(
            crate::poly::BigPoly::from_u64_asc(coeffs, f.clone())?,
        )),
    }
}

fn parse_coeffs_asc(s: &str, characteristic: u64) -> Result<Vec<u64>> {
    let s = s.replace(' ', "");
    if s.is_empty() || s == "0" {
        return Ok(vec![0]);
    }

    let var = "x";
    let mut coeffs: Vec<i64> = Vec::new();

    for term in s.split('+').filter(|t| !t.is_empty()) {
        let term = term.trim();
        let (coeff, power) = parse_term(term, var)?;
        let size = power + 1;
        if coeffs.len() < size {
            coeffs.resize(size, 0);
        }
        coeffs[power] += coeff;
    }

    let p = characteristic as i64;
    let mut asc = vec![0u64; coeffs.len()];
    for (i, &c) in coeffs.iter().enumerate() {
        let normalized = ((c % p) + p) % p;
        asc[i] = normalized as u64;
    }

    while asc.len() > 1 && *asc.last().unwrap() == 0 {
        asc.pop();
    }
    if asc.is_empty() {
        asc.push(0);
    }
    Ok(asc)
}

fn parse_term(term: &str, var: &str) -> Result<(i64, usize)> {
    if term == "1" && !term.contains(var) {
        return Ok((1, 0));
    }
    if term == "-1" && !term.contains(var) {
        return Ok((-1, 0));
    }

    let (coeff_part, power_part) = if let Some(idx) = term.find(var) {
        (&term[..idx], &term[idx..])
    } else {
        (term, "")
    };

    let coeff = match coeff_part {
        "" | "+" => 1,
        "-" => -1,
        c => c
            .parse::<i64>()
            .map_err(|_| GaloisError::InvalidPolynomialString(term.to_string()))?,
    };

    let power = if power_part.is_empty() {
        0
    } else if power_part == var {
        1
    } else if let Some(exp_str) = power_part.strip_prefix(&format!("{var}^")) {
        exp_str
            .parse::<usize>()
            .map_err(|_| GaloisError::InvalidPolynomialString(term.to_string()))?
    } else {
        return Err(GaloisError::InvalidPolynomialString(term.to_string()));
    };

    Ok((coeff, power))
}

impl Poly {
    /// Parse from string, e.g. `Poly::from_str("x^3 + x + 1", gf)?`
    pub fn from_str(s: &str, field: GaloisField) -> Result<Self> {
        parse_poly(s, field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GaloisField;

    #[test]
    fn parse_x3_plus_x_plus_1() {
        let gf = GaloisField::new(2, 1).unwrap();
        let p = parse_poly("x^3 + x + 1", gf).unwrap();
        assert_eq!(p.format_poly("x"), "x^3 + x + 1");
    }

    #[test]
    fn parse_field_poly_gf2_2() {
        let fk = crate::field::FieldKind::Big(crate::field::BigGaloisField::new(2, 2).unwrap());
        let p = parse_field_poly("x^2 + x + 1", &fk).unwrap();
        assert_eq!(p.degree(), 2);
    }
}
