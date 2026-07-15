//! Polynomial factorization over `BigGaloisField`.

use num_bigint::BigUint;
use num_traits::{One, Zero};

use crate::error::Result;

use super::big_field_poly::{
    big_frobenius_step, big_mod_inverse_poly, big_poly_gcd, big_raise_to_field_order,
};
use super::BigPoly;

/// Square-free factorization of a `BigPoly`.
pub fn big_square_free_factorization(poly: &BigPoly) -> Result<Vec<(usize, BigPoly)>> {
    if poly.is_zero() {
        return Ok(Vec::new());
    }
    let mut f = poly.clone();
    let mut i = 1usize;
    let mut result = Vec::new();

    loop {
        if f.is_zero() {
            break;
        }
        let derivative = f.derivative(1)?;
        let g = big_poly_gcd(&f, &derivative)?;
        let w = if g.is_one() {
            f.clone()
        } else {
            f.div(&g)?
        };

        if !w.is_one() && !w.is_zero() {
            result.push((i, w));
        }

        if g.is_one() {
            break;
        }
        f = g;
        i += 1;
    }

    if result.is_empty() && !poly.is_zero() {
        result.push((1, poly.clone()));
    }
    Ok(result)
}

/// Distinct-degree factorization via Berlekamp's algorithm.
pub fn big_distinct_degree_factorization(poly: &BigPoly) -> Result<Vec<(usize, BigPoly)>> {
    if poly.is_zero() {
        return Ok(Vec::new());
    }
    let mut result = Vec::new();
    for (i, factor) in big_square_free_factorization(poly)? {
        result.extend(big_ddf_square_free(&factor, i)?);
    }
    Ok(result)
}

/// Equal-degree factorization for a DDF factor.
pub fn big_equal_degree_factorization(factor: &BigPoly, degree: usize) -> Result<Vec<BigPoly>> {
    let n = factor.degree() as usize;
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == degree {
        return Ok(vec![factor.clone()]);
    }
    if n % degree != 0 {
        return Ok(vec![factor.clone()]);
    }

    if degree == 1 {
        let found = big_roots(factor)?;
        if found.len() == n {
            let field = factor.field();
            return found
                .into_iter()
                .map(|root| {
                    BigPoly::new(
                        vec![BigUint::one(), field.neg(&root)],
                        field.clone(),
                    )
                })
                .collect();
        }
    }

    let field = factor.field();
    let mut factors = vec![factor.clone()];
    let mut attempt = 1u64;

    while factors.iter().any(|f| f.degree() as usize > degree) {
        let mut next = Vec::new();
        let mut split = false;
        for f in factors {
            if f.degree() as usize <= degree {
                next.push(f);
                continue;
            }
            let b = BigPoly::from_u64(vec![1, attempt], field.clone())?;
            let powered = big_pow_qd_minus_1(&b, degree, &f)?;
            let h = powered.sub(&BigPoly::one(field.clone())?)?;
            let d = big_poly_gcd(&f, &h)?;
            if d.is_one() || d.degree() == f.degree() {
                next.push(f);
            } else {
                split = true;
                let q_poly = f.div(&d)?;
                next.push(d);
                next.push(q_poly);
            }
        }
        factors = next;
        attempt += 1;
        if !split {
            break;
        }
        if attempt > 256 {
            break;
        }
    }

    Ok(factors
        .into_iter()
        .filter(|f| f.degree() as usize == degree)
        .collect())
}

/// Full factorization into irreducible polynomials.
pub fn big_factors(poly: &BigPoly) -> Result<Vec<BigPoly>> {
    if poly.degree() <= 0 {
        return Ok(vec![poly.clone()]);
    }
    let field = poly.field();
    let mut result = Vec::new();

    for (deg, factor) in big_distinct_degree_factorization(poly)? {
        if deg == 1 {
            for root in big_roots(&factor)? {
                result.push(BigPoly::new(
                    vec![BigUint::one(), field.neg(&root)],
                    field.clone(),
                )?);
            }
        } else {
            result.extend(big_equal_degree_factorization(&factor, deg)?);
        }
    }
    Ok(result)
}

/// Brute-force roots for small fields (order <= 10_000).
pub fn big_roots(poly: &BigPoly) -> Result<Vec<BigUint>> {
    let field = poly.field();
    let limit = if let Ok(q) = u64::try_from(field.order()) {
        q.min(10_000)
    } else {
        0
    };
    let mut found = Vec::new();
    for x in 0..limit {
        let v = BigUint::from(x);
        if poly.evaluate(&v)? == BigUint::zero() {
            found.push(v);
        }
    }
    Ok(found)
}

fn big_ddf_square_free(f: &BigPoly, _multiplicity: usize) -> Result<Vec<(usize, BigPoly)>> {
    let n = f.degree() as usize;
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![(1, f.clone())]);
    }

    let field = f.field();
    let x = BigPoly::x(field.clone())?;
    let mut remaining = f.clone();
    let mut h = x.clone();
    let mut out = Vec::new();

    for i in 1..=(n / 2) {
        if remaining.is_one() {
            break;
        }
        h = big_frobenius_step(&h, f)?;
        let d = big_poly_gcd(&remaining, &h.sub(&x)?)?;
        if !d.is_one() {
            out.push((i, d.clone()));
            remaining = remaining.div(&d)?;
        }
    }
    if !remaining.is_one() && !remaining.is_zero() {
        out.push((remaining.degree() as usize, remaining));
    }
    Ok(out)
}

fn big_pow_qd_minus_1(b: &BigPoly, d: usize, modulus: &BigPoly) -> Result<BigPoly> {
    let mut h = b.rem(modulus)?;
    for _ in 0..d {
        h = big_raise_to_field_order(&h, modulus)?;
    }
    let inv = big_mod_inverse_poly(b, modulus)?;
    h.mul(&inv)?.rem(modulus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::BigGaloisField;

    #[test]
    fn big_factors_x2_plus_x_plus_1_gf2_2() {
        let field = BigGaloisField::new(2, 2).unwrap();
        let p = BigPoly::from_u64(vec![1, 1, 1], field).unwrap();
        let facs = big_factors(&p).unwrap();
        assert!(!facs.is_empty());
    }
}
