//! Polynomial factorization over finite fields.

use crate::error::Result;
use crate::poly::field_poly::{frobenius_step, mod_inverse_poly, raise_to_field_order};
use crate::poly::{poly_gcd, Poly};

/// Square-free factorization of a polynomial.
pub fn square_free_factorization(poly: &Poly) -> Result<Vec<(usize, Poly)>> {
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
        let g = poly_gcd(&f, &derivative)?;
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

/// Find roots of a polynomial (brute force for small fields).
pub fn roots(poly: &Poly) -> Result<Vec<u64>> {
    let field = poly.field();
    let q = field.order();
    let limit = q.min(10_000);
    let mut found = Vec::new();
    for x in 0..limit {
        if poly.evaluate(x)? == 0 {
            found.push(x);
        }
    }
    Ok(found)
}

/// Distinct-degree factorization via Berlekamp's algorithm.
pub fn distinct_degree_factorization(poly: &Poly) -> Result<Vec<(usize, Poly)>> {
    if poly.is_zero() {
        return Ok(Vec::new());
    }
    let mut result = Vec::new();
    for (i, factor) in square_free_factorization(poly)? {
        result.extend(ddf_square_free(&factor, i)?);
    }
    Ok(result)
}

/// Equal-degree factorization: split a DDF factor into irreducibles of degree `degree`.
pub fn equal_degree_factorization(factor: &Poly, degree: usize) -> Result<Vec<Poly>> {
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
        let found = roots(factor)?;
        if found.len() == n {
            let field = factor.field();
            return found
                .into_iter()
                .map(|root| Poly::new(vec![1, field.neg(root)], field.clone()))
                .collect::<Result<Vec<_>>>();
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
            let b = Poly::new(
                vec![1, field.validate_element(attempt)?],
                field.clone(),
            )?;
            let powered = pow_qd_minus_1(&b, degree, &f)?;
            let h = powered.sub(&Poly::one(field.clone())?)?;
            let d = poly_gcd(&f, &h)?;
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

fn ddf_square_free(f: &Poly, _multiplicity: usize) -> Result<Vec<(usize, Poly)>> {
    let n = f.degree() as usize;
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![(1, f.clone())]);
    }

    let field = f.field();
    let x = Poly::x(field.clone())?;
    let mut remaining = f.clone();
    let mut h = x.clone();
    let mut out = Vec::new();

    for i in 1..=(n / 2) {
        if remaining.is_one() {
            break;
        }
        h = frobenius_step(&h, f)?;
        let d = poly_gcd(&remaining, &h.sub(&x)?)?;
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

fn pow_qd_minus_1(b: &Poly, d: usize, modulus: &Poly) -> Result<Poly> {
    let mut h = b.rem(modulus)?;
    for _ in 0..d {
        h = raise_to_field_order(&h, modulus)?;
    }
    let inv = mod_inverse_poly(b, modulus)?;
    h.mul(&inv)?.rem(modulus)
}

/// Full factorization into irreducible polynomials.
pub fn factors(poly: &Poly) -> Result<Vec<Poly>> {
    if poly.degree() <= 0 {
        return Ok(vec![poly.clone()]);
    }
    let field = poly.field();
    let mut result = Vec::new();

    for (deg, factor) in distinct_degree_factorization(poly)? {
        if deg == 1 {
            for root in roots(&factor)? {
                result.push(Poly::new(
                    vec![1, field.neg(root)],
                    field.clone(),
                )?);
            }
        } else {
            result.extend(equal_degree_factorization(&factor, deg)?);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GaloisField;

    #[test]
    fn roots_of_x2_minus_1_gf7() {
        let gf = GaloisField::new(7, 1).unwrap();
        let p = Poly::new(vec![1, 0, 6], gf).unwrap();
        let r = roots(&p).unwrap();
        assert!(r.contains(&1));
        assert!(r.contains(&6));
    }

    #[test]
    fn ddf_x4_minus_1_gf5() {
        let gf = GaloisField::new(5, 1).unwrap();
        let p = Poly::new(vec![1, 0, 0, 0, 4], gf).unwrap();
        let ddf = distinct_degree_factorization(&p).unwrap();
        let degrees: Vec<usize> = ddf.iter().map(|(d, _)| *d).collect();
        assert!(degrees.contains(&1));
    }

    #[test]
    fn factors_x4_minus_1_gf5() {
        let gf = GaloisField::new(5, 1).unwrap();
        let p = Poly::new(vec![1, 0, 0, 0, 4], gf).unwrap();
        let facs = factors(&p).unwrap();
        assert_eq!(facs.len(), 4);
        assert!(facs.iter().all(|f| f.degree() == 1));
    }

    #[test]
    fn edf_quadratic_gf5() {
        let gf = GaloisField::new(5, 1).unwrap();
        // x^2 + 1 = (x-2)(x+2) over GF(5) since 2^2 = 4 = -1
        let p = Poly::new(vec![1, 0, 1], gf).unwrap();
        let facs = equal_degree_factorization(&p, 1).unwrap();
        assert_eq!(facs.len(), 2);
    }
}
