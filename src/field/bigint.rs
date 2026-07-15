//! Big integer field element support for large GF(p^m).

use num_bigint::BigUint;
use num_traits::{One, Zero};

use crate::error::{GaloisError, Result};
use crate::poly::PrimePoly;

/// A field element that may exceed u64 range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldValue {
    Small(u64),
    Large(BigUint),
}

impl FieldValue {
    pub fn from_u64(v: u64) -> Self {
        Self::Small(v)
    }

    pub fn from_big(v: BigUint) -> Self {
        if let Ok(s) = u64::try_from(v.clone()) {
            Self::Small(s)
        } else {
            Self::Large(v)
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Small(v) => Some(*v),
            Self::Large(v) => u64::try_from(v).ok(),
        }
    }

    pub fn to_biguint(&self) -> BigUint {
        match self {
            Self::Small(v) => BigUint::from(*v),
            Self::Large(v) => v.clone(),
        }
    }
}

pub fn order_fits_u64(p: u64, m: u32) -> bool {
    p.checked_pow(m).is_some()
}

pub fn field_order_big(p: u64, m: u32) -> BigUint {
    BigUint::from(p).pow(m)
}

pub fn needs_bigint(p: u64, m: u32) -> bool {
    !order_fits_u64(p, m)
}

pub fn big_add(a: &FieldValue, b: &FieldValue, p: u64) -> FieldValue {
    let pa = BigUint::from(p);
    let sum = (a.to_biguint() + b.to_biguint()) % &pa;
    FieldValue::from_big(sum)
}

pub fn big_sub(a: &FieldValue, b: &FieldValue, p: u64) -> FieldValue {
    let pa = BigUint::from(p);
    let diff = (a.to_biguint() + &pa - b.to_biguint()) % &pa;
    FieldValue::from_big(diff)
}

pub fn big_neg(a: &FieldValue, p: u64) -> FieldValue {
    if a.to_biguint().is_zero() {
        FieldValue::from_u64(0)
    } else {
        FieldValue::from_big(BigUint::from(p) - a.to_biguint())
    }
}

pub fn big_mul(a: &FieldValue, b: &FieldValue, p: u64) -> FieldValue {
    let pa = BigUint::from(p);
    let prod = (a.to_biguint() * b.to_biguint()) % &pa;
    FieldValue::from_big(prod)
}

pub fn big_div(a: &FieldValue, b: &FieldValue, p: u64) -> FieldValue {
    let inv = big_pow(b, p - 2, p);
    big_mul(a, &inv, p)
}

pub fn big_pow(a: &FieldValue, mut exp: u64, p: u64) -> FieldValue {
    let mut result = FieldValue::from_u64(1);
    let mut base = a.clone();
    while exp > 0 {
        if exp % 2 == 1 {
            result = big_mul(&result, &base, p);
        }
        base = big_mul(&base, &base, p);
        exp /= 2;
    }
    result
}

pub fn big_mul_ext(
    a: &FieldValue,
    b: &FieldValue,
    p: u64,
    m: u32,
    irreducible: &PrimePoly,
) -> FieldValue {
    let a_coeffs = big_int_to_coeffs(&a.to_biguint(), p, m);
    let b_coeffs = big_int_to_coeffs(&b.to_biguint(), p, m);
    let product = big_mul_polynomials(&a_coeffs, &b_coeffs, p);
    let reduced = big_reduce_polynomial(&product, p, m, irreducible);
    FieldValue::from_big(big_coeffs_to_int(&reduced, p))
}

pub fn big_int_to_coeffs(v: &BigUint, p: u64, m: u32) -> Vec<u64> {
    let mut coeffs = vec![0u64; m as usize];
    let mut val = v.clone();
    let pb = BigUint::from(p);
    for c in coeffs.iter_mut() {
        let rem = &val % &pb;
        *c = rem.to_string().parse().unwrap_or(0);
        val /= &pb;
        if val.is_zero() {
            break;
        }
    }
    coeffs
}

pub fn big_coeffs_to_int(coeffs: &[u64], p: u64) -> BigUint {
    let mut value = BigUint::zero();
    let mut power = BigUint::one();
    let pb = BigUint::from(p);
    for &c in coeffs {
        value += BigUint::from(c) * &power;
        power *= &pb;
    }
    value
}

pub fn big_mul_polynomials(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    let mut out = vec![0u64; a.len() + b.len() - 1];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] = (out[i + j] + av * bv) % p;
        }
    }
    out
}

pub fn big_reduce_polynomial(poly: &[u64], p: u64, m: u32, irreducible: &PrimePoly) -> Vec<u64> {
    let m = m as usize;
    let mut work = poly.to_vec();
    let f = irreducible.coeffs();

    while work.len() > m {
        let deg = work.len() - 1;
        let coeff = work[deg];
        if coeff != 0 {
            let shift = deg - m;
            for (i, &fi) in f.iter().enumerate() {
                let idx = shift + i;
                if idx < work.len() {
                    work[idx] = (work[idx] + p - (coeff * fi) % p) % p;
                }
            }
        }
        work.pop();
    }
    work.resize(m, 0);
    work
}

pub fn validate_big_element(value: &BigUint, order: &BigUint) -> Result<()> {
    if value >= order {
        return Err(GaloisError::InvalidElement {
            value: u64::try_from(value.clone()).unwrap_or(u64::MAX),
            characteristic: 0,
            degree: 0,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_order_detection() {
        assert!(needs_bigint(u64::MAX, 2));
        assert!(!needs_bigint(3, 5));
    }

    #[test]
    fn big_prime_mul() {
        let p = 1000000007u64;
        let a = FieldValue::from_u64(p - 1);
        let b = FieldValue::from_u64(2);
        let c = big_mul(&a, &b, p);
        assert_eq!(c.as_u64(), Some((p - 1) * 2 % p));
    }
}
