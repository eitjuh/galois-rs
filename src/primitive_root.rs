//! Primitive root functions modulo n.

use crate::error::{GaloisError, Result};
use crate::modular::gcd;
use crate::prime::factorize;
use crate::prime::is_prime;

/// Find a primitive root modulo n in [start, stop).
pub fn primitive_root(n: u64, start: u64, stop: u64) -> Result<u64> {
    for g in start..stop {
        if is_primitive_root(g, n) {
            return Ok(g);
        }
    }
    Err(GaloisError::NoPrimitiveRoot(n))
}

/// Whether g is a primitive root modulo n.
pub fn is_primitive_root(g: u64, n: u64) -> bool {
    if gcd(g, n) != 1 {
        return false;
    }
    let phi = euler_phi_simple(n);
    let (factors, _) = factorize(phi).unwrap_or((vec![phi], vec![1]));
    for &p in &factors {
        if mod_pow(g, phi / p, n) == 1 {
            return false;
        }
    }
    mod_pow(g, phi, n) == 1
}

fn euler_phi_simple(n: u64) -> u64 {
    crate::modular::euler_phi(n)
}

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exp /= 2;
    }
    result
}

/// All primitive roots modulo n in [start, stop).
pub fn primitive_roots(n: u64, start: u64, stop: u64) -> Vec<u64> {
    (start..stop)
        .filter(|&g| is_primitive_root(g, n))
        .collect()
}

/// Find primitive element of extension field defined by irreducible polynomial.
pub fn primitive_element(
    field: &crate::field::GaloisField,
) -> u64 {
    field.primitive_element()
}

/// Whether element is primitive in the given field.
pub fn is_primitive_element(value: u64, field: &crate::field::GaloisField) -> bool {
    if value == 0 {
        return false;
    }
    let q = field.order();
    let (factors, _) = factorize(q - 1).unwrap_or((vec![q - 1], vec![1]));
    for &p in &factors {
        if field.pow(value, (q - 1) / p) == 1 {
            return false;
        }
    }
    field.pow(value, q - 1) == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_root_mod_11() {
        assert!(is_primitive_root(2, 11));
        assert!(!is_primitive_root(3, 11));
    }
}
