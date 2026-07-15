//! Modular arithmetic and number theory functions.

use crate::error::{GaloisError, Result};
use crate::poly::mod_inverse;
use crate::prime::{factorize, is_prime};

/// Greatest common divisor.
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Extended GCD: returns (g, s, t) such that a*s + b*t = g = gcd(a,b).
pub fn egcd(a: u64, b: u64) -> (u64, i64, i64) {
    let (mut old_r, mut r) = (a as i128, b as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    let (mut old_t, mut t) = (0i128, 1i128);
    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
        (old_t, t) = (t, old_t - quotient * t);
    }
    (
        old_r as u64,
        old_s as i64,
        old_t as i64,
    )
}

/// Least common multiple.
pub fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    a / gcd(a, b) * b
}

/// Euler's totient function phi(n).
pub fn euler_phi(n: u64) -> u64 {
    if n < 1 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let (factors, mults) = factorize(n).unwrap_or((vec![n], vec![1]));
    let mut result = n;
    for (&p, &m) in factors.iter().zip(mults.iter()) {
        let _ = m;
        result = result / p * (p - 1);
    }
    result
}

/// Möbius function mu(n).
pub fn mobius(n: u64) -> i32 {
    if n == 1 {
        return 1;
    }
    let (factors, mults) = factorize(n).unwrap_or((vec![n], vec![1]));
    for &m in &mults {
        if m > 1 {
            return 0;
        }
    }
    if factors.len() % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Carmichael lambda function.
pub fn carmichael_lambda(n: u64) -> u64 {
    if n == 1 {
        return 1;
    }
    let (factors, mults) = factorize(n).unwrap_or((vec![n], vec![1]));
    let mut result = 1u64;
    for (&p, &m) in factors.iter().zip(mults.iter()) {
        let lambda = if p == 2 && m >= 3 {
            1u64 << (m - 2)
        } else {
            (p - 1) * p.pow(m - 1)
        };
        result = lcm(result, lambda);
    }
    result
}

/// Totatives: integers in [1, n) coprime to n.
pub fn totatives(n: u64) -> Vec<u64> {
    (1..n).filter(|&i| gcd(i, n) == 1).collect()
}

/// Whether arguments are pairwise coprime.
pub fn are_coprime(values: &[u64]) -> bool {
    for i in 0..values.len() {
        for j in i + 1..values.len() {
            if gcd(values[i], values[j]) != 1 {
                return false;
            }
        }
    }
    true
}

/// Legendre symbol (a/p) for odd prime p.
pub fn legendre_symbol(a: u64, p: u64) -> Result<i32> {
    if !is_prime(p) || p == 2 {
        return Err(GaloisError::NotPrime(p));
    }
    let a = a % p;
    if a == 0 {
        return Ok(0);
    }
    Ok(jacobi_symbol(a, p))
}

/// Jacobi symbol (a/n) for odd n.
pub fn jacobi_symbol(mut a: u64, mut n: u64) -> i32 {
    if n % 2 == 0 {
        return 0;
    }
    let mut result = 1i32;
    a %= n;
    while a != 0 {
        while a % 2 == 0 {
            a /= 2;
            if n % 8 == 3 || n % 8 == 5 {
                result = -result;
            }
        }
        std::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            result = -result;
        }
        a %= n;
    }
    if n == 1 { result } else { 0 }
}

/// Kronecker symbol (a/n) for all n.
pub fn kronecker_symbol(a: i64, n: i64) -> i32 {
    if n == 0 {
        return if a.abs() == 1 { 1 } else { 0 };
    }
    if n == 1 {
        return 1;
    }
    if n < 0 {
        return if a < 0 {
            -kronecker_symbol(a, -n)
        } else {
            kronecker_symbol(a, -n)
        };
    }
    jacobi_symbol(a.unsigned_abs(), n as u64)
}

/// Chinese Remainder Theorem: solve x ≡ residues[i] (mod moduli[i]).
pub fn crt(residues: &[u64], moduli: &[u64]) -> Result<u64> {
    if residues.len() != moduli.len() || residues.is_empty() {
        return Err(GaloisError::LengthMismatch);
    }
    let mut result = 0u64;
    let mut product = 1u64;
    for &m in moduli {
        product *= m;
    }
    for (&r, &m) in residues.iter().zip(moduli.iter()) {
        let p = product / m;
        let (_, inv, _) = egcd(p, m);
        let inv = ((inv % m as i64) + m as i64) as u64;
        result = (result + r * p * inv) % product;
    }
    Ok(result)
}

/// Whether (Z/nZ)* is cyclic.
pub fn is_cyclic(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    let (factors, mults) = factorize(n).unwrap_or((vec![n], vec![1]));
    if factors.len() == 1 {
        return factors[0] == 2 || factors[0] == 4 || mults[0] == 1;
    }
    if factors.len() == 2 && factors[0] == 2 && mults[0] <= 2 {
        return factors[1] % 2 == 1;
    }
    false
}

/// Fermat primality test.
pub fn fermat_primality_test(n: u64, base: u64) -> bool {
    if n < 2 || base % n == 0 {
        return false;
    }
    mod_pow(base % n, n - 1, n) != 1
}

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    while exp > 0 {
        if exp % 2 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exp /= 2;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn totient() {
        assert_eq!(euler_phi(12), 4);
        assert_eq!(mobius(6), 1);
    }
}
