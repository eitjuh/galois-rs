//! Prime number generation and primality testing.

use crate::databases::prime_factors_lookup;
use crate::error::{GaloisError, Result};
use crate::poly::is_prime as is_prime_u64;

pub use crate::poly::is_prime;

/// Miller-Rabin primality test.
pub fn is_prime_mr(n: u64, rounds: u32) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }

    let mut d = n - 1;
    let mut r = 0u32;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }

    'next: for _ in 0..rounds {
        let a = 2 + (simple_hash(n) % (n - 3));
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'next;
            }
        }
        return false;
    }
    true
}

fn simple_hash(n: u64) -> u64 {
    (n.wrapping_mul(0x9E3779B97F4A7C15) >> 32) | 1
}

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = mul_mod(result, base, modulus);
        }
        base = mul_mod(base, base, modulus);
        exp /= 2;
    }
    result
}

fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Returns whether n is composite (not prime and n >= 4, or n < 2).
pub fn is_composite(n: u64) -> bool {
    n < 2 || (n > 3 && !is_prime(n))
}

/// Returns whether n = p^k for some prime p and k >= 1.
pub fn is_prime_power(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    match crate::poly::factor_prime_power(n) {
        Ok(_) => true,
        Err(_) => false,
    }
}

/// Returns whether n = c^e with e > 1.
pub fn is_perfect_power(n: u64) -> bool {
    if n < 4 {
        return false;
    }
    for e in 2..=63 {
        let root = isqrt(n);
        for b in 2..=root {
            let mut power = 1u64;
            for _ in 0..e {
                power = power.saturating_mul(b);
            }
            if power == n {
                return true;
            }
            if power > n {
                break;
            }
        }
    }
    false
}

/// Integer square root: floor(sqrt(n)).
pub fn isqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Floor log base b.
pub fn ilog(n: u64, b: u64) -> u64 {
    if n < 1 || b < 2 {
        return 0;
    }
    let mut result = 0u64;
    let mut power = 1u64;
    while power <= n / b {
        power *= b;
        result += 1;
    }
    result
}

/// Floor n^(1/k).
pub fn iroot(n: u64, k: u32) -> u64 {
    if k == 0 {
        return 0;
    }
    if k == 1 {
        return n;
    }
    if n == 0 {
        return 0;
    }
    let mut x = n;
    for _ in 0..64 {
        let x_new = ((k as u64 - 1) * x + n / x.pow(k - 1)) / k as u64;
        if x_new >= x {
            break;
        }
        x = x_new;
    }
    while x.pow(k) > n {
        x -= 1;
    }
    x
}

/// Next prime greater than n.
pub fn next_prime(n: u64) -> u64 {
    let mut candidate = if n < 2 { 2 } else { n + 1 };
    if candidate % 2 == 0 {
        candidate += 1;
    }
    while !is_prime(candidate) {
        candidate += 2;
    }
    candidate
}

/// Previous prime <= n.
pub fn prev_prime(n: u64) -> Option<u64> {
    if n < 2 {
        return None;
    }
    let mut candidate = if n % 2 == 0 { n - 1 } else { n };
    while candidate >= 2 {
        if is_prime(candidate) {
            return Some(candidate);
        }
        candidate = candidate.saturating_sub(2);
    }
    Some(2)
}

/// All primes <= n.
pub fn primes_up_to(n: u64) -> Vec<u64> {
    if n < 2 {
        return Vec::new();
    }
    let mut sieve = vec![true; (n + 1) as usize];
    sieve[0] = false;
    sieve[1] = false;
    let limit = isqrt(n) as usize;
    for i in 2..=limit {
        if sieve[i] {
            let mut j = i * i;
            while j <= n as usize {
                sieve[j] = false;
                j += i;
            }
        }
    }
    (2..=n).filter(|&p| sieve[p as usize]).collect()
}

/// k-th prime (1-indexed: k=1 -> 2).
pub fn kth_prime(k: u64) -> Option<u64> {
    if k == 0 {
        return None;
    }
    let mut count = 0u64;
    let mut candidate = 2u64;
    loop {
        if is_prime(candidate) {
            count += 1;
            if count == k {
                return Some(candidate);
            }
        }
        candidate += 1;
        if candidate == 0 {
            return None;
        }
    }
}

/// Prime factorization using database lookup with trial division fallback.
pub fn factorize(n: u64) -> Result<(Vec<u64>, Vec<u32>)> {
    if n < 2 {
        return Ok((Vec::new(), Vec::new()));
    }
    if let Ok((factors, mults, _)) = prime_factors_lookup(n) {
        return Ok((factors, mults));
    }
    Ok(trial_factorize(n))
}

fn trial_factorize(mut n: u64) -> (Vec<u64>, Vec<u32>) {
    let mut factors = Vec::new();
    let mut mults = Vec::new();
    let mut d = 2u64;
    while d * d <= n {
        if n % d == 0 {
            let mut m = 0;
            while n % d == 0 {
                n /= d;
                m += 1;
            }
            factors.push(d);
            mults.push(m);
        }
        d += if d == 2 { 1 } else { 2 };
    }
    if n > 1 {
        factors.push(n);
        mults.push(1);
    }
    (factors, mults)
}

/// Product of arguments.
pub fn prod(values: &[u64]) -> u64 {
    values.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes() {
        assert_eq!(next_prime(10), 11);
        assert_eq!(prev_prime(10), Some(7));
        assert!(is_prime(97));
        assert!(!is_prime(99));
    }
}
