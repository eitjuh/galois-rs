//! Basic math utilities.

/// Absolute value for i64.
pub fn abs_i64(x: i64) -> i64 {
    x.abs()
}

/// Sign of integer: -1, 0, or 1.
pub fn sign_i64(x: i64) -> i32 {
    if x > 0 {
        1
    } else if x < 0 {
        -1
    } else {
        0
    }
}

/// Binomial coefficient C(n, k) mod p.
pub fn binomial(n: u64, k: u64, p: u64) -> u64 {
    if k > n {
        return 0;
    }
    let mut num = 1u64;
    let mut den = 1u64;
    for i in 0..k {
        num = (num * (n - i)) % p;
        den = (den * (i + 1)) % p;
    }
    crate::poly::mod_inverse(den, p).map(|inv| (num * inv) % p).unwrap_or(0)
}
