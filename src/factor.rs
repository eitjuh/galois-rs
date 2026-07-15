//! Integer factorization utilities.

pub use crate::prime::factorize;

/// All positive divisors of n.
pub fn divisors(n: u64) -> Vec<u64> {
    if n == 0 {
        return Vec::new();
    }
    let mut divs = Vec::new();
    let mut i = 1u64;
    while i * i <= n {
        if n % i == 0 {
            divs.push(i);
            if i != n / i {
                divs.push(n / i);
            }
        }
        i += 1;
    }
    divs.sort_unstable();
    divs
}

/// Perfect power decomposition: n = c^e.
pub fn perfect_power(n: u64) -> (u64, u32) {
    if n < 2 {
        return (n, 1);
    }
    for e in 2..=63 {
        let root = crate::prime::iroot(n, e);
        if root.pow(e) == n {
            let (inner, inner_e) = perfect_power(root);
            return (inner, e * inner_e);
        }
    }
    (n, 1)
}

/// Sum of k-th powers of divisors of n.
pub fn divisor_sigma(k: u32, n: u64) -> u64 {
    divisors(n).iter().map(|&d| d.pow(k)).sum()
}

/// Whether n is square-free.
pub fn is_square_free(n: u64) -> bool {
    if n < 2 {
        return true;
    }
    let (factors, mults) = factorize(n).unwrap_or((vec![n], vec![1]));
    mults.iter().all(|&m| m <= 1) && !factors.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divisors_of_12() {
        assert_eq!(divisors(12), vec![1, 2, 3, 4, 6, 12]);
    }
}
