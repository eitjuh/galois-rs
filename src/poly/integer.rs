use crate::error::{GaloisError, Result};

/// Coefficients in ascending order: c[0] + c[1]*x + c[2]*x^2 + ...
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerPoly {
    coeffs: Vec<u64>,
}

impl IntegerPoly {
    pub fn new(coeffs: Vec<u64>) -> Self {
        let mut poly = Self { coeffs };
        poly.strip();
        poly
    }

    pub fn zero() -> Self {
        Self { coeffs: vec![0] }
    }

    pub fn one() -> Self {
        Self { coeffs: vec![1] }
    }

    pub fn x() -> Self {
        Self { coeffs: vec![0, 1] }
    }

    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
    }

    pub fn degree(&self) -> isize {
        if self.coeffs.len() <= 1 && self.coeffs[0] == 0 {
            -1
        } else {
            self.coeffs.len() as isize - 1
        }
    }

    pub fn is_zero(&self) -> bool {
        self.degree() < 0
    }

    pub fn is_one(&self) -> bool {
        self.coeffs == [1]
    }

    fn strip(&mut self) {
        while self.coeffs.len() > 1 && *self.coeffs.last().unwrap() == 0 {
            self.coeffs.pop();
        }
        if self.coeffs.is_empty() {
            self.coeffs.push(0);
        }
    }

    pub fn mod_coeffs(&self, p: u64) -> Vec<u64> {
        self.coeffs.iter().map(|c| c % p).collect()
    }

    pub fn eval_mod(&self, x: u64, p: u64) -> u64 {
        let mut result = 0u64;
        for coeff in self.coeffs.iter().rev() {
            result = (result * x + *coeff) % p;
        }
        result
    }

    pub fn format_mod(&self, p: u64, var: &str) -> String {
        if self.is_zero() {
            return "0".to_string();
        }

        let coeffs = self.mod_coeffs(p);
        let mut terms = Vec::new();
        for (power, coeff) in coeffs.iter().enumerate() {
            let c = *coeff % p;
            if c == 0 {
                continue;
            }
            let coeff_str = if c == 1 && power > 0 {
                String::new()
            } else {
                c.to_string()
            };
            let var_str = match power {
                0 => String::new(),
                1 => var.to_string(),
                _ => format!("{var}^{power}"),
            };
            terms.push(match (coeff_str.is_empty(), var_str.is_empty()) {
                (true, true) => "1".to_string(),
                (false, true) => coeff_str,
                (true, false) => var_str,
                (false, false) => format!("{coeff_str}{var_str}"),
            });
        }
        terms.join(" + ")
    }
}

pub fn factor_prime_power(order: u64) -> Result<(u64, u32)> {
    if order < 2 {
        return Err(GaloisError::InvalidOrder(order));
    }

    let mut n = order;
    let mut p = 0u64;
    for candidate in 2..=n {
        if n % candidate == 0 {
            if p == 0 {
                p = candidate;
            } else if candidate != p {
                return Err(GaloisError::UnfactorizableOrder(order));
            }
            n /= candidate;
            while n % candidate == 0 {
                n /= candidate;
            }
        }
    }

    if p == 0 {
        return Err(GaloisError::UnfactorizableOrder(order));
    }
    if n != 1 {
        return Err(GaloisError::UnfactorizableOrder(order));
    }

    let mut m = 0u32;
    let mut q = order;
    while q % p == 0 {
        q /= p;
        m += 1;
    }

    Ok((p, m))
}

pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3u64;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_243() {
        assert_eq!(factor_prime_power(243).unwrap(), (3, 5));
    }

    #[test]
    fn factor_prime_field() {
        assert_eq!(factor_prime_power(17).unwrap(), (17, 1));
    }
}
