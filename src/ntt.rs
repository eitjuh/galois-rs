//! Number-Theoretic Transform (NTT) and inverse NTT.

use crate::error::{GaloisError, Result};
use crate::field::{FieldArray, GaloisField};
use crate::primitive_root::primitive_root;
use crate::prime::is_prime;

/// Compute the Number-Theoretic Transform of `x` over GF(modulus).
pub fn ntt(x: &[u64], size: Option<usize>, modulus: Option<u64>) -> Result<FieldArray> {
    ntt_inner(x, size, modulus, true, true)
}

/// Compute the inverse Number-Theoretic Transform of `X` over GF(modulus).
pub fn intt(x: &[u64], size: Option<usize>, modulus: Option<u64>, scaled: bool) -> Result<FieldArray> {
    ntt_inner(x, size, modulus, false, scaled)
}

fn ntt_inner(
    x: &[u64],
    size: Option<usize>,
    modulus: Option<u64>,
    forward: bool,
    scaled: bool,
) -> Result<FieldArray> {
    let n = size.unwrap_or(x.len());
    if n < x.len() {
        return Err(GaloisError::NttInvalidSize {
            size: n,
            input_len: x.len(),
        });
    }

    let max_val = *x.iter().max().unwrap_or(&0);
    let p = match modulus {
        Some(m) => m,
        None => find_ntt_modulus(max_val, n)?,
    };

    if !is_prime(p) {
        return Err(GaloisError::NttModulusNotPrime(p));
    }
    if (p - 1) % n as u64 != 0 {
        return Err(GaloisError::NttModulusInvalid(p));
    }
    if p <= max_val {
        return Err(GaloisError::NttModulusTooSmall {
            modulus: p,
            max_value: max_val,
        });
    }

    let field = GaloisField::new(p, 1)?;
    let mut data: Vec<u64> = x.iter().map(|&v| field.validate_element(v)).collect::<Result<_>>()?;
    data.resize(n, 0);

    let m = (p - 1) / n as u64;
    let omega_p = primitive_root(p, 2, p)?;
    let omega_n = field.pow(omega_p, m);

    if forward {
        cooley_tukey(&mut data, &field, omega_n, false)?;
        Ok(FieldArray::new(field, data))
    } else {
        let inv_omega = field.div(1, omega_n)?;
        cooley_tukey(&mut data, &field, inv_omega, false)?;
        if scaled {
            let inv_n = field.div(1, n as u64)?;
            for v in &mut data {
                *v = field.mul(*v, inv_n);
            }
        }
        Ok(FieldArray::new(field, data))
    }
}

fn find_ntt_modulus(max_val: u64, n: usize) -> Result<u64> {
    let mut m = (max_val / n as u64).max(1);
    loop {
        let candidate = m * n as u64 + 1;
        if is_prime(candidate) && candidate > max_val {
            return Ok(candidate);
        }
        m += 1;
        if m > 1_000_000 {
            return Err(GaloisError::NttModulusInvalid(0));
        }
    }
}

fn cooley_tukey(data: &mut [u64], field: &GaloisField, omega: u64, inverse: bool) -> Result<()> {
    let n = data.len();
    if n <= 1 {
        return Ok(());
    }
    if n % 2 != 0 {
        return dft_brute_force(data, field, omega, inverse);
    }

    // Bit-reversal permutation
    let bits = (n as f64).log2() as u32;
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if i < j {
            data.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = field.pow(omega, (n / len) as u64);
        let mut w_len = 1u64;
        for i in (0..n).step_by(len) {
            let mut w = 1u64;
            for j in 0..half {
                let u = data[i + j];
                let v = field.mul(data[i + j + half], w);
                data[i + j] = field.add(u, v);
                data[i + j + half] = field.sub(u, v);
                w = field.mul(w, w_len);
            }
            w_len = field.mul(w_len, step);
        }
        len *= 2;
    }
    Ok(())
}

fn dft_brute_force(data: &mut [u64], field: &GaloisField, omega: u64, _inverse: bool) -> Result<()> {
    let n = data.len();
    let mut out = vec![0u64; n];
    for k in 0..n {
        for j in 0..n {
            let exp = (k * j) % n;
            let w = field.pow(omega, exp as u64);
            out[k] = field.add(out[k], field.mul(data[j], w));
        }
    }
    data.copy_from_slice(&out);
    Ok(())
}

fn bit_reverse(mut x: usize, bits: u32) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntt_roundtrip() {
        let x = [1, 2, 3, 4];
        let X = ntt(&x, Some(4), Some(5)).unwrap();
        let round = intt(X.values(), Some(4), Some(5), true).unwrap();
        assert_eq!(round.values(), &[1, 2, 3, 4]);
    }
}
