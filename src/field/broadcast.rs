//! Broadcasting utilities for field arrays (NumPy-style).

use num_bigint::BigUint;

use crate::error::{GaloisError, Result};

/// Compute the broadcast shape of two array shapes (NumPy rules).
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut result = vec![1usize; max_len];

    for i in 0..max_len {
        let da = if i < a.len() {
            a[a.len() - 1 - i]
        } else {
            1
        };
        let db = if i < b.len() {
            b[b.len() - 1 - i]
        } else {
            1
        };
        if da != db && da != 1 && db != 1 {
            return Err(GaloisError::ShapeMismatch {
                expected: a.to_vec(),
                actual: b.to_vec(),
            });
        }
        result[max_len - 1 - i] = da.max(db);
    }
    Ok(result)
}

/// Map a linear index in the output shape to an index in a source shape (broadcasting).
pub fn broadcast_index(output_shape: &[usize], input_shape: &[usize], linear_idx: usize) -> usize {
    let out_coords = linear_to_coords(linear_idx, output_shape);
    let in_coords = broadcast_coords(&out_coords, input_shape, output_shape.len());
    coords_to_linear(&in_coords, input_shape)
}

pub fn linear_to_coords(linear: usize, shape: &[usize]) -> Vec<usize> {
    let strides = row_major_strides(shape);
    let mut rem = linear;
    shape
        .iter()
        .zip(strides.iter())
        .map(|(&dim, &stride)| {
            let c = rem / stride;
            rem %= stride;
            c.min(dim.saturating_sub(1))
        })
        .collect()
}

pub fn coords_to_linear(coords: &[usize], shape: &[usize]) -> usize {
    let strides = row_major_strides(shape);
    coords.iter().zip(strides.iter()).map(|(&c, &s)| c * s).sum()
}

pub fn broadcast_coords(out_coords: &[usize], input_shape: &[usize], output_ndim: usize) -> Vec<usize> {
    let pad = output_ndim.saturating_sub(input_shape.len());
    out_coords
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let input_i = i.saturating_sub(pad);
            if input_i >= input_shape.len() {
                0
            } else {
                let dim = input_shape[input_i];
                if dim == 1 {
                    0
                } else {
                    c
                }
            }
        })
        .collect()
}

pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn row_major_strides_padded(shape: &[usize], ndim: usize) -> Vec<usize> {
    let pad = ndim.saturating_sub(shape.len());
    let mut padded: Vec<usize> = std::iter::repeat_n(1, pad).chain(shape.iter().copied()).collect();
    if padded.is_empty() {
        padded.push(1);
    }
    row_major_strides(&padded)
}

pub fn total_size(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Apply a binary field operation with broadcasting.
pub fn broadcast_binary<F>(a_shape: &[usize], a: &[u64], b_shape: &[usize], b: &[u64], op: F) -> Result<(Vec<usize>, Vec<u64>)>
where
    F: Fn(u64, u64) -> u64,
{
    let out_shape = broadcast_shapes(a_shape, b_shape)?;
    let out_len = total_size(&out_shape);
    let mut out = Vec::with_capacity(out_len);

    for linear in 0..out_len {
        let out_coords = linear_to_coords(linear, &out_shape);
        let a_coords = broadcast_coords(&out_coords, a_shape, out_shape.len());
        let b_coords = broadcast_coords(&out_coords, b_shape, out_shape.len());
        let av = a[coords_to_linear(&a_coords, a_shape)];
        let bv = b[coords_to_linear(&b_coords, b_shape)];
        out.push(op(av, bv));
    }
    Ok((out_shape, out))
}

/// Apply a binary field operation with broadcasting, where op returns Result.
pub fn broadcast_binary_result<F>(a_shape: &[usize], a: &[u64], b_shape: &[usize], b: &[u64], op: F) -> Result<(Vec<usize>, Vec<u64>)>
where
    F: Fn(u64, u64) -> Result<u64>,
{
    let out_shape = broadcast_shapes(a_shape, b_shape)?;
    let out_len = total_size(&out_shape);
    let mut out = Vec::with_capacity(out_len);

    for linear in 0..out_len {
        let out_coords = linear_to_coords(linear, &out_shape);
        let a_coords = broadcast_coords(&out_coords, a_shape, out_shape.len());
        let b_coords = broadcast_coords(&out_coords, b_shape, out_shape.len());
        let av = a[coords_to_linear(&a_coords, a_shape)];
        let bv = b[coords_to_linear(&b_coords, b_shape)];
        out.push(op(av, bv)?);
    }
    Ok((out_shape, out))
}

/// Apply a binary operation with broadcasting over `BigUint` slices.
pub fn broadcast_big<F>(
    a_shape: &[usize],
    a: &[BigUint],
    b_shape: &[usize],
    b: &[BigUint],
    op: F,
) -> Result<(Vec<usize>, Vec<BigUint>)>
where
    F: Fn(&BigUint, &BigUint) -> BigUint,
{
    let out_shape = broadcast_shapes(a_shape, b_shape)?;
    let out_len = total_size(&out_shape);
    let mut out = Vec::with_capacity(out_len);

    for linear in 0..out_len {
        let out_coords = linear_to_coords(linear, &out_shape);
        let a_coords = broadcast_coords(&out_coords, a_shape, out_shape.len());
        let b_coords = broadcast_coords(&out_coords, b_shape, out_shape.len());
        let av = &a[coords_to_linear(&a_coords, a_shape)];
        let bv = &b[coords_to_linear(&b_coords, b_shape)];
        out.push(op(av, bv));
    }
    Ok((out_shape, out))
}

/// Apply a binary operation with broadcasting, where op returns Result.
pub fn broadcast_big_result<F>(
    a_shape: &[usize],
    a: &[BigUint],
    b_shape: &[usize],
    b: &[BigUint],
    op: F,
) -> Result<(Vec<usize>, Vec<BigUint>)>
where
    F: Fn(&BigUint, &BigUint) -> Result<BigUint>,
{
    let out_shape = broadcast_shapes(a_shape, b_shape)?;
    let out_len = total_size(&out_shape);
    let mut out = Vec::with_capacity(out_len);

    for linear in 0..out_len {
        let out_coords = linear_to_coords(linear, &out_shape);
        let a_coords = broadcast_coords(&out_coords, a_shape, out_shape.len());
        let b_coords = broadcast_coords(&out_coords, b_shape, out_shape.len());
        let av = &a[coords_to_linear(&a_coords, a_shape)];
        let bv = &b[coords_to_linear(&b_coords, b_shape)];
        out.push(op(av, bv)?);
    }
    Ok((out_shape, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_1d_with_scalar() {
        let shape = broadcast_shapes(&[4], &[1]).unwrap();
        assert_eq!(shape, vec![4]);
    }

    #[test]
    fn broadcast_matrix_row() {
        let shape = broadcast_shapes(&[3, 1], &[3]).unwrap();
        assert_eq!(shape, vec![3, 3]);
    }
}
