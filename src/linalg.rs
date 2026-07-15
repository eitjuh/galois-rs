//! Linear algebra over finite fields.

mod galois_linalg;

use ndarray::{Array1, Array2};
use crate::error::{GaloisError, Result};
use crate::field::GaloisField;

/// Matrix over a Galois field stored in row-major order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldMatrix {
    field: GaloisField,
    data: Array2<u64>,
}

impl FieldMatrix {
    pub fn new(field: GaloisField, rows: usize, cols: usize, data: Vec<u64>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![rows, cols],
                actual: vec![data.len()],
            });
        }
        for &v in &data {
            field.validate_element(v)?;
        }
        Ok(Self {
            field,
            data: Array2::from_shape_vec((rows, cols), data).unwrap(),
        })
    }

    pub fn zeros(field: GaloisField, rows: usize, cols: usize) -> Self {
        Self {
            field,
            data: Array2::zeros((rows, cols)),
        }
    }

    pub fn identity(field: GaloisField, n: usize) -> Self {
        let mut m = Self::zeros(field, n, n);
        for i in 0..n {
            m.data[[i, i]] = 1;
        }
        m
    }

    pub fn field(&self) -> &GaloisField {
        &self.field
    }

    pub fn shape(&self) -> (usize, usize) {
        self.data.dim()
    }

    pub fn get(&self, row: usize, col: usize) -> u64 {
        self.data[[row, col]]
    }

    pub fn set(&mut self, row: usize, col: usize, value: u64) -> Result<()> {
        self.data[[row, col]] = self.field.validate_element(value)?;
        Ok(())
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        ensure_same_shape(self, other)?;
        let (r, c) = self.shape();
        let mut out = vec![0u64; r * c];
        for i in 0..r {
            for j in 0..c {
                out[i * c + j] = self.field.add(self.get(i, j), other.get(i, j));
            }
        }
        Self::new(self.field.clone(), r, c, out)
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        ensure_same_shape(self, other)?;
        let (r, c) = self.shape();
        let mut out = vec![0u64; r * c];
        for i in 0..r {
            for j in 0..c {
                out[i * c + j] = self.field.sub(self.get(i, j), other.get(i, j));
            }
        }
        Self::new(self.field.clone(), r, c, out)
    }

    /// Matrix trace.
    pub fn trace(&self) -> u64 {
        let n = self.shape().0.min(self.shape().1);
        let mut t = 0u64;
        for i in 0..n {
            t = self.field.add(t, self.get(i, i));
        }
        t
    }

    /// A basis for the null space (kernel) of the matrix.
    pub fn null_space(&self) -> Result<Vec<FieldVector>> {
        let (rows, cols) = self.shape();
        let mut m = self.data.clone();
        let mut pivot_cols = Vec::new();
        let mut pivot_row = 0usize;

        for col in 0..cols {
            if pivot_row >= rows {
                break;
            }
            let mut pivot = pivot_row;
            while pivot < rows && m[[pivot, col]] == 0 {
                pivot += 1;
            }
            if pivot == rows {
                continue;
            }
            if pivot != pivot_row {
                for j in 0..cols {
                    let tmp = m[[pivot_row, j]];
                    m[[pivot_row, j]] = m[[pivot, j]];
                    m[[pivot, j]] = tmp;
                }
            }
            let inv = self.field.div(1, m[[pivot_row, col]])?;
            for j in 0..cols {
                m[[pivot_row, j]] = self.field.mul(m[[pivot_row, j]], inv);
            }
            for row in 0..rows {
                if row == pivot_row {
                    continue;
                }
                let factor = m[[row, col]];
                if factor == 0 {
                    continue;
                }
                for j in 0..cols {
                    m[[row, j]] = self.field.sub(
                        m[[row, j]],
                        self.field.mul(factor, m[[pivot_row, j]]),
                    );
                }
            }
            pivot_cols.push(col);
            pivot_row += 1;
        }

        let free_cols: Vec<usize> = (0..cols).filter(|c| !pivot_cols.contains(c)).collect();
        let mut basis = Vec::new();
        for &free in &free_cols {
            let mut vec = vec![0u64; cols];
            vec[free] = 1;
            for (r, &pcol) in pivot_cols.iter().enumerate() {
                vec[pcol] = self.field.neg(m[[r, free]]);
            }
            basis.push(FieldVector::new(self.field.clone(), vec)?);
        }
        Ok(basis)
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        if self.shape().1 != other.shape().0 {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![self.shape().0, other.shape().1],
                actual: vec![self.shape().1, other.shape().0],
            });
        }
        let (m, k) = self.shape();
        let (_, n) = other.shape();
        let mut out = vec![0u64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0u64;
                for l in 0..k {
                    sum = self.field.add(sum, self.field.mul(self.get(i, l), other.get(l, j)));
                }
                out[i * n + j] = sum;
            }
        }
        Self::new(self.field.clone(), m, n, out)
    }

    pub fn mul_vec(&self, vec: &FieldVector) -> Result<FieldVector> {
        if self.shape().1 != vec.len() {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![self.shape().1],
                actual: vec![vec.len()],
            });
        }
        let m = self.shape().0;
        let mut out = vec![0u64; m];
        for i in 0..m {
            let mut sum = 0u64;
            for j in 0..self.shape().1 {
                sum = self.field.add(sum, self.field.mul(self.get(i, j), vec.get(j)));
            }
            out[i] = sum;
        }
        FieldVector::new(self.field.clone(), out)
    }

    /// Solve Ax = b via Gaussian elimination.
    pub fn solve(&self, b: &FieldVector) -> Result<FieldVector> {
        if self.shape().0 != self.shape().1 || self.shape().0 != b.len() {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![self.shape().0],
                actual: vec![b.len()],
            });
        }
        let n = self.shape().0;
        let mut aug = self.data.clone();
        let mut rhs: Array1<u64> = b.data.clone();

        for col in 0..n {
            let mut pivot = col;
            while pivot < n && aug[[pivot, col]] == 0 {
                pivot += 1;
            }
            if pivot == n {
                return Err(GaloisError::SingularMatrix);
            }
            if pivot != col {
                for j in 0..n {
                    let tmp = aug[[col, j]];
                    aug[[col, j]] = aug[[pivot, j]];
                    aug[[pivot, j]] = tmp;
                }
                let tmp = rhs[col];
                rhs[col] = rhs[pivot];
                rhs[pivot] = tmp;
            }

            let inv = self.field.div(1, aug[[col, col]])?;
            for j in 0..n {
                aug[[col, j]] = self.field.mul(aug[[col, j]], inv);
            }
            rhs[col] = self.field.mul(rhs[col], inv);

            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[[row, col]];
                if factor == 0 {
                    continue;
                }
                for j in 0..n {
                    aug[[row, j]] = self.field.sub(
                        aug[[row, j]],
                        self.field.mul(factor, aug[[col, j]]),
                    );
                }
                rhs[row] = self.field.sub(rhs[row], self.field.mul(factor, rhs[col]));
            }
        }

        FieldVector::new(self.field.clone(), rhs.to_vec())
    }

    pub fn inverse(&self) -> Result<Self> {
        let n = self.shape().0;
        if n != self.shape().1 {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![n, n],
                actual: vec![self.shape().0, self.shape().1],
            });
        }
        let _identity = Self::identity(self.field.clone(), n);
        let mut result_cols = Vec::new();
        for col in 0..n {
            let mut e = vec![0u64; n];
            e[col] = 1;
            let b = FieldVector::new(self.field.clone(), e)?;
            result_cols.push(self.solve(&b)?.data.to_vec());
        }
        let mut flat = Vec::with_capacity(n * n);
        for row in 0..n {
            for col in 0..n {
                flat.push(result_cols[col][row]);
            }
        }
        Self::new(self.field.clone(), n, n, flat)
    }

    /// Determinant via Gaussian elimination.
    pub fn det(&self) -> Result<u64> {
        if self.shape().0 != self.shape().1 {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![self.shape().0, self.shape().0],
                actual: vec![self.shape().0, self.shape().1],
            });
        }
        let n = self.shape().0;
        let mut m = self.data.clone();
        let mut det = 1u64;
        for col in 0..n {
            let mut pivot = col;
            while pivot < n && m[[pivot, col]] == 0 {
                pivot += 1;
            }
            if pivot == n {
                return Ok(0);
            }
            if pivot != col {
                for j in 0..n {
                    let tmp = m[[col, j]];
                    m[[col, j]] = m[[pivot, j]];
                    m[[pivot, j]] = tmp;
                }
                det = self.field.neg(det);
            }
            let pivot_val = m[[col, col]];
            det = self.field.mul(det, pivot_val);
            let inv = self.field.div(1, pivot_val)?;
            for j in col..n {
                m[[col, j]] = self.field.mul(m[[col, j]], inv);
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = m[[row, col]];
                if factor == 0 {
                    continue;
                }
                for j in col..n {
                    m[[row, j]] = self.field.sub(
                        m[[row, j]],
                        self.field.mul(factor, m[[col, j]]),
                    );
                }
            }
        }
        Ok(det)
    }

    pub fn rank(&self) -> usize {
        self.row_echelon_form().1
    }

    /// Transpose of the matrix.
    pub fn transpose(&self) -> Result<Self> {
        let (rows, cols) = self.shape();
        let mut out = vec![0u64; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = self.get(i, j);
            }
        }
        Self::new(self.field.clone(), cols, rows, out)
    }

    /// Reduced row echelon form and pivot column indices.
    pub fn row_echelon_form(&self) -> (Self, usize) {
        let (rows, cols) = self.shape();
        let mut m = self.data.clone();
        let mut pivot_row = 0usize;
        let mut pivot_col = 0usize;

        while pivot_row < rows && pivot_col < cols {
            let mut pivot = pivot_row;
            while pivot < rows && m[[pivot, pivot_col]] == 0 {
                pivot += 1;
            }
            if pivot == rows {
                pivot_col += 1;
                continue;
            }
            if pivot != pivot_row {
                for j in 0..cols {
                    let tmp = m[[pivot_row, j]];
                    m[[pivot_row, j]] = m[[pivot, j]];
                    m[[pivot, j]] = tmp;
                }
            }
            let inv = self.field.div(1, m[[pivot_row, pivot_col]]).unwrap_or(0);
            for j in pivot_col..cols {
                m[[pivot_row, j]] = self.field.mul(m[[pivot_row, j]], inv);
            }
            for row in 0..rows {
                if row == pivot_row {
                    continue;
                }
                let factor = m[[row, pivot_col]];
                if factor == 0 {
                    continue;
                }
                for j in pivot_col..cols {
                    m[[row, j]] = self.field.sub(
                        m[[row, j]],
                        self.field.mul(factor, m[[pivot_row, j]]),
                    );
                }
            }
            pivot_row += 1;
            pivot_col += 1;
        }

        let rank = pivot_row;
        let flat: Vec<u64> = m.iter().copied().collect();
        let rref = Self::new(self.field.clone(), rows, cols, flat).unwrap_or_else(|_| self.clone());
        (rref, rank)
    }

    /// Eigenvalues via the characteristic polynomial.
    pub fn eigenvalues(&self) -> Result<Vec<u64>> {
        if self.shape().0 != self.shape().1 {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![self.shape().0, self.shape().0],
                actual: vec![self.shape().0, self.shape().1],
            });
        }
        let char_poly = characteristic_polynomial(self)?;
        crate::poly::poly_roots(&char_poly)
    }

    /// LU decomposition with partial pivoting: P A = L U.
    /// Returns lower triangular L, upper triangular U, and pivot row indices.
    pub fn lu_decomposition(&self) -> Result<(Self, Self, Vec<usize>)> {
        let n = self.shape().0;
        if n != self.shape().1 {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![n, n],
                actual: vec![self.shape().0, self.shape().1],
            });
        }
        lu_decomposition(self)
    }

    /// LU decomposition for rank-deficient matrices.
    pub fn rank_revealing_lu_decomposition(
        &self,
    ) -> Result<(Self, Self, Vec<usize>, usize)> {
        let n = self.shape().0;
        if n != self.shape().1 {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![n, n],
                actual: vec![self.shape().0, self.shape().1],
            });
        }
        rank_revealing_lu(self)
    }
}

/// Characteristic polynomial det(xI - A) of a square matrix.
pub fn characteristic_polynomial(matrix: &FieldMatrix) -> Result<crate::poly::Poly> {
    let n = matrix.shape().0;
    let field = matrix.field().clone();
    if n == 1 {
        return crate::poly::Poly::new(vec![1, matrix.field.neg(matrix.get(0, 0))], field);
    }
    if n == 2 {
        let a = matrix.get(0, 0);
        let b = matrix.get(0, 1);
        let c = matrix.get(1, 0);
        let d = matrix.get(1, 1);
        let trace = field.add(a, d);
        let det = field.sub(field.mul(a, d), field.mul(b, c));
        return crate::poly::Poly::new(vec![1, field.neg(trace), det], field);
    }

    // Faddeev–LeVerrier: det(xI - A) = x^n - c_1 x^{n-1} - ... - c_n
    let mut companion = FieldMatrix::identity(field.clone(), n);
    let mut coeffs = vec![0u64; n];
    for k in 1..=n {
        let product = matrix.mul(&companion)?;
        let ck = field.div(product.trace(), k as u64)?;
        coeffs[k - 1] = ck;
        let mut scaled = FieldMatrix::identity(field.clone(), n);
        for i in 0..n {
            scaled.data[[i, i]] = ck;
        }
        companion = product.sub(&scaled)?;
    }
    let mut desc = vec![1u64];
    for ck in coeffs {
        desc.push(field.neg(ck));
    }
    crate::poly::Poly::new(desc, field)
}

/// LU decomposition with partial pivoting for a square matrix.
pub fn lu_decomposition(matrix: &FieldMatrix) -> Result<(FieldMatrix, FieldMatrix, Vec<usize>)> {
    let n = matrix.shape().0;
    let field = matrix.field().clone();
    let mut a = matrix.data.clone();
    let mut pivots: Vec<usize> = (0..n).collect();

    for k in 0..n {
        let mut max_row = k;
        for i in (k + 1)..n {
            if a[[i, k]] != 0 {
                max_row = i;
            }
        }
        if a[[max_row, k]] == 0 {
            return Err(GaloisError::SingularMatrix);
        }
        if max_row != k {
            pivots.swap(k, max_row);
            for j in 0..n {
                let tmp = a[[k, j]];
                a[[k, j]] = a[[max_row, j]];
                a[[max_row, j]] = tmp;
            }
        }

        let pivot = a[[k, k]];
        let inv_pivot = field.div(1, pivot)?;
        for i in (k + 1)..n {
            let factor = field.mul(a[[i, k]], inv_pivot);
            a[[i, k]] = factor;
            for j in (k + 1)..n {
                a[[i, j]] = field.sub(a[[i, j]], field.mul(factor, a[[k, j]]));
            }
        }
    }

    let mut l_data = vec![0u64; n * n];
    let mut u_data = vec![0u64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i > j {
                l_data[i * n + j] = a[[i, j]];
            } else if i == j {
                l_data[i * n + j] = 1;
                u_data[i * n + j] = a[[i, j]];
            } else {
                u_data[i * n + j] = a[[i, j]];
            }
        }
    }

    let l = FieldMatrix::new(field.clone(), n, n, l_data)?;
    let u = FieldMatrix::new(field, n, n, u_data)?;
    Ok((l, u, pivots))
}

/// LU decomposition that succeeds for rank-deficient matrices.
///
/// Returns `L`, `U`, pivot row indices, and the matrix rank.
pub fn rank_revealing_lu(
    matrix: &FieldMatrix,
) -> Result<(FieldMatrix, FieldMatrix, Vec<usize>, usize)> {
    let n = matrix.shape().0;
    let field = matrix.field().clone();
    let mut a = matrix.data.clone();
    let mut pivots: Vec<usize> = (0..n).collect();
    let mut rank = 0usize;

    for k in 0..n {
        let mut max_row = k;
        for i in (k + 1)..n {
            if a[[i, k]] != 0 {
                max_row = i;
            }
        }
        if a[[max_row, k]] == 0 {
            continue;
        }
        if max_row != k {
            pivots.swap(k, max_row);
            for j in 0..n {
                let tmp = a[[k, j]];
                a[[k, j]] = a[[max_row, j]];
                a[[max_row, j]] = tmp;
            }
        }

        let pivot = a[[k, k]];
        let inv_pivot = field.div(1, pivot)?;
        rank += 1;
        for i in (k + 1)..n {
            let factor = field.mul(a[[i, k]], inv_pivot);
            a[[i, k]] = factor;
            for j in (k + 1)..n {
                a[[i, j]] = field.sub(a[[i, j]], field.mul(factor, a[[k, j]]));
            }
        }
    }

    let mut l_data = vec![0u64; n * n];
    let mut u_data = vec![0u64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i > j {
                l_data[i * n + j] = a[[i, j]];
            } else if i == j {
                l_data[i * n + j] = 1;
                u_data[i * n + j] = a[[i, j]];
            } else {
                u_data[i * n + j] = a[[i, j]];
            }
        }
    }

    let l = FieldMatrix::new(field.clone(), n, n, l_data)?;
    let u = FieldMatrix::new(field, n, n, u_data)?;
    Ok((l, u, pivots, rank))
}

/// LU decomposition that succeeds for rank-deficient matrices.
pub fn rank_revealing_lu_decomposition(
    matrix: &FieldMatrix,
) -> Result<(FieldMatrix, FieldMatrix, Vec<usize>, usize)> {
    rank_revealing_lu(matrix)
}

/// Vector over a Galois field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldVector {
    field: GaloisField,
    data: Array1<u64>,
}

impl FieldVector {
    pub fn new(field: GaloisField, data: Vec<u64>) -> Result<Self> {
        for &v in &data {
            field.validate_element(v)?;
        }
        Ok(Self {
            field,
            data: Array1::from_vec(data),
        })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, i: usize) -> u64 {
        self.data[i]
    }

    pub fn field(&self) -> &GaloisField {
        &self.field
    }

    pub fn values(&self) -> &[u64] {
        self.data.as_slice().unwrap()
    }
}

pub use galois_linalg::{BigFieldMatrix, GaloisElementValue, GaloisMatrix};

fn ensure_same_shape(a: &FieldMatrix, b: &FieldMatrix) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(GaloisError::ShapeMismatch {
            expected: vec![a.shape().0, a.shape().1],
            actual: vec![b.shape().0, b.shape().1],
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GaloisField;

    #[test]
    fn solve_2x2() {
        let gf = GaloisField::new(5, 1).unwrap();
        // [1 2; 3 4] x = [1; 2] — det = 4-6 = -2 ≡ 3 mod 5
        let a = FieldMatrix::new(gf.clone(), 2, 2, vec![1, 2, 3, 4]).unwrap();
        let b = FieldVector::new(gf.clone(), vec![1, 2]).unwrap();
        let x = a.solve(&b).unwrap();
        let check = a.mul_vec(&x).unwrap();
        assert_eq!(check.values(), b.values());
    }

    #[test]
    fn det_and_rank() {
        let gf = GaloisField::new(5, 1).unwrap();
        let a = FieldMatrix::new(gf.clone(), 2, 2, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(a.det().unwrap(), 3);
        assert_eq!(a.rank(), 2);
        assert_eq!(a.trace(), 0);
        let singular = FieldMatrix::new(gf.clone(), 2, 2, vec![1, 2, 2, 4]).unwrap();
        assert_eq!(singular.rank(), 1);
        let null = singular.null_space().unwrap();
        assert_eq!(null.len(), 1);
    }

    #[test]
    fn eigenvalues_3x3() {
        let gf = GaloisField::new(5, 1).unwrap();
        let a = FieldMatrix::identity(gf, 3);
        let evals = a.eigenvalues().unwrap();
        assert!(evals.contains(&1));
    }

    #[test]
    fn lu_identity() {
        let gf = GaloisField::new(5, 1).unwrap();
        let a = FieldMatrix::identity(gf, 3);
        let (l, u, pivots) = a.lu_decomposition().unwrap();
        assert_eq!(pivots, vec![0, 1, 2]);
        assert_eq!(l.get(1, 0), 0);
        assert_eq!(u.get(1, 0), 0);
        let lu = l.mul(&u).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(lu.get(i, j), a.get(i, j));
            }
        }
    }
}
