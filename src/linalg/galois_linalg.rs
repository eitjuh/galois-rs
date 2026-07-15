//! Linear algebra over small and big Galois fields.

use ndarray::{Array1, Array2};
use num_bigint::BigUint;
use num_traits::{One, Zero};

use crate::error::{GaloisError, Result};
use crate::field::{BigGaloisField, FieldKind, GaloisArray};
use crate::poly::FieldPoly;

use super::{
    characteristic_polynomial, lu_decomposition as small_lu, rank_revealing_lu as small_rank_lu,
    FieldMatrix, FieldVector,
};

/// Matrix over a `BigGaloisField`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BigFieldMatrix {
    field: BigGaloisField,
    data: Array2<BigUint>,
}

impl BigFieldMatrix {
    pub fn new(field: BigGaloisField, rows: usize, cols: usize, data: Vec<BigUint>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![rows, cols],
                actual: vec![data.len()],
            });
        }
        for v in &data {
            field.validate(v)?;
        }
        Ok(Self {
            field,
            data: Array2::from_shape_vec((rows, cols), data).unwrap(),
        })
    }

    pub fn from_u64(field: BigGaloisField, rows: usize, cols: usize, data: Vec<u64>) -> Result<Self> {
        Self::new(
            field,
            rows,
            cols,
            data.into_iter().map(BigUint::from).collect(),
        )
    }

    pub fn zeros(field: BigGaloisField, rows: usize, cols: usize) -> Self {
        Self {
            field,
            data: Array2::zeros((rows, cols)),
        }
    }

    pub fn identity(field: BigGaloisField, n: usize) -> Self {
        let mut m = Self::zeros(field, n, n);
        for i in 0..n {
            m.data[[i, i]] = BigUint::one();
        }
        m
    }

    pub fn field(&self) -> &BigGaloisField {
        &self.field
    }

    pub fn shape(&self) -> (usize, usize) {
        self.data.dim()
    }

    pub fn get(&self, row: usize, col: usize) -> &BigUint {
        &self.data[[row, col]]
    }

    pub fn set(&mut self, row: usize, col: usize, value: BigUint) -> Result<()> {
        self.data[[row, col]] = self.field.validate(&value)?;
        Ok(())
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        ensure_same_shape(self, other)?;
        let (r, c) = self.shape();
        let mut out = Vec::with_capacity(r * c);
        for i in 0..r {
            for j in 0..c {
                out.push(self.field.add(self.get(i, j), other.get(i, j)));
            }
        }
        Self::new(self.field.clone(), r, c, out)
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        ensure_same_shape(self, other)?;
        let (r, c) = self.shape();
        let mut out = Vec::with_capacity(r * c);
        for i in 0..r {
            for j in 0..c {
                out.push(self.field.sub(self.get(i, j), other.get(i, j)));
            }
        }
        Self::new(self.field.clone(), r, c, out)
    }

    pub fn trace(&self) -> BigUint {
        let n = self.shape().0.min(self.shape().1);
        let mut t = BigUint::zero();
        for i in 0..n {
            t = self.field.add(&t, self.get(i, i));
        }
        t
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
        let mut out = vec![BigUint::zero(); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = BigUint::zero();
                for l in 0..k {
                    sum = self.field.add(
                        &sum,
                        &self.field.mul(self.get(i, l), other.get(l, j)),
                    );
                }
                out[i * n + j] = sum;
            }
        }
        Self::new(self.field.clone(), m, n, out)
    }

    pub fn mul_array(&self, vec: &GaloisArray) -> Result<GaloisArray> {
        let values = match vec {
            GaloisArray::Big(v) => v,
            _ => return Err(GaloisError::FieldMismatch),
        };
        if self.shape().1 != values.len() {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![self.shape().1],
                actual: vec![values.len()],
            });
        }
        let m = self.shape().0;
        let mut out = Vec::with_capacity(m);
        for i in 0..m {
            let mut sum = BigUint::zero();
            for j in 0..self.shape().1 {
                sum = self.field.add(
                    &sum,
                    &self.field.mul(self.get(i, j), &values.values()[j]),
                );
            }
            out.push(sum);
        }
        Ok(GaloisArray::Big(crate::field::BigFieldArray::new(
            self.field.clone(),
            out,
        )?))
    }

    pub fn solve(&self, b: &GaloisArray) -> Result<GaloisArray> {
        let rhs = match b {
            GaloisArray::Big(v) => v,
            _ => return Err(GaloisError::FieldMismatch),
        };
        if self.shape().0 != self.shape().1 || self.shape().0 != rhs.len() {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![self.shape().0],
                actual: vec![rhs.len()],
            });
        }
        let n = self.shape().0;
        let mut aug = self.data.clone();
        let mut rhs_data: Array1<BigUint> = Array1::from_iter(rhs.values().iter().cloned());

        for col in 0..n {
            let mut pivot = col;
            while pivot < n && aug[[pivot, col]].is_zero() {
                pivot += 1;
            }
            if pivot == n {
                return Err(GaloisError::SingularMatrix);
            }
            if pivot != col {
                for j in 0..n {
                    let tmp = aug[[col, j]].clone();
                    aug[[col, j]] = aug[[pivot, j]].clone();
                    aug[[pivot, j]] = tmp;
                }
                let tmp = rhs_data[col].clone();
                rhs_data[col] = rhs_data[pivot].clone();
                rhs_data[pivot] = tmp;
            }

            let inv = self.field.div(&BigUint::one(), &aug[[col, col]])?;
            for j in 0..n {
                aug[[col, j]] = self.field.mul(&aug[[col, j]], &inv);
            }
            rhs_data[col] = self.field.mul(&rhs_data[col], &inv);

            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[[row, col]].clone();
                if factor.is_zero() {
                    continue;
                }
                for j in 0..n {
                    aug[[row, j]] = self.field.sub(
                        &aug[[row, j]],
                        &self.field.mul(&factor, &aug[[col, j]]),
                    );
                }
                rhs_data[row] = self.field.sub(
                    &rhs_data[row],
                    &self.field.mul(&factor, &rhs_data[col]),
                );
            }
        }

        Ok(GaloisArray::Big(crate::field::BigFieldArray::new(
            self.field.clone(),
            rhs_data.to_vec(),
        )?))
    }

    /// Matrix inverse for square nonsingular matrices.
    pub fn inverse(&self) -> Result<Self> {
        let n = self.shape().0;
        if n != self.shape().1 {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![n, n],
                actual: vec![self.shape().0, self.shape().1],
            });
        }
        let mut result_cols = Vec::new();
        for col in 0..n {
            let mut e = vec![BigUint::zero(); n];
            e[col] = BigUint::one();
            let b = GaloisArray::Big(crate::field::BigFieldArray::new(
                self.field.clone(),
                e,
            )?);
            result_cols.push(self.solve(&b)?.as_big().unwrap().values());
        }
        let mut flat = Vec::with_capacity(n * n);
        for row in 0..n {
            for col in 0..n {
                flat.push(result_cols[col][row].clone());
            }
        }
        Self::new(self.field.clone(), n, n, flat)
    }

    /// A basis for the null space (kernel) of the matrix.
    pub fn null_space(&self) -> Result<Vec<GaloisArray>> {
        let (rows, cols) = self.shape();
        let mut m = self.data.clone();
        let mut pivot_cols = Vec::new();
        let mut pivot_row = 0usize;
        let field = self.field.clone();

        for col in 0..cols {
            if pivot_row >= rows {
                break;
            }
            let mut pivot = pivot_row;
            while pivot < rows && m[[pivot, col]].is_zero() {
                pivot += 1;
            }
            if pivot == rows {
                continue;
            }
            if pivot != pivot_row {
                for j in 0..cols {
                    let tmp = m[[pivot_row, j]].clone();
                    m[[pivot_row, j]] = m[[pivot, j]].clone();
                    m[[pivot, j]] = tmp;
                }
            }
            let inv = field.div(&BigUint::one(), &m[[pivot_row, col]])?;
            for j in 0..cols {
                m[[pivot_row, j]] = field.mul(&m[[pivot_row, j]], &inv);
            }
            for row in 0..rows {
                if row == pivot_row {
                    continue;
                }
                let factor = m[[row, col]].clone();
                if factor.is_zero() {
                    continue;
                }
                for j in 0..cols {
                    m[[row, j]] = field.sub(
                        &m[[row, j]],
                        &field.mul(&factor, &m[[pivot_row, j]]),
                    );
                }
            }
            pivot_cols.push(col);
            pivot_row += 1;
        }

        let free_cols: Vec<usize> = (0..cols).filter(|c| !pivot_cols.contains(c)).collect();
        let mut basis = Vec::new();
        for &free in &free_cols {
            let mut vec = vec![BigUint::zero(); cols];
            vec[free] = BigUint::one();
            for (r, &pcol) in pivot_cols.iter().enumerate() {
                vec[pcol] = field.neg(&m[[r, free]]);
            }
            basis.push(GaloisArray::Big(crate::field::BigFieldArray::new(
                field.clone(),
                vec,
            )?));
        }
        Ok(basis)
    }

    pub fn det(&self) -> Result<BigUint> {
        if self.shape().0 != self.shape().1 {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![self.shape().0, self.shape().0],
                actual: vec![self.shape().0, self.shape().1],
            });
        }
        let n = self.shape().0;
        let mut m = self.data.clone();
        let mut det = BigUint::one();
        for col in 0..n {
            let mut pivot = col;
            while pivot < n && m[[pivot, col]].is_zero() {
                pivot += 1;
            }
            if pivot == n {
                return Ok(BigUint::zero());
            }
            if pivot != col {
                for j in 0..n {
                    let tmp = m[[col, j]].clone();
                    m[[col, j]] = m[[pivot, j]].clone();
                    m[[pivot, j]] = tmp;
                }
                det = self.field.neg(&det);
            }
            let pivot_val = m[[col, col]].clone();
            det = self.field.mul(&det, &pivot_val);
            let inv = self.field.div(&BigUint::one(), &pivot_val)?;
            for j in col..n {
                m[[col, j]] = self.field.mul(&m[[col, j]], &inv);
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = m[[row, col]].clone();
                if factor.is_zero() {
                    continue;
                }
                for j in col..n {
                    m[[row, j]] = self.field.sub(
                        &m[[row, j]],
                        &self.field.mul(&factor, &m[[col, j]]),
                    );
                }
            }
        }
        Ok(det)
    }

    pub fn rank(&self) -> usize {
        self.row_echelon_form().1
    }

    pub fn transpose(&self) -> Result<Self> {
        let (rows, cols) = self.shape();
        let mut out = Vec::with_capacity(rows * cols);
        for j in 0..cols {
            for i in 0..rows {
                out.push(self.get(i, j).clone());
            }
        }
        Self::new(self.field.clone(), cols, rows, out)
    }

    pub fn row_echelon_form(&self) -> (Self, usize) {
        let (rows, cols) = self.shape();
        let mut m = self.data.clone();
        let mut pivot_row = 0usize;
        let mut pivot_col = 0usize;

        while pivot_row < rows && pivot_col < cols {
            let mut pivot = pivot_row;
            while pivot < rows && m[[pivot, pivot_col]].is_zero() {
                pivot += 1;
            }
            if pivot == rows {
                pivot_col += 1;
                continue;
            }
            if pivot != pivot_row {
                for j in 0..cols {
                    let tmp = m[[pivot_row, j]].clone();
                    m[[pivot_row, j]] = m[[pivot, j]].clone();
                    m[[pivot, j]] = tmp;
                }
            }
            let inv = self
                .field
                .div(&BigUint::one(), &m[[pivot_row, pivot_col]])
                .unwrap_or_else(|_| BigUint::zero());
            for j in pivot_col..cols {
                m[[pivot_row, j]] = self.field.mul(&m[[pivot_row, j]], &inv);
            }
            for row in 0..rows {
                if row == pivot_row {
                    continue;
                }
                let factor = m[[row, pivot_col]].clone();
                if factor.is_zero() {
                    continue;
                }
                for j in pivot_col..cols {
                    m[[row, j]] = self.field.sub(
                        &m[[row, j]],
                        &self.field.mul(&factor, &m[[pivot_row, j]]),
                    );
                }
            }
            pivot_row += 1;
            pivot_col += 1;
        }

        let rank = pivot_row;
        let flat: Vec<BigUint> = m.iter().cloned().collect();
        let rref = Self::new(self.field.clone(), rows, cols, flat)
            .unwrap_or_else(|_| self.clone());
        (rref, rank)
    }

    pub fn characteristic_polynomial(&self) -> Result<FieldPoly> {
        let n = self.shape().0;
        let field = self.field.clone();
        if n != self.shape().1 {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![n, n],
                actual: vec![self.shape().0, self.shape().1],
            });
        }
        if n == 1 {
            let c = self.field.neg(self.get(0, 0));
            return Ok(FieldPoly::Big(crate::poly::BigPoly::new(
                vec![BigUint::one(), c],
                field,
            )?));
        }
        if n == 2 {
            let a = self.get(0, 0);
            let b = self.get(0, 1);
            let c = self.get(1, 0);
            let d = self.get(1, 1);
            let trace = field.add(a, d);
            let det = field.sub(&field.mul(a, d), &field.mul(b, c));
            return Ok(FieldPoly::Big(crate::poly::BigPoly::new(
                vec![BigUint::one(), field.neg(&trace), det],
                field,
            )?));
        }

        let mut companion = Self::identity(field.clone(), n);
        let mut coeffs = vec![BigUint::zero(); n];
        for k in 1..=n {
            let product = self.mul(&companion)?;
            let ck = field.div(&product.trace(), &BigUint::from(k as u64))?;
            coeffs[k - 1] = ck.clone();
            let mut scaled = Self::identity(field.clone(), n);
            for i in 0..n {
                scaled.data[[i, i]] = ck.clone();
            }
            companion = product.sub(&scaled)?;
        }
        let mut desc = vec![BigUint::one()];
        for ck in coeffs {
            desc.push(field.neg(&ck));
        }
        Ok(FieldPoly::Big(crate::poly::BigPoly::new(desc, field)?))
    }

    /// LU decomposition with partial pivoting: P A = L U.
    pub fn lu_decomposition(&self) -> Result<(Self, Self, Vec<usize>)> {
        let n = self.shape().0;
        if n != self.shape().1 {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![n, n],
                actual: vec![self.shape().0, self.shape().1],
            });
        }
        let field = self.field.clone();
        let mut a = self.data.clone();
        let mut pivots: Vec<usize> = (0..n).collect();

        for k in 0..n {
            let mut max_row = k;
            for i in (k + 1)..n {
                if !a[[i, k]].is_zero() {
                    max_row = i;
                }
            }
            if a[[max_row, k]].is_zero() {
                return Err(GaloisError::SingularMatrix);
            }
            if max_row != k {
                pivots.swap(k, max_row);
                for j in 0..n {
                    let tmp = a[[k, j]].clone();
                    a[[k, j]] = a[[max_row, j]].clone();
                    a[[max_row, j]] = tmp;
                }
            }

            let pivot = a[[k, k]].clone();
            let inv_pivot = field.div(&BigUint::one(), &pivot)?;
            for i in (k + 1)..n {
                let factor = field.mul(&a[[i, k]], &inv_pivot);
                a[[i, k]] = factor.clone();
                for j in (k + 1)..n {
                    a[[i, j]] = field.sub(&a[[i, j]], &field.mul(&factor, &a[[k, j]]));
                }
            }
        }

        let mut l_data = vec![BigUint::zero(); n * n];
        let mut u_data = vec![BigUint::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    l_data[i * n + j] = a[[i, j]].clone();
                } else if i == j {
                    l_data[i * n + j] = BigUint::one();
                    u_data[i * n + j] = a[[i, j]].clone();
                } else {
                    u_data[i * n + j] = a[[i, j]].clone();
                }
            }
        }

        let l = Self::new(field.clone(), n, n, l_data)?;
        let u = Self::new(field, n, n, u_data)?;
        Ok((l, u, pivots))
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
        let field = self.field.clone();
        let mut a = self.data.clone();
        let mut pivots: Vec<usize> = (0..n).collect();
        let mut rank = 0usize;

        for k in 0..n {
            let mut max_row = k;
            for i in (k + 1)..n {
                if !a[[i, k]].is_zero() {
                    max_row = i;
                }
            }
            if a[[max_row, k]].is_zero() {
                continue;
            }
            if max_row != k {
                pivots.swap(k, max_row);
                for j in 0..n {
                    let tmp = a[[k, j]].clone();
                    a[[k, j]] = a[[max_row, j]].clone();
                    a[[max_row, j]] = tmp;
                }
            }

            let pivot = a[[k, k]].clone();
            let inv_pivot = field.div(&BigUint::one(), &pivot)?;
            rank += 1;
            for i in (k + 1)..n {
                let factor = field.mul(&a[[i, k]], &inv_pivot);
                a[[i, k]] = factor.clone();
                for j in (k + 1)..n {
                    a[[i, j]] = field.sub(&a[[i, j]], &field.mul(&factor, &a[[k, j]]));
                }
            }
        }

        let mut l_data = vec![BigUint::zero(); n * n];
        let mut u_data = vec![BigUint::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    l_data[i * n + j] = a[[i, j]].clone();
                } else if i == j {
                    l_data[i * n + j] = BigUint::one();
                    u_data[i * n + j] = a[[i, j]].clone();
                } else {
                    u_data[i * n + j] = a[[i, j]].clone();
                }
            }
        }

        let l = Self::new(field.clone(), n, n, l_data)?;
        let u = Self::new(field, n, n, u_data)?;
        Ok((l, u, pivots, rank))
    }

    /// Eigenvalues via the characteristic polynomial.
    pub fn eigenvalues(&self) -> Result<GaloisArray> {
        if self.shape().0 != self.shape().1 {
            return Err(GaloisError::ShapeMismatch {
                expected: vec![self.shape().0, self.shape().0],
                actual: vec![self.shape().0, self.shape().1],
            });
        }
        let cp = self.characteristic_polynomial()?;
        crate::poly::field_poly_roots(&cp)
    }
}

/// Matrix over either a `GaloisField` or `BigGaloisField`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GaloisMatrix {
    Small(FieldMatrix),
    Big(BigFieldMatrix),
}

impl GaloisMatrix {
    pub fn new(field: &FieldKind, rows: usize, cols: usize, data: Vec<u64>) -> Result<Self> {
        match field {
            FieldKind::Small(f) => Ok(Self::Small(FieldMatrix::new(
                f.clone(),
                rows,
                cols,
                data,
            )?)),
            FieldKind::Big(f) => Ok(Self::Big(BigFieldMatrix::from_u64(
                f.clone(),
                rows,
                cols,
                data,
            )?)),
        }
    }

    pub fn zeros(field: &FieldKind, rows: usize, cols: usize) -> Self {
        match field {
            FieldKind::Small(f) => Self::Small(FieldMatrix::zeros(f.clone(), rows, cols)),
            FieldKind::Big(f) => Self::Big(BigFieldMatrix::zeros(f.clone(), rows, cols)),
        }
    }

    pub fn identity(field: &FieldKind, n: usize) -> Self {
        match field {
            FieldKind::Small(f) => Self::Small(FieldMatrix::identity(f.clone(), n)),
            FieldKind::Big(f) => Self::Big(BigFieldMatrix::identity(f.clone(), n)),
        }
    }

    pub fn field(&self) -> FieldKind {
        match self {
            Self::Small(m) => FieldKind::Small(m.field().clone()),
            Self::Big(m) => FieldKind::Big(m.field().clone()),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::Small(m) => m.shape(),
            Self::Big(m) => m.shape(),
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => Ok(Self::Small(a.add(b)?)),
            (Self::Big(a), Self::Big(b)) => Ok(Self::Big(a.add(b)?)),
            _ => Err(GaloisError::FieldMismatch),
        }
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => Ok(Self::Small(a.sub(b)?)),
            (Self::Big(a), Self::Big(b)) => Ok(Self::Big(a.sub(b)?)),
            _ => Err(GaloisError::FieldMismatch),
        }
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => Ok(Self::Small(a.mul(b)?)),
            (Self::Big(a), Self::Big(b)) => Ok(Self::Big(a.mul(b)?)),
            _ => Err(GaloisError::FieldMismatch),
        }
    }

    pub fn mul_array(&self, vec: &GaloisArray) -> Result<GaloisArray> {
        match (self, vec) {
            (Self::Small(m), GaloisArray::Small(v)) => {
                let b = FieldVector::new(m.field().clone(), v.values().to_vec())?;
                let out = m.mul_vec(&b)?;
                Ok(GaloisArray::Small(crate::field::FieldArray::new(
                    m.field().clone(),
                    out.values().to_vec(),
                )))
            }
            (Self::Big(m), GaloisArray::Big(_)) => m.mul_array(vec),
            _ => Err(GaloisError::FieldMismatch),
        }
    }

    pub fn solve(&self, b: &GaloisArray) -> Result<GaloisArray> {
        match (self, b) {
            (Self::Small(m), GaloisArray::Small(v)) => {
                let rhs = FieldVector::new(m.field().clone(), v.values().to_vec())?;
                let out = m.solve(&rhs)?;
                Ok(GaloisArray::Small(crate::field::FieldArray::new(
                    m.field().clone(),
                    out.values().to_vec(),
                )))
            }
            (Self::Big(m), GaloisArray::Big(_)) => m.solve(b),
            _ => Err(GaloisError::FieldMismatch),
        }
    }

    pub fn det(&self) -> Result<GaloisElementValue> {
        match self {
            Self::Small(m) => Ok(GaloisElementValue::Small(m.det()?)),
            Self::Big(m) => Ok(GaloisElementValue::Big(m.det()?)),
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            Self::Small(m) => m.rank(),
            Self::Big(m) => m.rank(),
        }
    }

    pub fn trace(&self) -> GaloisElementValue {
        match self {
            Self::Small(m) => GaloisElementValue::Small(m.trace()),
            Self::Big(m) => GaloisElementValue::Big(m.trace()),
        }
    }

    pub fn transpose(&self) -> Result<Self> {
        match self {
            Self::Small(m) => Ok(Self::Small(m.transpose()?)),
            Self::Big(m) => Ok(Self::Big(m.transpose()?)),
        }
    }

    pub fn characteristic_polynomial(&self) -> Result<FieldPoly> {
        match self {
            Self::Small(m) => Ok(FieldPoly::Small(characteristic_polynomial(m)?)),
            Self::Big(m) => m.characteristic_polynomial(),
        }
    }

    /// LU decomposition with partial pivoting.
    pub fn lu_decomposition(&self) -> Result<(Self, Self, Vec<usize>)> {
        match self {
            Self::Small(m) => {
                let (l, u, pivots) = small_lu(m)?;
                Ok((Self::Small(l), Self::Small(u), pivots))
            }
            Self::Big(m) => {
                let (l, u, pivots) = m.lu_decomposition()?;
                Ok((Self::Big(l), Self::Big(u), pivots))
            }
        }
    }

    /// LU decomposition for rank-deficient matrices.
    pub fn rank_revealing_lu_decomposition(
        &self,
    ) -> Result<(Self, Self, Vec<usize>, usize)> {
        match self {
            Self::Small(m) => {
                let (l, u, pivots, rank) = small_rank_lu(m)?;
                Ok((Self::Small(l), Self::Small(u), pivots, rank))
            }
            Self::Big(m) => {
                let (l, u, pivots, rank) = m.rank_revealing_lu_decomposition()?;
                Ok((Self::Big(l), Self::Big(u), pivots, rank))
            }
        }
    }

    /// Eigenvalues via the characteristic polynomial.
    pub fn eigenvalues(&self) -> Result<GaloisArray> {
        match self {
            Self::Small(m) => {
                let cp = characteristic_polynomial(m)?;
                crate::poly::field_poly_roots(&FieldPoly::Small(cp))
            }
            Self::Big(m) => m.eigenvalues(),
        }
    }

    /// Matrix inverse for square nonsingular matrices.
    pub fn inverse(&self) -> Result<Self> {
        match self {
            Self::Small(m) => Ok(Self::Small(m.inverse()?)),
            Self::Big(m) => Ok(Self::Big(m.inverse()?)),
        }
    }

    /// A basis for the null space (kernel) of the matrix.
    pub fn null_space(&self) -> Result<Vec<GaloisArray>> {
        match self {
            Self::Small(m) => {
                let basis = m.null_space()?;
                Ok(basis
                    .into_iter()
                    .map(|v| {
                        GaloisArray::Small(crate::field::FieldArray::new(
                            m.field().clone(),
                            v.values().to_vec(),
                        ))
                    })
                    .collect())
            }
            Self::Big(m) => m.null_space(),
        }
    }
}

/// Determinant value over either field representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GaloisElementValue {
    Small(u64),
    Big(BigUint),
}

fn ensure_same_shape(a: &BigFieldMatrix, b: &BigFieldMatrix) -> Result<()> {
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
    use crate::field::{BigGaloisField, GaloisField};

    #[test]
    fn galois_matrix_small_solve() {
        let fk = FieldKind::Small(GaloisField::new(5, 1).unwrap());
        let a = GaloisMatrix::new(&fk, 2, 2, vec![1, 2, 3, 4]).unwrap();
        let b = fk.array([1, 2]).unwrap();
        let x = a.solve(&b).unwrap();
        let check = a.mul_array(&x).unwrap();
        assert_eq!(
            check.as_small().unwrap().values(),
            b.as_small().unwrap().values()
        );
    }

    #[test]
    fn galois_matrix_big_det() {
        let field = BigGaloisField::new(2, 2).unwrap();
        let fk = FieldKind::Big(field);
        let a = GaloisMatrix::new(&fk, 2, 2, vec![1, 1, 1, 0]).unwrap();
        match a.det().unwrap() {
            GaloisElementValue::Big(d) => assert!(!d.is_zero()),
            _ => panic!("expected big det"),
        }
        assert_eq!(a.rank(), 2);
    }

    #[test]
    fn galois_matrix_big_characteristic() {
        let field = BigGaloisField::new(2, 2).unwrap();
        let fk = FieldKind::Big(field);
        let a = GaloisMatrix::identity(&fk, 2);
        let cp = a.characteristic_polynomial().unwrap();
        assert_eq!(cp.degree(), 2);
    }

    #[test]
    fn galois_matrix_lu_and_eigenvalues() {
        let fk = FieldKind::Small(GaloisField::new(5, 1).unwrap());
        let a = GaloisMatrix::identity(&fk, 3);
        let (l, u, pivots) = a.lu_decomposition().unwrap();
        assert_eq!(pivots, vec![0, 1, 2]);
        let lu = l.mul(&u).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                match (&a, &lu) {
                    (GaloisMatrix::Small(am), GaloisMatrix::Small(lum)) => {
                        assert_eq!(am.get(i, j), lum.get(i, j));
                    }
                    _ => panic!("expected small matrices"),
                }
            }
        }
        let evals = a.eigenvalues().unwrap();
        assert!(evals.as_small().unwrap().values().contains(&1));
    }

    #[test]
    fn galois_matrix_big_eigenvalues() {
        let field = BigGaloisField::new(2, 2).unwrap();
        let fk = FieldKind::Big(field);
        let a = GaloisMatrix::identity(&fk, 2);
        let evals = a.eigenvalues().unwrap();
        assert!(!evals.as_big().unwrap().values().is_empty());
    }

    #[test]
    fn galois_matrix_inverse_and_null_space() {
        let fk = FieldKind::Small(GaloisField::new(5, 1).unwrap());
        let a = GaloisMatrix::new(&fk, 2, 2, vec![1, 2, 3, 4]).unwrap();
        let inv = a.inverse().unwrap();
        let prod = a.mul(&inv).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                match &prod {
                    GaloisMatrix::Small(m) => {
                        assert_eq!(m.get(i, j), if i == j { 1 } else { 0 });
                    }
                    _ => panic!("expected small matrix"),
                }
            }
        }

        let singular = GaloisMatrix::new(&fk, 2, 2, vec![1, 2, 2, 4]).unwrap();
        let null = singular.null_space().unwrap();
        assert_eq!(null.len(), 1);
    }

    #[test]
    fn galois_matrix_big_inverse() {
        let field = BigGaloisField::new(2, 2).unwrap();
        let fk = FieldKind::Big(field);
        let a = GaloisMatrix::new(&fk, 2, 2, vec![1, 1, 1, 0]).unwrap();
        let inv = a.inverse().unwrap();
        let prod = a.mul(&inv).unwrap();
        match &prod {
            GaloisMatrix::Big(m) => {
                assert_eq!(m.get(0, 0), &BigUint::one());
                assert_eq!(m.get(1, 1), &BigUint::one());
            }
            _ => panic!("expected big matrix"),
        }
    }

    #[test]
    fn galois_matrix_singular_lu_fails() {
        let fk = FieldKind::Small(GaloisField::new(5, 1).unwrap());
        let singular = GaloisMatrix::new(&fk, 2, 2, vec![1, 2, 2, 4]).unwrap();
        assert_eq!(singular.rank(), 1);
        assert!(matches!(
            singular.lu_decomposition(),
            Err(GaloisError::SingularMatrix)
        ));

        let big_fk = FieldKind::Big(BigGaloisField::new(2, 2).unwrap());
        let big_singular = GaloisMatrix::new(&big_fk, 2, 2, vec![1, 1, 2, 2]).unwrap();
        assert_eq!(big_singular.rank(), 1);
        assert!(matches!(
            big_singular.lu_decomposition(),
            Err(GaloisError::SingularMatrix)
        ));
    }

    #[test]
    fn galois_matrix_rank_revealing_lu() {
        let fk = FieldKind::Small(GaloisField::new(5, 1).unwrap());
        let singular = GaloisMatrix::new(&fk, 2, 2, vec![1, 2, 2, 4]).unwrap();
        let (_l, _u, _pivots, rank) = singular.rank_revealing_lu_decomposition().unwrap();
        assert_eq!(rank, singular.rank());
        assert_eq!(rank, 1);

        let full = GaloisMatrix::identity(&fk, 2);
        let (_, _, _, full_rank) = full.rank_revealing_lu_decomposition().unwrap();
        assert_eq!(full_rank, 2);
    }
}
