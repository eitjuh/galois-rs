//! Galois fields over GF(p^m), a Rust equivalent of the Python [galois](https://mhostetter.github.io/galois/) library.
//!
//! ```
//! use galois::GaloisField;
//!
//! let gf = GaloisField::new(3, 5).unwrap();
//! let x = gf.array([236, 87, 38, 112]).unwrap();
//! let y = gf.array([109, 17, 108, 224]).unwrap();
//!
//! assert_eq!(x.add(&y).unwrap().values(), &[18, 95, 146, 0]);
//! assert_eq!(x.mul(&y).unwrap().values(), &[21, 241, 179, 82]);
//! ```

mod codes;
mod conway;
mod databases;
mod error;
mod factor;
mod field;
mod lfsr;
mod linalg;
mod math;
mod modular;
mod ntt;
mod options;
mod poly;
mod prime;
mod primitive_root;

pub use codes::{Bch, FieldBch, FieldReedSolomon, ReedSolomon};
pub use conway::{conway_poly, default_irreducible, irreducible_poly};
pub use error::{GaloisError, Result};
pub use factor::{divisor_sigma, divisors, factorize, is_square_free, perfect_power};
pub use field::{
    cached_big_field, cached_field, cached_field_from_order, clear_field_cache, field_from_order,
    galois_field, is_normal_element, normal_element, needs_bigint, BigFieldArray, BigFieldElement,
    BigGaloisField, ElementRepr, FieldArray, FieldElement, FieldKind, FieldValue, GaloisArray,
    GaloisElement, GaloisField, GF, GF2,
};
pub use lfsr::{
    berlekamp_massey, berlekamp_massey_array, FieldFlfsr, FieldGlfsr, Flfsr, Glfsr,
};
pub use linalg::{
    characteristic_polynomial, lu_decomposition, BigFieldMatrix, FieldMatrix, FieldVector,
    GaloisElementValue, GaloisMatrix,
};
pub use math::binomial;
pub use modular::{
    are_coprime, carmichael_lambda, crt, egcd, euler_phi, gcd, jacobi_symbol, kronecker_symbol,
    lcm, legendre_symbol, mobius, totatives,
};
pub use ntt::{intt, ntt};
pub use options::{get_printoptions, set_printoptions, CoeffOrder, PrintOptions};
pub use poly::{
    big_distinct_degree_factorization, big_equal_degree_factorization, big_factors,
    big_frobenius_step, big_mod_inverse_poly, big_poly_gcd, big_raise_to_field_order,
    big_roots, big_square_free_factorization, distinct_degree_factorization,
    equal_degree_factorization, factor_prime_power, field_poly_factors, field_poly_roots,
    is_primitive, lagrange_poly, lagrange_poly_array, mod_inverse, mod_inverse_poly,
    parse_field_poly, parse_poly, poly_factors, poly_from_roots, poly_from_roots_kind,
    poly_from_roots_values, poly_gcd, poly_roots, primitive_poly, square_free_factorization,
    BigPoly, FieldPoly, IntegerPoly, Poly, PrimePoly,
};
pub use prime::{
    ilog, iroot, is_composite, is_perfect_power, is_prime, is_prime_mr, is_prime_power, isqrt,
    kth_prime, next_prime, prev_prime, primes_up_to, prod,
};
pub use primitive_root::{
    is_primitive_element, is_primitive_root, primitive_element, primitive_root, primitive_roots,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
