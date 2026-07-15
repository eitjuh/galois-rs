# galois-rs

A Rust library for Galois fields GF(p^m), a Rust equivalent of the Python [galois](https://mhostetter.github.io/galois/) package.

## Features

### Galois fields
- GF(p) and GF(p^m) with runtime construction
- Multi-dimensional `FieldArray` with NumPy-style broadcasting
- Operator traits: `&a + &b`, `&a * &b`, etc.
- `sqrt`, `log`, `multiplicative_order` on arrays
- Conway / irreducible polynomial databases (SQLite)
- BigInt infrastructure for fields exceeding u64

### Polynomials
- `Poly` over GF(p^m) with full arithmetic
- `Poly::from_str("x^3 + x + 1", gf)?`
- `conway_poly`, `irreducible_poly`, `primitive_poly`
- `square_free_factorization`, `poly_factors`, `poly_roots`

### Transforms & codes
- NTT / INTT
- Reed-Solomon and BCH encode/decode
- Berlekamp-Massey, FLFSR, GLFSR

### Linear algebra & number theory
- Matrix solve/inverse over GF(p^m)
- Primes, factorization, GCD, Euler phi, CRT, primitive roots
- Normal elements

## Quick start

```rust
use galois::GaloisField;

let gf = GaloisField::new(3, 5).unwrap();
let x = gf.array([236, 87, 38, 112]).unwrap();
let y = gf.array([109, 17, 108, 224]).unwrap();

// Broadcasting: add scalar to array
let two = gf.array([2]).unwrap();
assert_eq!(x.add(&two).unwrap().values(), &[235, 89, 37, 111]);

// Sqrt (matches Python np.sqrt)
let roots = x.sqrt().unwrap();
assert_eq!(roots.mul(&roots).unwrap().values(), x.values());
```

## Python galois mapping

| Python | Rust |
|--------|------|
| `galois.GF(p**m)` | `GaloisField::new(p, m)?` |
| `GF([a,b,c])` | `gf.array([a,b,c])?` |
| `GF(shape, data)` | `gf.array_shape(&shape, data)?` |
| `x + y` (broadcast) | `x.add(&y)?` |
| `np.sqrt(x)` | `x.sqrt()?` |
| `np.log(x)` | `x.log()?` |
| `galois.Poly.Str("x^2+1")` | `Poly::from_str("x^2+1", gf)?` |
| `galois.conway_poly(p,m)` | `conway_poly(p, m)?` |
| `galois.ntt(x)` | `ntt(x.values(), None, None)?` |
| `RS.decode(cw)` | `rs.decode(&cw)?` |

## License

MIT
