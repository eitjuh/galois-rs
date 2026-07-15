use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GaloisError {
    #[error("characteristic must be a prime >= 2, got {0}")]
    InvalidCharacteristic(u64),
    #[error("extension degree must be >= 1, got {0}")]
    InvalidDegree(u64),
    #[error("field order must be >= 2, got {0}")]
    InvalidOrder(u64),
    #[error("could not factor field order {0} into p^m with prime p")]
    UnfactorizableOrder(u64),
    #[error("value {value} is not a valid element of GF({characteristic}^{degree})")]
    InvalidElement { value: u64, characteristic: u64, degree: u32 },
    #[error("division by zero in GF({characteristic}^{degree})")]
    DivisionByZero { characteristic: u64, degree: u32 },
    #[error("irreducible polynomial must have degree {expected}, got {actual}")]
    InvalidIrreducibleDegree { expected: u32, actual: usize },
    #[error("irreducible polynomial must be monic")]
    NonMonicIrreducible,
    #[error("irreducible polynomial is not irreducible over GF({characteristic})")]
    ReduciblePolynomial { characteristic: u64 },
    #[error("arrays must belong to the same field")]
    FieldMismatch,
    #[error("arrays must have the same length")]
    LengthMismatch,
    #[error("polynomials must be over the same field")]
    PolynomialFieldMismatch,
    #[error("polynomial division by zero")]
    PolynomialDivisionByZero,
    #[error("Conway polynomial C_{{{characteristic},{degree}}} not found in database")]
    ConwayPolyNotFound { characteristic: u64, degree: u32 },
    #[error("irreducible polynomial over GF({characteristic}) of degree {degree} not found in database")]
    IrreduciblePolyNotFound { characteristic: u64, degree: u32 },
    #[error("prime factorization of {n} not found in database")]
    FactorizationNotFound { n: u64 },
    #[error("database error: {0}")]
    DatabaseError(String),
    #[error("database lock poisoned")]
    DatabaseLock,
    #[error("argument must be prime, {0} is not")]
    NotPrime(u64),
    #[error("NTT size {size} must be at least input length {input_len}")]
    NttInvalidSize { size: usize, input_len: usize },
    #[error("NTT modulus {0} must be prime")]
    NttModulusNotPrime(u64),
    #[error("NTT modulus {0} must equal m * size + 1 for some m")]
    NttModulusInvalid(u64),
    #[error("NTT modulus {modulus} must exceed max input value {max_value}")]
    NttModulusTooSmall { modulus: u64, max_value: u64 },
    #[error("NTT input must be over a prime field")]
    NttPrimeFieldRequired,
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("matrix is singular")]
    SingularMatrix,
    #[error("invalid polynomial string: {0}")]
    InvalidPolynomialString(String),
    #[error("no primitive root modulo {0}")]
    NoPrimitiveRoot(u64),
    #[error("no primitive element found in GF({characteristic}^{degree})")]
    NoPrimitiveElement { characteristic: u64, degree: u32 },
}

pub type Result<T> = std::result::Result<T, GaloisError>;
