//! Cached Galois field instances (Python galois singleton pattern).

use std::collections::HashMap;
use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::error::Result;
use super::{BigGaloisField, GaloisField};

static FIELD_CACHE: Lazy<Mutex<HashMap<(u64, u32), GaloisField>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static BIG_FIELD_CACHE: Lazy<Mutex<HashMap<(u64, u32), BigGaloisField>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Get or create a cached `GaloisField` for GF(p^m).
///
/// Mirrors Python galois where `GF(p^m)` returns the same field class for identical parameters.
pub fn cached_field(p: u64, m: u32) -> Result<GaloisField> {
    let key = (p, m);
    let mut cache = FIELD_CACHE
        .lock()
        .map_err(|_| crate::error::GaloisError::DatabaseLock)?;
    if let Some(field) = cache.get(&key) {
        return Ok(field.clone());
    }
    let field = GaloisField::new_uncached(p, m)?;
    cache.insert(key, field.clone());
    Ok(field)
}

/// Get or create a cached `BigGaloisField` for GF(p^m).
pub fn cached_big_field(p: u64, m: u32) -> Result<BigGaloisField> {
    let key = (p, m);
    let mut cache = BIG_FIELD_CACHE
        .lock()
        .map_err(|_| crate::error::GaloisError::DatabaseLock)?;
    if let Some(field) = cache.get(&key) {
        return Ok(field.clone());
    }
    let field = BigGaloisField::new_uncached(p, m)?;
    cache.insert(key, field.clone());
    Ok(field)
}

/// Create GF(order) with caching.
pub fn cached_field_from_order(order: u64) -> Result<GaloisField> {
    let (p, m) = crate::poly::factor_prime_power(order)?;
    cached_field(p, m)
}

/// Clear the field caches (useful for tests).
pub fn clear_field_cache() {
    if let Ok(mut cache) = FIELD_CACHE.lock() {
        cache.clear();
    }
    if let Ok(mut cache) = BIG_FIELD_CACHE.lock() {
        cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_returns_same_arc_data() {
        clear_field_cache();
        let a = cached_field(3, 5).unwrap();
        let b = cached_field(3, 5).unwrap();
        assert_eq!(a.order(), b.order());
        assert_eq!(a.primitive_element(), b.primitive_element());
    }

    #[test]
    fn big_cache_returns_same_field() {
        clear_field_cache();
        let a = cached_big_field(2, 2).unwrap();
        let b = cached_big_field(2, 2).unwrap();
        assert_eq!(a.order(), b.order());
        assert_eq!(a.characteristic(), b.characteristic());
    }
}
