use once_cell::sync::Lazy;
use rusqlite::Connection;
use std::sync::Mutex;

use super::database_path;
use crate::error::{GaloisError, Result};

static PRIME_FACTORS_DB: Lazy<Mutex<Connection>> = Lazy::new(|| {
    let path = database_path("prime_factors.db");
    Mutex::new(Connection::open(path).expect("failed to open prime_factors.db"))
});

/// Look up prime factorization from the database.
///
/// Returns `(factors, multiplicities, composite)` where composite is 1 if n is composite, 0 if prime.
pub fn prime_factors_lookup(n: u64) -> Result<(Vec<u64>, Vec<u32>, bool)> {
    let conn = PRIME_FACTORS_DB
        .lock()
        .map_err(|_| GaloisError::DatabaseLock)?;
    let mut stmt = conn
        .prepare("SELECT factors, multiplicities, composite FROM factorizations WHERE value=?1")
        .map_err(|e| GaloisError::DatabaseError(e.to_string()))?;

    let n_str = n.to_string();
    match stmt.query_row(rusqlite::params![n_str], |row| {
        let factors: String = row.get(0)?;
        let multiplicities: String = row.get(1)?;
        let composite: i64 = row.get(2)?;
        Ok((factors, multiplicities, composite))
    }) {
        Ok((factors, multiplicities, composite)) => {
            let factors: Vec<u64> = factors.split(',').filter_map(|s| s.parse().ok()).collect();
            let multiplicities: Vec<u32> =
                multiplicities.split(',').filter_map(|s| s.parse().ok()).collect();
            Ok((factors, multiplicities, composite != 0))
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => Err(GaloisError::FactorizationNotFound { n }),
        Err(e) => Err(GaloisError::DatabaseError(e.to_string())),
    }
}
