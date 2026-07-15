use once_cell::sync::Lazy;
use rusqlite::Connection;
use std::sync::Mutex;

use super::database_path;
use crate::error::{GaloisError, Result};

static IRREDUCIBLE_DB: Lazy<Mutex<Connection>> = Lazy::new(|| {
    let path = database_path("irreducible_polys.db");
    Mutex::new(Connection::open(path).expect("failed to open irreducible_polys.db"))
});

/// Look up a default irreducible polynomial from the database.
pub fn irreducible_poly_lookup(characteristic: u64, degree: u32) -> Result<(Vec<u32>, Vec<u64>)> {
    let conn = IRREDUCIBLE_DB
        .lock()
        .map_err(|_| GaloisError::DatabaseLock)?;
    let mut stmt = conn
        .prepare(
            "SELECT nonzero_degrees, nonzero_coeffs FROM polys WHERE characteristic=?1 AND degree=?2",
        )
        .map_err(|e| GaloisError::DatabaseError(e.to_string()))?;

    match stmt.query_row(
        rusqlite::params![characteristic, degree],
        |row| {
            let degrees: String = row.get(0)?;
            let coeffs: String = row.get(1)?;
            Ok((degrees, coeffs))
        },
    ) {
        Ok((degrees, coeffs)) => {
            let degrees: Vec<u32> = degrees.split(',').filter_map(|s| s.parse().ok()).collect();
            let coeffs: Vec<u64> = coeffs.split(',').filter_map(|s| s.parse().ok()).collect();
            Ok((degrees, coeffs))
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => Err(GaloisError::IrreduciblePolyNotFound {
            characteristic,
            degree,
        }),
        Err(e) => Err(GaloisError::DatabaseError(e.to_string())),
    }
}
