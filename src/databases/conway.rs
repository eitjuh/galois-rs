use once_cell::sync::Lazy;
use rusqlite::{Connection, Result as SqlResult};
use std::sync::Mutex;

use super::database_path;
use crate::error::{GaloisError, Result};

static CONWAY_DB: Lazy<Mutex<Connection>> = Lazy::new(|| {
    let path = database_path("conway_polys.db");
    Mutex::new(Connection::open(path).expect("failed to open conway_polys.db"))
});

/// Look up Conway polynomial coefficients from Frank Luebeck's database.
///
/// Returns `(nonzero_degrees, nonzero_coeffs)` in descending degree order.
pub fn conway_poly_lookup(characteristic: u64, degree: u32) -> Result<(Vec<u32>, Vec<u64>)> {
    let conn = CONWAY_DB.lock().map_err(|_| GaloisError::DatabaseLock)?;
    let mut stmt = conn
        .prepare("SELECT nonzero_degrees, nonzero_coeffs FROM polys WHERE characteristic=?1 AND degree=?2")
        .map_err(|e| GaloisError::DatabaseError(e.to_string()))?;

    let result: SqlResult<(String, String)> = stmt.query_row(
        rusqlite::params![characteristic, degree],
        |row| Ok((row.get(0)?, row.get(1)?)),
    );

    match result {
        Ok((degrees, coeffs)) => Ok(parse_sparse_poly(&degrees, &coeffs)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Err(GaloisError::ConwayPolyNotFound {
            characteristic,
            degree,
        }),
        Err(e) => Err(GaloisError::DatabaseError(e.to_string())),
    }
}

fn parse_sparse_poly(degrees: &str, coeffs: &str) -> (Vec<u32>, Vec<u64>) {
    let degrees = degrees
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();
    let coeffs = coeffs
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();
    (degrees, coeffs)
}
