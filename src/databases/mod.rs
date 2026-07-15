mod conway;
mod irreducible;
mod prime_factors;

pub use conway::conway_poly_lookup;
pub use irreducible::irreducible_poly_lookup;
pub use prime_factors::prime_factors_lookup;

use std::path::PathBuf;

pub(crate) fn database_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join(name)
}
