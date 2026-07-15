use once_cell::sync::Lazy;
use std::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoeffOrder {
    Desc,
    Asc,
}

#[derive(Debug, Clone)]
pub struct PrintOptions {
    pub coeff_order: CoeffOrder,
    pub poly_var: String,
    pub field_var: String,
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            coeff_order: CoeffOrder::Desc,
            poly_var: "x".to_string(),
            field_var: "α".to_string(),
        }
    }
}

static PRINT_OPTIONS: Lazy<RwLock<PrintOptions>> =
    Lazy::new(|| RwLock::new(PrintOptions::default()));

/// Get current print options.
pub fn get_printoptions() -> PrintOptions {
    PRINT_OPTIONS.read().unwrap().clone()
}

/// Set coefficient display order.
pub fn set_printoptions(coeff_order: CoeffOrder) {
    let mut opts = PRINT_OPTIONS.write().unwrap();
    opts.coeff_order = coeff_order;
}

/// Current polynomial variable name for display.
pub fn poly_var() -> String {
    get_printoptions().poly_var
}

/// Current field element variable name for display.
pub fn field_var() -> String {
    get_printoptions().field_var
}
