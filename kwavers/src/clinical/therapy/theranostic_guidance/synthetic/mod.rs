//! Synthetic CT phantoms for clinical device placement verification.
//!
//! Used as fallback data when real NIfTI patient scans are absent (e.g., during
//! CI, development, or book figure generation without proprietary data).  All
//! phantoms are dimensioned in clinically realistic ranges and satisfy the
//! invariants assumed by the placement algorithms.

mod abdominal;
mod brain;

pub use abdominal::{synthetic_abdominal_kidney_phantom, synthetic_abdominal_liver_phantom};
pub use brain::synthetic_brain_phantom;
