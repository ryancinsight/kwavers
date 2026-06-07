//! Micromachined ultrasonic transducer (MUT) element models.
//!
//! First-principles, closed-form models for **CMUT** (capacitive) and **PMUT**
//! (piezoelectric) micromachined cells, plus an IVUS figure-of-merit comparison.
//! Both build on the shared clamped-circular-plate physics in [`plate`]. These
//! back the "CMUT vs PMUT" chapter's simulations and the
//! [`flexible`](crate::flexible) transducer design comparisons.

pub mod cmut;
pub mod comparison;
pub mod plate;
pub mod pmut;

pub use cmut::CmutCell;
pub use comparison::{evaluate_ivus, IvusVerdict, IvusWeights, MutKind};
pub use pmut::{PiezoFilm, PmutCell};
