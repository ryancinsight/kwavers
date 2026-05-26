//! PyO3 bindings for the theranostic inverse pipeline.
//!
//! Submodule topology:
//! - `run`               — `run_theranostic_inverse_from_ritk` pyfunction
//! - `result_serializer` — `result_to_dict` + private geometry helpers

pub(super) mod result_serializer;
mod run;

pub use run::run_theranostic_inverse_from_ritk;
