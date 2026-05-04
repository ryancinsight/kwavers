//! Backward-compatible progress module.
//!
//! `solver::interface::progress` is the authoritative progress-reporting
//! boundary. This module remains only to preserve existing public paths.

pub use crate::solver::interface::progress::{
    ConsoleProgressReporter, FieldsSummary, ProgressData, ProgressReporter, ProgressUpdate,
};
