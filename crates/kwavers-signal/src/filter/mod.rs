//! Signal Filtering Module
//!
//! This module defines the `Filter` trait interface and provides basic
//! signal filtering implementations used across the codebase.
//!
//! ## Architecture
//!
//! - **Domain Layer** (this module): Defines `Filter` trait and basic implementations
//!   - `FrequencyFilter`: FFT-based bandpass/lowpass/highpass filtering
//! - **Analysis Layer**: Advanced filter implementations (adaptive, ML-based, etc.)
//!
//! ## Rationale for FrequencyFilter in Domain Layer
//!
//! `FrequencyFilter` is placed in the domain layer because:
//! 1. It's a fundamental signal processing primitive (not high-level analysis)
//! 2. Lower layers (solver, physics) need access to basic filtering
//! 3. It has no dependencies on higher layers (only core + domain)
//! 4. It implements the `Filter` trait which is already in domain
//!
//! This placement follows proper architectural layering:
//! - Domain (Layer 2): Basic primitives and interfaces
//! - Solver (Layer 4): Can use domain-layer filters
//! - Analysis (Layer 7): Advanced processing, ML, experimental algorithms
//!
//! ## Usage
//!
//! The `Filter` trait can be used to write filter-agnostic code:
//!
//! ```rust,no_run
//! use kwavers_signal::{Filter, FrequencyFilter};
//! use kwavers_core::error::KwaversResult;
//!
//! fn process_signal(filter: &dyn Filter, signal: &[f64], dt: f64) -> KwaversResult<Vec<f64>> {
//!     filter.apply(signal, dt)
//! }
//!
//! fn example() -> KwaversResult<()> {
//!     let freq_filter = FrequencyFilter::new();
//!     let signal = vec![1.0, 0.5, -0.3, 0.8];
//!     let dt = 0.0001;
//!
//!     let filtered = process_signal(&freq_filter, &signal, dt)?;
//!     Ok(())
//! }
//! ```

pub mod frequency_filter;

pub use frequency_filter::FrequencyFilter;

use crate::SignalPrimitive;
use kwavers_core::error::KwaversResult;

/// Trait for signal filtering operations.
///
/// Uses [`SignalPrimitive`] as its base bound (`Debug + Send + Sync`) so the
/// triple does not need to be repeated here.
///
/// # Example
///
/// ```rust,no_run
/// use kwavers_signal::{Filter, FrequencyFilter};
/// use kwavers_core::error::KwaversResult;
///
/// fn process(filter: &dyn Filter, signal: &[f64], dt: f64) -> KwaversResult<Vec<f64>> {
///     filter.apply(signal, dt)
/// }
/// ```
pub trait Filter: SignalPrimitive {
    /// Apply the filter to a time-domain signal.
    /// # Errors
    /// Returns `KwaversError` if filtering fails.
    fn apply(&self, signal: &[f64], dt: f64) -> KwaversResult<Vec<f64>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct PassThroughFilter;

    impl Filter for PassThroughFilter {
        fn apply(&self, signal: &[f64], _dt: f64) -> KwaversResult<Vec<f64>> {
            Ok(signal.to_vec())
        }
    }

    #[test]
    fn test_filter_trait_can_be_implemented() {
        let filter = PassThroughFilter;
        let signal = vec![1.0, 2.0, 3.0];
        let result = filter.apply(&signal, 0.001).unwrap();
        assert_eq!(result, signal);
    }

    #[test]
    fn test_filter_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PassThroughFilter>();
    }
}
