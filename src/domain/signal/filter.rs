//! Signal Filtering Trait Interface
//!
//! This module defines the `Filter` trait, which represents the interface for
//! signal filtering operations. Implementations of this trait reside in the
//! analysis layer.
//!
//! ## Architecture
//!
//! - **Domain Layer** (this file): Defines the `Filter` trait (interface/contract)
//! - **Analysis Layer**: Contains filter implementations (`FrequencyFilter`, etc.)
//!
//! This separation follows the **Dependency Inversion Principle**:
//! - High-level modules depend on abstractions (this trait)
//! - Low-level implementations satisfy the abstraction
//!
//! ## Migration Notice
//!
//! **⚠️ IMPORTANT**: The `FrequencyFilter` implementation has been moved to the
//! analysis layer as of Sprint 188 Phase 3 (Domain Layer Cleanup).
//!
//! ### Old Import (No Longer Valid)
//! ```rust,ignore
//! use crate::domain::signal::filter::FrequencyFilter;
//! ```
//!
//! ### New Import (Correct Location)
//! ```rust,ignore
//! use crate::analysis::signal_processing::filtering::FrequencyFilter;
//! ```
//!
//! ## Usage
//!
//! The `Filter` trait can be used to write filter-agnostic code:
//!
//! ```rust,no_run
//! use kwavers::domain::signal::Filter;
//! use kwavers::analysis::signal_processing::filtering::FrequencyFilter;
//! use kwavers::core::error::KwaversResult;
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

use crate::core::error::KwaversResult;
use std::fmt::Debug;

/// Trait for signal filtering operations
///
/// This trait defines the interface for applying filters to time-domain signals.
/// Implementations should transform the input signal according to their specific
/// filtering characteristics (bandpass, lowpass, highpass, etc.).
///
/// # Design Principles
///
/// - **Stateless**: Filters should be reusable and thread-safe
/// - **Pure Functions**: Same input → same output (no hidden state)
/// - **Error Handling**: Return `KwaversResult` for graceful error handling
///
/// # Thread Safety
///
/// Implementations should be `Send + Sync` to enable parallel processing.
///
/// # Examples
///
/// ```rust,no_run
/// use kwavers::domain::signal::Filter;
/// use kwavers::core::error::KwaversResult;
///
/// #[derive(Debug)]
/// struct IdentityFilter;
///
/// impl Filter for IdentityFilter {
///     fn apply(&self, signal: &[f64], _dt: f64) -> KwaversResult<Vec<f64>> {
///         Ok(signal.to_vec())
///     }
/// }
/// ```
pub trait Filter: Debug + Send + Sync {
    /// Apply the filter to a time-domain signal
    ///
    /// # Arguments
    ///
    /// * `signal` - Input time-domain signal samples
    /// * `dt` - Time step (sampling interval in seconds)
    ///
    /// # Returns
    ///
    /// Filtered signal with same or different length depending on implementation
    ///
    /// # Errors
    ///
    /// Returns `KwaversError` if filtering fails due to:
    /// - Invalid input parameters
    /// - Numerical issues
    /// - Memory allocation failures
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
