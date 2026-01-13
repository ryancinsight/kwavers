//! Signal Filtering Module
//!
//! This module provides frequency-domain and time-domain filtering algorithms
//! for acoustic and ultrasound signal processing.
//!
//! ## Architecture
//!
//! This module resides in the **analysis layer** because filtering is a signal
//! processing operation, not a domain primitive. The domain layer defines the
//! `Filter` trait (interface), while this module provides implementations.
//!
//! ## Available Filters
//!
//! - **`FrequencyFilter`**: FFT-based frequency-domain filtering
//!   - Bandpass filtering
//!   - Lowpass filtering
//!   - Highpass filtering
//!   - Time-domain windowing
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use kwavers::analysis::signal_processing::filtering::FrequencyFilter;
//! use kwavers::core::error::KwaversResult;
//!
//! fn filter_ultrasound_signal() -> KwaversResult<()> {
//!     let filter = FrequencyFilter::new();
//!
//!     // Ultrasound RF data sampled at 40 MHz
//!     let sample_rate = 40e6;
//!     let dt = 1.0 / sample_rate;
//!     let rf_data: Vec<f64> = vec![/* ... */];
//!
//!     // Apply 2-8 MHz bandpass (typical for 5 MHz transducer)
//!     let filtered = filter.bandpass(&rf_data, dt, 2e6, 8e6)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Migration Note
//!
//! This module was created in Sprint 188 Phase 3 by moving filter implementations
//! from `domain::signal::filter` to enforce proper architectural layering.
//!
//! ### Old Location (Deprecated)
//! ```rust,ignore
//! use crate::domain::signal::filter::FrequencyFilter;
//! ```
//!
//! ### New Location (Use This)
//! ```rust,ignore
//! use crate::analysis::signal_processing::filtering::FrequencyFilter;
//! ```

pub mod frequency_filter;

pub use frequency_filter::FrequencyFilter;

// Re-export the Filter trait from domain for convenience
pub use crate::domain::signal::Filter;
