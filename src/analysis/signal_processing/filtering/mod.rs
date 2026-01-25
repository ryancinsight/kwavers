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
//! **DEPRECATED**: This module is deprecated. `FrequencyFilter` has been moved back
//! to `domain::signal::filter` to fix layer violations (solver layer cannot depend
//! on analysis layer).
//!
//! ### Old Location (Deprecated)
//! ```rust,ignore
//! use crate::analysis::signal_processing::filtering::FrequencyFilter;
//! ```
//!
//! ### New Location (Use This)
//! ```rust,ignore
//! use crate::domain::signal::FrequencyFilter;
//! ```

// Re-export from domain for backward compatibility during migration
pub use crate::domain::signal::{Filter, FrequencyFilter};
