//! # Signal Processing Module
//!
//! This module provides signal processing algorithms for acoustic and ultrasound data.
//! It serves as the proper home for all post-acquisition data processing, including
//! beamforming, localization, and passive acoustic mapping.
//!
//! ## Architecture
//!
//! Signal processing sits in the **analysis layer** (Layer 7), which means it can:
//! - ✅ Import from all lower layers (core, math, domain, physics, solver, simulation, clinical)
//! - ✅ Process data from sensors and simulations
//! - ✅ Implement analysis algorithms without domain constraints
//!
//! ## Modules
//!
//! - **`beamforming/`**: Array signal processing for image formation
//! - **`localization/`**: Source localization algorithms (trilateration, beamforming search)
//! - **`pam/`**: Passive acoustic mapping for cavitation monitoring
//!
//! ## Migration Strategy
//!
//! ### Historical Context
//!
//! Previously, signal processing algorithms resided in `domain::sensor::beamforming`.
//! This violated architectural layering principles:
//! - Domain layer should contain only primitives (sensor geometry, sampling)
//! - Signal processing is analysis, not domain primitives
//! - Led to circular dependencies and unclear boundaries
//!
//! ### Deprecation Plan
//!
//! **Phase 1 (Week 2):** Structure Creation ✅ **COMPLETE**
//! - Create `analysis::signal_processing` module
//! - Define trait interfaces
//! - Document deprecation strategy
//!
//! **Phase 2 (Week 3-4):** Gradual Migration 🟡 **IN PROGRESS**
//! - [x] Migrate time-domain DAS (Delay-and-Sum) ✅
//! - [x] Migrate delay reference policy ✅
//! - [ ] Add deprecation warnings to `domain::sensor::beamforming`
//! - [ ] Create backward compatibility shims
//! - [ ] Migrate adaptive beamforming (Capon, MUSIC)
//! - [ ] Migrate localization algorithms
//! - [ ] Migrate PAM algorithms
//!
//! **Phase 3 (Week 5+):** Cleanup
//! - Migrate all callers to new location
//! - Remove deprecated `domain::sensor::beamforming`
//! - Clean domain layer to pure primitives
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::beamforming::{
//!     delay_and_sum, DelayReference, DEFAULT_DELAY_REFERENCE
//! };
//! use kwavers::domain::sensor::GridSensorSet;
//! use ndarray::Array3;
//!
//! // Sensor geometry (domain primitive)
//! let sensors = GridSensorSet::new(positions, sample_rate)?;
//!
//! // Compute delays from sensor geometry and physics
//! let focal_point = [0.0, 0.0, 0.02]; // 20mm depth
//! let sound_speed = 1540.0; // m/s
//! let delays = sensors.compute_tof_delays(&focal_point, sound_speed)?;
//!
//! // Process sensor data (analysis algorithm)
//! let rf_data: Array3<f64> = sensors.get_recorded_data();
//! let weights = vec![1.0; sensors.len()]; // Equal weights
//! let beamformed = delay_and_sum(
//!     &rf_data,
//!     sample_rate,
//!     &delays,
//!     &weights,
//!     DEFAULT_DELAY_REFERENCE,
//! )?;
//! ```
//!
//! ## Design Principles
//!
//! 1. **Separation of Concerns**: Sensor geometry (domain) vs processing (analysis)
//! 2. **Trait-Based**: All algorithms implement common traits
//! 3. **Literature-Validated**: Each algorithm cites original papers
//! 4. **Performance**: GPU acceleration available via feature flags
//! 5. **Type Safety**: Strong typing prevents misuse
//!
//! ## Algorithms Provided
//!
//! ### Beamforming
//! - **Delay-and-Sum (DAS)**: Standard time-domain beamforming ✅ **IMPLEMENTED**
//! - **Minimum Variance (Capon)**: Adaptive beamforming (planned)
//! - **MUSIC**: Multiple signal classification (planned)
//! - **Neural Beamforming**: ML-based beamforming (experimental, planned)
//!
//! ### Localization
//! - **Trilateration**: Time-of-arrival based localization
//! - **Beamforming Search**: Grid-based source search
//! - **Multilateration**: Multi-sensor localization
//!
//! ### Passive Acoustic Mapping
//! - **PAM**: Real-time cavitation monitoring
//! - **Array Processing**: Spatial-temporal filtering
//!
//! ## Feature Flags
//!
//! - `gpu`: GPU-accelerated beamforming
//! - `experimental_neural`: Neural network beamforming (requires `pinn`)
//!
//! ## References
//!
//! - Van Trees, H. L. (2002). *Optimum Array Processing*. Wiley.
//! - Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis."
//!   *Proceedings of the IEEE*, 57(8), 1408-1418.
//! - Schmidt, R. O. (1986). "Multiple emitter location and signal parameter estimation."
//!   *IEEE Transactions on Antennas and Propagation*, 34(3), 276-280.
//!
//! ## Layer Dependencies
//!
//! ```text
//! analysis::signal_processing  (Layer 7)
//!   ↓ (can import from)
//! clinical         (Layer 6)
//! simulation       (Layer 5)
//! solver           (Layer 4)
//! physics          (Layer 3)
//! domain           (Layer 2)
//! math             (Layer 1)
//! core             (Layer 0)
//! ```
//!
//! ## Migration Notes for Developers
//!
//! If you're migrating from `domain::sensor::beamforming`:
//!
//! ### Old (Deprecated)
//! ```rust,ignore
//! use crate::analysis::signal_processing::beamforming::time_domain::delay_and_sum;
//! ```
//!
//! ### New (Correct)
//! ```rust,ignore
//! use crate::analysis::signal_processing::beamforming::time_domain::delay_and_sum;
//! // Or use the convenience re-export:
//! use crate::analysis::signal_processing::beamforming::delay_and_sum;
//! ```
//!
//! ### Compatibility Period
//!
//! During migration (Weeks 2-4), both locations will exist:
//! - `domain::sensor::beamforming` - DEPRECATED, will be removed
//! - `analysis::signal_processing::beamforming` - NEW, use this
//!
//! Update your imports to the new location to avoid deprecation warnings.

pub mod beamforming;
pub mod filtering;
pub mod localization;
pub mod pam;

// Clutter filtering for Doppler and functional ultrasound
// Based on: Demené et al. (2015), Nouhoum et al. (2021)
pub mod clutter_filter;

// Future modules (planned)
// pub mod spectral;       // Spectral analysis (FFT, STFT, etc.)
// pub mod deconvolution;  // Point spread function deconvolution
// pub mod reconstruction; // Image reconstruction algorithms

// Re-export main types from implemented modules
pub use beamforming::{
    alignment_shifts_s, delay_and_sum, relative_delays_s, DelayReference, DEFAULT_DELAY_REFERENCE,
};
// Re-export FrequencyFilter from domain (moved back to fix layer violations)
pub use crate::domain::signal::FrequencyFilter;

// Future re-exports (will be populated as modules are implemented)
// pub use beamforming::{Beamformer, BeamformingConfig, MinimumVariance, MUSIC};
// pub use localization::{Localizer, LocalizationConfig, Trilateration};
// pub use pam::{PassiveAcousticMapper, PAMConfig};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        assert_eq!(DEFAULT_DELAY_REFERENCE, DelayReference::SensorIndex(0));
    }

    #[test]
    fn test_beamforming_exports() {
        // Verify key beamforming types are accessible
        let ref_policy = DelayReference::recommended_default();
        assert_eq!(ref_policy, DelayReference::SensorIndex(0));
    }
}
