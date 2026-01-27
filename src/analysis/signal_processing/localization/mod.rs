//! # Source Localization Module
//!
//! This module provides algorithms for localizing acoustic sources using sensor
//! array measurements. Localization determines the spatial coordinates of sound
//! sources from time-of-arrival, signal strength, or beamforming data.
//!
//! ## Overview
//!
//! Source localization is critical for:
//! - **Cavitation Monitoring**: Locate bubble collapse events
//! - **Passive Imaging**: Map acoustic emissions without active transmission
//! - **Quality Control**: Verify focal point accuracy in HIFU
//! - **Safety Monitoring**: Detect unexpected acoustic sources
//!
//! ## Algorithm Categories
//!
//! ### Time-of-Arrival (TOA) Methods
//! - **Trilateration**: Geometric solution using arrival time differences
//! - **Multilateration**: Overdetermined system for improved accuracy
//! - **Time Difference of Arrival (TDOA)**: Relative timing between sensors
//!
//! ### Signal-Based Methods
//! - **Beamforming Search**: Grid search using beamformer output
//! - **Maximum Likelihood**: Statistical estimation of source location
//! - **Matched Field Processing**: Compare measurements to forward model
//!
//! ### Hybrid Methods
//! - **Coarse-to-Fine**: Initial grid search refined by gradient descent
//! - **Multi-Resolution**: Hierarchical spatial discretization
//!
//! ## Migration from `domain::sensor::localization`
//!
//! This module is the new home for localization algorithms, previously in
//! `domain::sensor::localization`. The old location violated layering by
//! placing analysis algorithms in the domain layer.
//!
//! ### Migration Timeline
//!
//! - **Week 2 (Current)**: Module structure created
//! - **Week 3-4**: Algorithm migration with backward compatibility
//! - **Week 5+**: Remove deprecated location
//!
//! ### What Belongs Here
//!
//! ✅ **Should be in `analysis::signal_processing::localization`:**
//! - Trilateration algorithms
//! - Beamforming-based search
//! - Maximum likelihood estimation
//! - Source coordinate calculation
//! - Uncertainty quantification
//!
//! ❌ **Should stay in `domain::sensor`:**
//! - Sensor array geometry
//! - Sensor positions and orientations
//! - Time synchronization parameters
//!
//! ## Architecture
//!
//! ```text
//! Localization Flow:
//!
//! 1. Domain Layer (Sensor Array)
//!    ├── Sensor positions {r₁, r₂, ..., rₙ}
//!    ├── Time synchronization
//!    └── Coordinate system
//!
//! 2. Data Acquisition
//!    ├── Time-of-arrival measurements {t₁, t₂, ..., tₙ}
//!    ├── Signal amplitudes
//!    └── Cross-correlation data
//!
//! 3. Analysis Layer (THIS MODULE)
//!    ├── Time difference calculation
//!    ├── Geometric solution (trilateration)
//!    ├── Optimization (if nonlinear)
//!    └── Uncertainty estimation
//!
//! 4. Output
//!    ├── Source position {x, y, z}
//!    ├── Localization error bounds
//!    └── Confidence metrics
//! ```
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::localization::Trilateration;
//! use kwavers::domain::sensor::GridSensorSet;
//! use nalgebra::Point3;
//!
//! // 1. Define sensor geometry (domain layer)
//! let sensor_positions = vec![
//!     Point3::new(0.0, 0.0, 0.0),
//!     Point3::new(0.01, 0.0, 0.0),
//!     Point3::new(0.0, 0.01, 0.0),
//!     Point3::new(0.0, 0.0, 0.01),
//! ];
//!
//! // 2. Measure time-of-arrival at each sensor
//! let arrival_times = vec![0.0e-6, 6.5e-6, 6.5e-6, 6.5e-6]; // microseconds
//!
//! // 3. Create localizer (analysis layer)
//! let sound_speed = 1540.0; // m/s (soft tissue)
//! let localizer = Trilateration::new(sensor_positions, sound_speed);
//!
//! // 4. Compute source location
//! let result = localizer.localize(&arrival_times)?;
//!
//! println!("Source located at: {:?}", result.position);
//! println!("Uncertainty: ±{:.3} mm", result.uncertainty * 1000.0);
//! ```
//!
//! ## Mathematical Foundation
//!
//! ### Trilateration
//!
//! For a source at position **r** and sensors at **rᵢ**, the time-of-arrival is:
//!
//! ```text
//! tᵢ = t₀ + |r - rᵢ| / c
//! ```
//!
//! where:
//! - `t₀` = emission time (unknown)
//! - `c` = sound speed
//! - `|·|` = Euclidean distance
//!
//! Using time differences (TDOA) eliminates t₀:
//!
//! ```text
//! Δtᵢⱼ = tᵢ - tⱼ = (|r - rᵢ| - |r - rⱼ|) / c
//! ```
//!
//! This forms a system of nonlinear equations solved by:
//! - **Analytical**: Closed-form for 4+ sensors (overdetermined)
//! - **Iterative**: Newton-Raphson or Levenberg-Marquardt
//!
//! ### Beamforming Search
//!
//! Evaluate beamformer output on a spatial grid and find maximum:
//!
//! ```text
//! r̂ = argmax_r |y(r)|²
//! ```
//!
//! where `y(r)` is the beamformed output at location r.
//!
//! ## Trait Hierarchy
//!
//! ```text
//! Localizer (trait)
//!   ├── localize(measurements) -> LocalizationResult
//!   ├── set_sound_speed(c)
//!   └── estimate_uncertainty() -> f64
//!
//! TOALocalizer: Localizer
//!   ├── Trilateration (analytical solution)
//!   └── Multilateration (least-squares)
//!
//! BeamformingLocalizer: Localizer
//!   ├── GridSearch (exhaustive)
//!   └── CoarseToFine (hierarchical)
//! ```
//!
//! ## Performance Considerations
//!
//! - **Real-Time Constraints**: Localization must be fast for feedback control
//! - **GPU Acceleration**: Parallel grid search on GPU
//! - **Adaptive Resolution**: Coarse-to-fine for speed vs accuracy tradeoff
//! - **Caching**: Pre-compute delay tables for known geometries
//!
//! ## Error Analysis
//!
//! Sources of localization error:
//! - **Timing Jitter**: Sensor clock precision
//! - **Sound Speed Variation**: Inhomogeneous media
//! - **Geometric Dilution of Precision (GDOP)**: Sensor placement
//! - **Multipath**: Reflections and reverberations
//!
//! ## References
//!
//! - Friedlander, B. (1987). "A passive localization algorithm and its accuracy analysis."
//!   *IEEE Journal of Oceanic Engineering*, 12(1), 234-245.
//!   DOI: 10.1109/JOE.1987.1145216
//!
//! - Foy, W. H. (1976). "Position-location solutions by Taylor-series estimation."
//!   *IEEE Trans. on Aerospace and Electronic Systems*, AES-12(2), 187-194.
//!   DOI: 10.1109/TAES.1976.308294
//!
//! - Arulampalam, M. S., et al. (2002). "A tutorial on particle filters for online
//!   nonlinear/non-Gaussian Bayesian tracking." *IEEE Trans. Signal Processing*, 50(2).
//!   DOI: 10.1109/78.978374
//!
//! ## Future Implementations
//!
//! This module is currently a placeholder. Algorithms will be migrated from
//! TODO_AUDIT: P1 - Source Localization Algorithms - Implement advanced acoustic source localization for cavitation bubble tracking
//! DEPENDS ON: analysis/signal_processing/localization/trilateration.rs, analysis/signal_processing/localization/music.rs, analysis/signal_processing/localization/bayes.rs
//! MISSING: MUSIC (MUltiple SIgnal Classification) algorithm for super-resolution localization
//! MISSING: Bayesian filtering with particle filters for bubble trajectory tracking
//! MISSING: Time-difference-of-arrival (TDOA) with iterative least squares refinement
//! MISSING: Wavefront curvature analysis for range estimation
//! MISSING: Multi-path interference rejection using coherence analysis
//! MISSING: Real-time localization with Kalman filtering for continuous tracking
//! `domain::sensor::localization` in Phase 2 execution:
//!
//! - [ ] Define `Localizer` trait
//! - [ ] Implement `Trilateration`
//! - [ ] Implement `Multilateration`
//! - [ ] Implement `BeamformingSearch`
//! - [ ] Add uncertainty quantification
//! - [ ] GPU-accelerated grid search
//! - [ ] Comprehensive testing with synthetic data
//!
//! ## Status
//!
//! **Current:** 🟡 Module structure created, awaiting implementation
//! **Next:** Migrate trilateration algorithm from domain layer
//! **Timeline:** Week 3 execution

// Implemented localization algorithms
pub mod beamforming_search;
pub mod multilateration;
pub mod music;
pub mod trilateration;

// Public API exports
pub use multilateration::{Multilateration, MultilaterationConfig};
pub use music::{MusicConfig, MusicLocalizer, MusicResult};
pub use trilateration::{LocalizationResult, Trilateration, TrilaterationConfig};

// Future implementations
// pub mod beamforming_search;
// pub mod maximum_likelihood;
