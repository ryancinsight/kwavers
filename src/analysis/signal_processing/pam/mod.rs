//! # Passive Acoustic Mapping (PAM) Module
//!
//! This module provides passive acoustic mapping algorithms for real-time monitoring
//! and localization of acoustic sources, particularly cavitation events during
//! therapeutic ultrasound procedures.
//!
//! ## Overview
//!
//! Passive Acoustic Mapping (PAM) is a technique for detecting and localizing
//! acoustic emissions from cavitation bubbles, shock waves, or other transient
//! sources. Unlike active ultrasound imaging, PAM listens passively to acoustic
//! emissions.
//!
//! ## Applications
//!
//! - **HIFU Monitoring**: Real-time cavitation detection during focused ultrasound
//! - **Lithotripsy**: Shock wave localization and stone fragmentation monitoring
//! - **Drug Delivery**: Microbubble activity monitoring
//! - **Safety**: Detection of unintended cavitation events
//! - **Research**: Cavitation dynamics studies
//!
//! ## Algorithm Categories
//!
//! ### Time-Domain Methods
//! - **Delay-and-Sum PAM**: Direct time-domain beamforming
//! - **Cross-Correlation**: Pairwise delay estimation
//!
//! ### Frequency-Domain Methods
//! - **Robust Capon PAM**: Adaptive frequency-domain beamforming
//! - **Spectral Analysis**: Frequency content analysis
//!
//! ### Advanced Methods
//! - **Compressive PAM**: Sparse reconstruction
//! - **Neural PAM**: Machine learning-based source detection
//!
//! ## Migration from `domain::sensor::passive_acoustic_mapping`
//!
//! This module is the new home for PAM algorithms, previously located in
//! `domain::sensor::passive_acoustic_mapping`. The migration corrects architectural
//! layering violations.
//!
//! ### Migration Timeline
//!
//! - **Week 2 (Current)**: Module structure created, documentation in place
//! - **Week 3-4**: Algorithm migration with backward compatibility
//! - **Week 5+**: Remove deprecated `domain::sensor::passive_acoustic_mapping`
//!
//! ### What to Migrate Here
//!
//! ✅ **Should be in `analysis::signal_processing::pam`:**
//! - PAM processors and algorithms
//! - Cavitation detection logic
//! - Source localization algorithms
//! - Spectral analysis for cavitation signatures
//! - Real-time processing pipelines
//!
//! ❌ **Should stay in `domain::sensor`:**
//! - Sensor array geometry
//! - Passive receiver configurations
//! - Data acquisition parameters
//!
//! ## Architecture
//!
//! ```text
//! PAM Processing Flow:
//!
//! 1. Data Acquisition (domain::sensor)
//!    ├── Passive listening array
//!    ├── High sample rate (MHz)
//!    └── Continuous recording
//!
//! 2. Signal Detection (THIS MODULE)
//!    ├── Energy threshold detection
//!    ├── Time-frequency analysis
//!    └── Event segmentation
//!
//! 3. Source Localization (THIS MODULE)
//!    ├── Time-of-arrival estimation
//!    ├── Beamforming
//!    └── 3D position reconstruction
//!
//! 4. Characterization (THIS MODULE)
//!    ├── Frequency content
//!    ├── Event duration
//!    └── Source intensity
//!
//! 5. Output
//!    ├── Cavitation map
//!    ├── Event locations
//!    └── Temporal statistics
//! ```
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::pam::{PassiveAcousticMapper, PAMConfig};
//! use kwavers::domain::sensor::GridSensorSet;
//! use ndarray::Array2;
//!
//! // 1. Define passive sensor array (domain layer)
//! let sensor_positions = vec![
//!     [0.0, 0.0, 0.0],
//!     [0.005, 0.0, 0.0],
//!     // ... more sensors
//! ];
//! let sensors = GridSensorSet::new(sensor_positions, 5e6)?; // 5 MHz sampling
//!
//! // 2. Configure PAM processor
//! let pam_config = PAMConfig {
//!     sound_speed: 1540.0,          // m/s
//!     detection_threshold: 3.0,     // SNR threshold
//!     frequency_range: (100e3, 1e6), // 100 kHz - 1 MHz
//!     window_size: 1024,            // samples
//!     overlap: 0.5,                 // 50% overlap
//! };
//!
//! // 3. Create PAM processor
//! let mut pam = PassiveAcousticMapper::new(&sensors, pam_config)?;
//!
//! // 4. Define monitoring region
//! let monitoring_grid = pam.create_monitoring_grid(
//!     x_range: (-0.01, 0.01),  // ±10 mm
//!     y_range: (-0.01, 0.01),
//!     z_range: (0.02, 0.05),   // 20-50 mm depth
//!     resolution: 0.001,       // 1 mm voxels
//! )?;
//!
//! // 5. Process passive acoustic data
//! let passive_data: Array2<f64> = sensors.get_recorded_data();
//! let cavitation_map = pam.process(&passive_data, &monitoring_grid)?;
//!
//! // 6. Extract detected events
//! let events = pam.detect_events(&passive_data)?;
//! for event in events {
//!     println!("Cavitation at {:?}, intensity: {}", event.position, event.intensity);
//! }
//! ```
//!
//! ## Mathematical Foundation
//!
//! ### Time-of-Arrival (TOA) Estimation
//!
//! For a source at position **r_s**, the arrival time at sensor i is:
//!
//! ```text
//! tᵢ = t₀ + |r_s - rᵢ| / c
//! ```
//!
//! where:
//! - `t₀` = source emission time
//! - `rᵢ` = position of sensor i
//! - `c` = sound speed
//!
//! ### Delay-and-Sum PAM
//!
//! The PAM output at candidate position **r** is:
//!
//! ```text
//! P(r) = ∑ᵢ₌₁ᴺ wᵢ · |xᵢ(t - τᵢ(r))|²
//! ```
//!
//! where the delays are computed assuming **r** is the source location.
//!
//! ### Robust Capon PAM
//!
//! Adaptive weights based on covariance matrix:
//!
//! ```text
//! P(r) = 1 / (aᴴ(r) R⁻¹ a(r))
//! ```
//!
//! where:
//! - `a(r)` = steering vector for location r
//! - `R` = covariance matrix of sensor signals
//!
//! ## Signal Characteristics
//!
//! ### Cavitation Signatures
//!
//! - **Inertial Cavitation**: Broadband emissions (100 kHz - 10 MHz)
//! - **Stable Cavitation**: Harmonic and subharmonic emissions
//! - **Shock Waves**: Sharp transients with wide bandwidth
//!
//! ### Detection Features
//!
//! - Energy threshold crossing
//! - Spectral content analysis
//! - Time-frequency characteristics
//! - Multi-sensor coincidence
//!
//! ## Performance Considerations
//!
//! - **Real-Time Processing**: GPU acceleration for live monitoring
//! - **High Sample Rates**: Efficient handling of MHz sampling
//! - **Memory Management**: Sliding window processing for continuous data
//! - **Parallel Processing**: Multi-threaded event detection
//!
//! ## References
//!
//! - Gyöngy, M., & Coussios, C. C. (2010). "Passive spatial mapping of inertial
//!   cavitation during HIFU exposure." *IEEE Trans. Biomed. Eng.*, 57(1), 48-56.
//!   DOI: 10.1109/TBME.2009.2026907
//!
//! - Arvanitis, C. D., et al. (2012). "Passive acoustic mapping with the angular
//!   spectrum method." *IEEE Trans. Med. Imaging*, 31(11), 2086-2091.
//!   DOI: 10.1109/TMI.2012.2208761
//!
//! - Haworth, K. J., et al. (2012). "Passive imaging with pulsed ultrasound
//!   insonations." *J. Acoust. Soc. Am.*, 132(1), 544-553.
//!   DOI: 10.1121/1.4728230
//!
//! - Crake, C., et al. (2014). "Passive acoustic mapping of magnetic microbubbles
//!   for cavitation enhancement and localization." *Phys. Med. Biol.*, 60(3), 785.
//!   DOI: 10.1088/0031-9155/60/3/785
//!
//! ## Clinical Applications
//!
//! ### HIFU Monitoring
//! - Real-time feedback during tumor ablation
//! - Dose control based on cavitation activity
//! - Safety monitoring for skull heating
//!
//! ### Lithotripsy
//! - Stone fragmentation monitoring
//! - Shock wave focusing verification
//! - Treatment endpoint determination
//!
//! ### Blood-Brain Barrier Opening
//! - Microbubble activity monitoring
//! - Spatial targeting verification
//! - Safety threshold monitoring
//!
//! ## Future Implementations
//!
//! This module is currently a placeholder. Algorithms will be migrated from
//! `domain::sensor::passive_acoustic_mapping` in Phase 2 execution:
//!
//! - [ ] Define `PassiveAcousticMapper` trait
//! - [ ] Implement delay-and-sum PAM
//! - [ ] Implement robust Capon PAM
//! - [ ] Event detection algorithms
//! - [ ] Spectral analysis tools
//! - [ ] Real-time processing pipeline
//! - [ ] GPU acceleration
//! - [ ] Comprehensive testing with synthetic cavitation
//!
//! ## Status
//!
//! **Current:** 🟡 Module structure created, awaiting implementation
//! **Next:** Migrate PAM algorithms from domain layer
//! **Timeline:** Week 3-4 execution

// Placeholder for future trait definitions
// pub mod traits;

// Placeholder for algorithm implementations
// pub mod delay_and_sum;
// pub mod robust_capon;
// pub mod detection;

// Placeholder for utility functions
// pub mod utils;
// pub mod spectral;
