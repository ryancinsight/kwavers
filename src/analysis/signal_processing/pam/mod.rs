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
//! ## Implemented Algorithms
//!
//! - ✅ **Delay-and-Sum PAM**: Time-domain beamforming for cavitation localization
//!
//! ## Future Implementations
//!
//! - [ ] Time-reversal PAM for cavitation bubble localization (Montaldo et al. 2009)
//! - [ ] Robust Capon beamforming for coherent cavitation noise suppression
//! - [ ] MUSIC algorithm for super-resolution bubble detection
//! - [ ] Subharmonic and ultraharmonic filtering for contrast agent differentiation
//! MISSING: Real-time PAM with GPU acceleration for clinical feedback
//! MISSING: PAM calibration using known cavitation sources (hydrophone validation)
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

// Implemented PAM algorithms
pub mod delay_and_sum;

// Public API exports
pub use delay_and_sum::{CavitationEvent, DelayAndSumConfig, DelayAndSumPAM};

// Future implementations
// pub mod robust_capon;
// pub mod detection;
// pub mod spectral;

use crate::analysis::signal_processing::beamforming::domain_processor::BeamformingProcessor;

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::sensor::beamforming::BeamformingCoreConfig;
use crate::domain::sensor::passive_acoustic_mapping::geometry::ArrayGeometry;
use ndarray::{Array3, Axis};
use rustfft::{num_complex::Complex, FftPlanner};

pub use crate::domain::sensor::passive_acoustic_mapping::geometry::{
    ArrayElement as PamArrayElement, ArrayGeometry as PamArrayGeometry,
    DirectivityPattern as PamDirectivityPattern,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PamBeamformingMethod {
    DelayAndSum,
    CaponDiagonalLoading { diagonal_loading: f64 },
    Music { num_sources: usize },
    EigenspaceMinVariance { signal_subspace_dimension: usize },
    TimeExposureAcoustics,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ApodizationType {
    None,
    Hamming,
    Hanning,
    Blackman,
    Kaiser { beta: f64 },
}

#[derive(Debug, Clone)]
pub struct PamBeamformingConfig {
    pub core: BeamformingCoreConfig,
    pub method: PamBeamformingMethod,
    pub frequency_range: (f64, f64),
    pub spatial_resolution: f64,
    pub apodization: ApodizationType,
    pub focal_point: [f64; 3],
}

impl PamBeamformingConfig {
    pub fn validate(&self) -> KwaversResult<()> {
        let (f_min, f_max) = self.frequency_range;

        if !(f_min.is_finite() && f_max.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: frequency_range must be finite".to_string(),
            ));
        }
        if f_min < 0.0 || f_max < 0.0 {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: frequency_range must be non-negative".to_string(),
            ));
        }
        if f_min > f_max {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: require f_min <= f_max".to_string(),
            ));
        }

        if !self.spatial_resolution.is_finite() || self.spatial_resolution <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: spatial_resolution must be finite and > 0".to_string(),
            ));
        }

        if self.focal_point.iter().any(|v| !v.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: focal_point must be finite".to_string(),
            ));
        }

        match self.method {
            PamBeamformingMethod::CaponDiagonalLoading { diagonal_loading } => {
                if !diagonal_loading.is_finite() || diagonal_loading < 0.0 {
                    return Err(KwaversError::InvalidInput(
                        "PAM beamforming config: diagonal_loading must be finite and >= 0"
                            .to_string(),
                    ));
                }
            }
            PamBeamformingMethod::Music { num_sources } => {
                if num_sources == 0 {
                    return Err(KwaversError::InvalidInput(
                        "PAM beamforming config: MUSIC requires num_sources >= 1".to_string(),
                    ));
                }
            }
            PamBeamformingMethod::EigenspaceMinVariance {
                signal_subspace_dimension,
            } => {
                if signal_subspace_dimension == 0 {
                    return Err(KwaversError::InvalidInput(
                        "PAM beamforming config: ESMV requires signal_subspace_dimension >= 1"
                            .to_string(),
                    ));
                }
            }
            PamBeamformingMethod::DelayAndSum | PamBeamformingMethod::TimeExposureAcoustics => {}
        }

        Ok(())
    }

    #[must_use]
    pub fn reference_frequency_midpoint(&self) -> f64 {
        let (f_min, f_max) = self.frequency_range;
        0.5 * (f_min + f_max)
    }
}

impl Default for PamBeamformingConfig {
    fn default() -> Self {
        Self {
            core: BeamformingCoreConfig::default(),
            method: PamBeamformingMethod::DelayAndSum,
            frequency_range: (20e3, 10e6),
            spatial_resolution: 1e-3,
            apodization: ApodizationType::Hamming,
            focal_point: [0.0, 0.0, 0.0],
        }
    }
}

#[derive(Debug, Clone)]
pub struct PAMConfig {
    pub beamforming: PamBeamformingConfig,
    pub frequency_bands: Vec<(f64, f64)>,
    pub integration_time: f64,
    pub threshold: f64,
    pub enable_harmonic_analysis: bool,
    pub enable_broadband_analysis: bool,
}

pub struct PAMProcessor {
    config: PAMConfig,
    fft_planner: FftPlanner<f64>,
}

impl std::fmt::Debug for PAMProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PAMProcessor")
            .field("config", &self.config)
            .field("fft_planner", &"<FftPlanner>")
            .finish()
    }
}

impl PAMProcessor {
    pub fn new(config: PAMConfig) -> KwaversResult<Self> {
        Ok(Self {
            config,
            fft_planner: FftPlanner::new(),
        })
    }

    pub fn process(&mut self, beamformed_data: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let shape = beamformed_data.shape();
        let (nx, ny, nt) = (shape[0], shape[1], shape[2]);

        let mut cavitation_map = Array3::zeros((nx, ny, self.config.frequency_bands.len()));

        for ix in 0..nx {
            for iy in 0..ny {
                let time_series: Vec<f64> =
                    (0..nt).map(|it| beamformed_data[[ix, iy, it]]).collect();

                let spectrum = self.compute_spectrum(&time_series)?;

                for (band_idx, &(f_min, f_max)) in self.config.frequency_bands.iter().enumerate() {
                    let power = self.integrate_band_power(&spectrum, f_min, f_max);

                    if power > self.config.threshold {
                        cavitation_map[[ix, iy, band_idx]] = power;
                    }
                }

                if self.config.enable_harmonic_analysis {
                    self.analyze_harmonics(&spectrum, ix, iy, &mut cavitation_map)?;
                }
            }
        }

        Ok(cavitation_map)
    }

    fn compute_spectrum(&mut self, time_series: &[f64]) -> KwaversResult<Vec<f64>> {
        let n = time_series.len();
        let mut complex_data: Vec<Complex<f64>> =
            time_series.iter().map(|&x| Complex::new(x, 0.0)).collect();

        let fft = self.fft_planner.plan_fft_forward(n);
        fft.process(&mut complex_data);

        let spectrum: Vec<f64> = complex_data
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        Ok(spectrum)
    }

    fn integrate_band_power(&self, spectrum: &[f64], f_min: f64, f_max: f64) -> f64 {
        let n = spectrum.len();
        if n == 0 {
            return 0.0;
        }

        let f_s = self.config.beamforming.core.sampling_frequency;
        if !f_s.is_finite() || f_s <= 0.0 {
            return 0.0;
        }

        if !f_min.is_finite() || !f_max.is_finite() || f_min < 0.0 || f_max < 0.0 || f_min > f_max {
            return 0.0;
        }

        let idx_min = ((f_min * n as f64) / f_s).floor().max(0.0) as usize;
        let idx_max = ((f_max * n as f64) / f_s).ceil().max(0.0) as usize;

        let lo = idx_min.min(n - 1);
        let hi = idx_max.min(n);
        if lo >= hi {
            return 0.0;
        }

        spectrum[lo..hi].iter().sum()
    }

    fn analyze_harmonics(
        &self,
        spectrum: &[f64],
        ix: usize,
        iy: usize,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let fundamental_idx = self.find_fundamental(spectrum);

        for harmonic in 2..5 {
            let harmonic_idx = fundamental_idx * harmonic;
            if harmonic_idx < spectrum.len() {
                let harmonic_power = spectrum[harmonic_idx];
                if harmonic_power > self.config.threshold * 0.5 {
                    if harmonic - 2 < output.shape()[2] {
                        output[[ix, iy, harmonic - 2]] += harmonic_power;
                    }
                }
            }
        }

        Ok(())
    }

    fn find_fundamental(&self, spectrum: &[f64]) -> usize {
        spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map_or(0, |(idx, _)| idx)
    }

    #[must_use]
    pub fn config(&self) -> &PAMConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: PAMConfig) -> KwaversResult<()> {
        self.config = config;
        Ok(())
    }
}

impl Default for PAMConfig {
    fn default() -> Self {
        Self {
            beamforming: PamBeamformingConfig::default(),
            frequency_bands: vec![(20e3, 100e3), (100e3, 500e3), (500e3, 2e6), (2e6, 10e6)],
            integration_time: 0.1,
            threshold: 1e-6,
            enable_harmonic_analysis: true,
            enable_broadband_analysis: true,
        }
    }
}

#[derive(Debug)]
pub struct PassiveAcousticMapper {
    processor: PAMProcessor,
    beamformer: BeamformingProcessor,
}

impl PassiveAcousticMapper {
    pub fn new(config: PAMConfig, geometry: ArrayGeometry) -> KwaversResult<Self> {
        config.beamforming.validate()?;

        let element_positions = geometry.element_positions();
        let core_cfg = config.beamforming.clone().into();
        let beamformer = BeamformingProcessor::new(core_cfg, element_positions);
        let processor = PAMProcessor::new(config)?;

        Ok(Self {
            processor,
            beamformer,
        })
    }

    pub fn process(
        &mut self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
    ) -> KwaversResult<Array3<f64>> {
        let config = self.processor.config();

        config.beamforming.validate()?;

        let delays = self
            .beamformer
            .compute_delays(config.beamforming.focal_point);

        let beamformed = match config.beamforming.method {
            PamBeamformingMethod::DelayAndSum => {
                let weights = vec![1.0; self.beamformer.num_sensors()];
                self.beamformer
                    .delay_and_sum_with(sensor_data, sample_rate, &delays, &weights)?
            }
            PamBeamformingMethod::CaponDiagonalLoading { diagonal_loading } => self
                .beamformer
                .mvdr_unsteered_weights_time_series(sensor_data, diagonal_loading)?,
            PamBeamformingMethod::Music { .. } => {
                return Err(KwaversError::InvalidInput(
                    "PAM beamforming: MUSIC is not yet wired to the shared subspace implementation. Use DelayAndSum or CaponDiagonalLoading for PAM mapping."
                        .to_string(),
                ));
            }
            PamBeamformingMethod::EigenspaceMinVariance { .. } => {
                return Err(KwaversError::InvalidInput(
                    "PAM beamforming: EigenspaceMinVariance is not yet wired to the shared subspace implementation. Use DelayAndSum or CaponDiagonalLoading for PAM mapping."
                        .to_string(),
                ));
            }
            PamBeamformingMethod::TimeExposureAcoustics => {
                let weights = vec![1.0; self.beamformer.num_sensors()];
                let das = self.beamformer.delay_and_sum_with(
                    sensor_data,
                    sample_rate,
                    &delays,
                    &weights,
                )?;

                let mut squared = das.clone();
                squared.mapv_inplace(|x| x * x);

                let integrated = squared.sum_axis(Axis(2));
                let (nx, ny) = (integrated.shape()[0], integrated.shape()[1]);

                let mut tea = Array3::<f64>::zeros((nx, ny, 1));
                for ix in 0..nx {
                    for iy in 0..ny {
                        tea[[ix, iy, 0]] = integrated[[ix, iy]];
                    }
                }

                tea
            }
        };

        self.processor.process(&beamformed)
    }

    #[must_use]
    pub fn config(&self) -> &PAMConfig {
        self.processor.config()
    }

    pub fn set_config(&mut self, config: PAMConfig) -> KwaversResult<()> {
        config.beamforming.validate()?;

        let element_positions: Vec<[f64; 3]> = self.beamformer.sensor_positions().to_vec();
        let core_cfg = config.beamforming.clone().into();
        self.beamformer = BeamformingProcessor::new(core_cfg, element_positions);

        self.processor.set_config(config)?;
        Ok(())
    }
}

impl From<PamBeamformingConfig> for BeamformingCoreConfig {
    fn from(pam: PamBeamformingConfig) -> Self {
        let (f_min, f_max) = pam.frequency_range;
        let reference_frequency = 0.5 * (f_min + f_max);

        let mut core = pam.core;
        core.reference_frequency = reference_frequency;

        if let PamBeamformingMethod::CaponDiagonalLoading { diagonal_loading } = pam.method {
            core.diagonal_loading = diagonal_loading;
        }

        core
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::{SAMPLING_FREQUENCY_DEFAULT, SOUND_SPEED_TISSUE};

    #[test]
    fn pam_policy_to_core_capon_loading_and_midpoint_frequency() {
        let pam = PamBeamformingConfig {
            core: BeamformingCoreConfig::default(),
            method: PamBeamformingMethod::CaponDiagonalLoading {
                diagonal_loading: 0.05,
            },
            frequency_range: (1.0e6, 3.0e6),
            spatial_resolution: 1e-3,
            apodization: ApodizationType::Hamming,
            focal_point: [0.0, 0.0, 0.0],
        };

        let core: BeamformingCoreConfig = pam.into();
        assert!((core.reference_frequency - 2.0e6).abs() < 1.0);
        assert!((core.diagonal_loading - 0.05).abs() < 1e-12);

        assert_eq!(core.sound_speed, SOUND_SPEED_TISSUE);
        assert_eq!(core.sampling_frequency, SAMPLING_FREQUENCY_DEFAULT);
        assert_eq!(core.num_snapshots, 100);
        assert_eq!(core.spatial_smoothing, None);
    }

    #[test]
    fn pam_policy_to_core_non_capon_preserves_core_loading_and_sets_reference_frequency() {
        let embedded_core = BeamformingCoreConfig {
            diagonal_loading: 0.123,
            ..Default::default()
        };

        let pam = PamBeamformingConfig {
            core: embedded_core.clone(),
            method: PamBeamformingMethod::DelayAndSum,
            frequency_range: (2.0e6, 2.0e6),
            spatial_resolution: 1e-3,
            apodization: ApodizationType::None,
            focal_point: [0.0, 0.0, 0.0],
        };

        let core: BeamformingCoreConfig = pam.into();
        assert!((core.reference_frequency - 2.0e6).abs() < 1.0);
        assert!((core.diagonal_loading - embedded_core.diagonal_loading).abs() < 1e-12);
    }
}
