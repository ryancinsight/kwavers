//! Doppler Velocity Estimation and Flow Imaging
//!
//! This module implements Doppler ultrasound techniques for blood flow velocity estimation
//! and vascular imaging. It provides both pulsed-wave (PW) Doppler and color flow imaging
//! capabilities essential for cardiovascular and vascular diagnostics.
//!
//! # Overview
//!
//! Doppler ultrasound exploits the Doppler effect to measure blood flow velocities:
//! - **Frequency shift** proportional to velocity component along beam axis
//! - **Spectral analysis** for pulsed-wave Doppler waveforms
//! - **2D velocity maps** for color flow imaging
//!
//! # Mathematical Foundation
//!
//! ## Doppler Equation
//!
//! For a scatterer (blood cell) moving with velocity **v** at angle θ to the beam:
//!
//! ```text
//! f_d = (2 * f₀ * v * cos(θ)) / c
//! ```
//!
//! Where:
//! - `f_d`: Doppler frequency shift (Hz)
//! - `f₀`: Transmitted frequency (Hz)
//! - `v`: Blood flow velocity (m/s)
//! - `θ`: Angle between flow direction and ultrasound beam
//! - `c`: Speed of sound in tissue (~1540 m/s)
//!
//! Solving for velocity:
//!
//! ```text
//! v = (f_d * c) / (2 * f₀ * cos(θ))
//! ```
//!
//! ## Autocorrelation Method
//!
//! For color flow imaging, velocity is estimated from the phase shift between successive pulses:
//!
//! ```text
//! φ = arctan[Im(R₁) / Re(R₁)]
//! v = (φ * c) / (4π * f₀ * T_prf * cos(θ))
//! ```
//!
//! Where:
//! - `R₁`: Lag-1 autocorrelation of received signals
//! - `T_prf`: Pulse repetition period
//!
//! # Clinical Applications
//!
//! ## Pulsed-Wave Doppler
//! - **Cardiac**: Valve flow assessment, chamber filling analysis
//! - **Vascular**: Stenosis detection, flow waveform characterization
//! - **Obstetrics**: Umbilical artery resistance indices
//!
//! ## Color Flow Imaging
//! - **2D velocity maps**: Visualize flow patterns in real-time
//! - **Turbulence detection**: Identify disturbed flow (stenosis, regurgitation)
//! - **Perfusion assessment**: Tissue blood supply evaluation
//!
//! # Literature References
//!
//! - Evans, D.H. & McDicken, W.N. (2000). "Doppler Ultrasound: Physics, Instrumentation and Signal Processing" (2nd ed.). Wiley.
//! - Hoskins, P.R. (2010). "Ultrasound techniques for measurement of blood flow and tissue motion". *Biorheology*, 47(3-4), 159-177.
//! - Jensen, J.A. (1996). "Estimation of Blood Velocities Using Ultrasound". Cambridge University Press.
//! - Kasai, C. et al. (1985). "Real-time two-dimensional blood flow imaging using an autocorrelation technique". *IEEE Trans. Sonics Ultrason.*, 32(3), 458-464.

pub mod autocorrelation;
pub mod color_flow;
pub mod pulsed_wave;
pub mod spectral;
pub mod types;
pub mod wall_filter;

pub use autocorrelation::{AutocorrelationConfig, AutocorrelationEstimator};
pub use color_flow::{ColorFlowConfig, ColorFlowImaging, VelocityMap};
pub use pulsed_wave::{PWDConfig, PulsedWaveDoppler, SpectralWaveform};
pub use spectral::{SpectralAnalysis, SpectralConfig};
pub use types::{DopplerResult, FlowDirection, VelocityEstimate};
pub use wall_filter::{FilterType, WallFilter, WallFilterConfig};

/// Default Doppler imaging parameters for clinical ultrasound
///
/// Based on typical values for cardiac and vascular imaging:
/// - Center frequency: 2.5 MHz (cardiac) to 7.5 MHz (vascular)
/// - PRF: 1-10 kHz (trade-off between velocity range and frame rate)
/// - Ensemble size: 8-16 pulses per estimate
#[derive(Debug, Clone, Copy)]
pub struct DopplerDefaults;

impl DopplerDefaults {
    /// Cardiac imaging parameters (lower frequency, deeper penetration)
    pub const CARDIAC_FREQUENCY: f64 = 2.5e6; // 2.5 MHz
    pub const CARDIAC_PRF: f64 = 4e3; // 4 kHz
    pub const CARDIAC_ENSEMBLE_SIZE: usize = 12;

    /// Vascular imaging parameters (higher frequency, better resolution)
    pub const VASCULAR_FREQUENCY: f64 = 7.5e6; // 7.5 MHz
    pub const VASCULAR_PRF: f64 = 5e3; // 5 kHz
    pub const VASCULAR_ENSEMBLE_SIZE: usize = 8;

    /// Obstetric imaging parameters (balanced for fetal imaging)
    pub const OBSTETRIC_FREQUENCY: f64 = 3.5e6; // 3.5 MHz
    pub const OBSTETRIC_PRF: f64 = 4e3; // 4 kHz
    pub const OBSTETRIC_ENSEMBLE_SIZE: usize = 10;

    /// Speed of sound in soft tissue (m/s)
    pub const SPEED_OF_SOUND: f64 = 1540.0;

    /// Maximum velocity before aliasing (m/s)
    /// For cardiac: ±1.5 m/s, vascular: ±0.5 m/s
    pub fn max_velocity_cardiac() -> f64 {
        (Self::CARDIAC_PRF * Self::SPEED_OF_SOUND) / (4.0 * Self::CARDIAC_FREQUENCY)
    }

    pub fn max_velocity_vascular() -> f64 {
        (Self::VASCULAR_PRF * Self::SPEED_OF_SOUND) / (4.0 * Self::VASCULAR_FREQUENCY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doppler_defaults() {
        // Verify cardiac Nyquist velocity is reasonable
        let v_max_cardiac = DopplerDefaults::max_velocity_cardiac();
        assert!(
            v_max_cardiac > 0.5 && v_max_cardiac < 2.0,
            "Cardiac max velocity should be 0.5-2.0 m/s, got {}",
            v_max_cardiac
        );

        // Verify vascular Nyquist velocity
        let v_max_vascular = DopplerDefaults::max_velocity_vascular();
        assert!(
            v_max_vascular > 0.2 && v_max_vascular < 1.0,
            "Vascular max velocity should be 0.2-1.0 m/s, got {}",
            v_max_vascular
        );
    }
}
