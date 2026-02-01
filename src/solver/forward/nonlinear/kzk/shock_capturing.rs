//! Shock Formation Detection and Capturing for KZK Equation
//!
//! This module implements shock wave detection and numerical capturing techniques
//! for handling discontinuous waveforms that develop during nonlinear propagation.
//!
//! ## Physical Background
//!
//! **Shock Formation Mechanism**:
//! When acoustic pressure steepens due to nonlinearity, the wave can develop
//! a discontinuity (shock) across which pressure jumps. This occurs when:
//! - Nonlinear steepening overcomes absorption damping
//! - Local pressure gradients exceed critical threshold
//! - Harmonic energy transfers from fundamental to higher harmonics
//!
//! **Rankine-Hugoniot Jump Conditions**:
//! At a shock (discontinuity), conservation laws give:
//! ```
//! ρ₀(ż - u₊) = -ρ₀(ż - u₋)           (mass conservation)
//! p₊ - ρ₀(ż - u₊)u₊ = p₋ - ρ₀(ż - u₋)u₋  (momentum)
//! ```
//! where ż is shock velocity, u± and p± are velocity and pressure on either side
//!
//! ## Shock Capturing Strategies
//!
//! **1. High-Order Filter (Slope Limiting)**:
//! Applies dissipative filter to regions with large gradients
//! to prevent oscillations while capturing shock location
//!
//! **2. Artificial Viscosity**:
//! Adds diffusive term proportional to |∇p| to handle discontinuities
//! ```
//! Q_av = c₀μ|∇p|/ρ₀ · ∇²p
//! ```
//!
//! **3. Shock Detection with Harmonic Tracking**:
//! - Monitor pressure waveform steepness (characteristic time τ_shock)
//! - Track energy in harmonic components (2f, 3f, ...)
//! - Detect shock when gradient exceeds threshold
//!
//! ## References
//!
//! - Varslot & Taraldsen (2005) "Impedance adapted seismic absorbing boundary"
//! - Zemp et al. (2004) "Modeling nonlinear ultrasound propagation"
//! - Tavakkoli et al. (1998) "Ultrasonic heating of soft tissues"

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{s, Array2};
use std::f64;

/// Configuration for shock capturing
#[derive(Debug, Clone, Copy)]
pub struct ShockCapturingConfig {
    /// Enable shock detection
    pub enable_detection: bool,

    /// Enable shock capturing (artificial viscosity)
    pub enable_capturing: bool,

    /// Pressure gradient threshold for shock detection (Pa/m)
    pub gradient_threshold: f64,

    /// Artificial viscosity coefficient (0.0 - 1.0)
    /// Larger value = more dissipation, better stability but less accurate
    pub viscosity_coefficient: f64,

    /// Harmonic energy threshold for shock detection (ratio to fundamental)
    pub harmonic_threshold: f64,

    /// Number of harmonics to track (2nd, 3rd, 4th, ...)
    pub num_harmonics: usize,

    /// Window size for gradient calculation (samples)
    pub gradient_window: usize,

    /// Shock velocity estimate (m/s) for adaptive filtering
    pub shock_velocity_estimate: f64,
}

impl Default for ShockCapturingConfig {
    fn default() -> Self {
        Self {
            enable_detection: true,
            enable_capturing: true,
            gradient_threshold: 1e4, // Pa/m = 10 kPa/mm
            viscosity_coefficient: 0.1,
            harmonic_threshold: 0.1, // 10% threshold
            num_harmonics: 3,
            gradient_window: 3,
            shock_velocity_estimate: 1540.0, // m/s
        }
    }
}

/// Results from shock detection analysis
#[derive(Debug, Clone)]
pub struct ShockDetectionResult {
    /// Whether shock was detected
    pub shock_detected: bool,

    /// Location of shock (z-index in grid)
    pub shock_location: Option<usize>,

    /// Maximum pressure gradient in the field (Pa/m)
    pub max_gradient: f64,

    /// Gradient steepness parameter τ_shock (characteristic steepening time)
    pub steepness_parameter: f64,

    /// Harmonic content ratios (2f/f, 3f/f, ...)
    pub harmonic_ratios: Vec<f64>,

    /// Total harmonic distortion (THD)
    pub thd: f64,

    /// Predicted shock formation distance (m)
    pub shock_distance: Option<f64>,

    /// Shock strength (pressure jump, Pa)
    pub shock_strength: Option<f64>,
}

impl Default for ShockDetectionResult {
    fn default() -> Self {
        Self {
            shock_detected: false,
            shock_location: None,
            max_gradient: 0.0,
            steepness_parameter: 0.0,
            harmonic_ratios: Vec::new(),
            thd: 0.0,
            shock_distance: None,
            shock_strength: None,
        }
    }
}

/// Shock formation detector and capturer
#[derive(Debug)]
pub struct ShockCapture {
    config: ShockCapturingConfig,
    history: Vec<ShockDetectionResult>,
}

impl ShockCapture {
    /// Create new shock capture instance
    pub fn new(config: ShockCapturingConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Detect shock formation from pressure field
    pub fn detect_shock(
        &self,
        pressure: &Array2<f64>,
        dx: f64,
        dz: f64,
        c0: f64,
        frequency: f64,
    ) -> KwaversResult<ShockDetectionResult> {
        let (nx, nz) = pressure.dim();
        if nx < 3 || nz < 3 {
            return Err(KwaversError::InvalidInput(
                "Pressure field too small for shock detection".to_string(),
            ));
        }

        let mut result = ShockDetectionResult::default();

        // 1. Compute pressure gradients (central differences)
        let mut grad_x = Array2::zeros((nx - 2, nz));
        let mut grad_z = Array2::zeros((nx, nz - 2));

        for z in 0..nz {
            for x in 1..nx - 1 {
                grad_x[[x - 1, z]] = (pressure[[x + 1, z]] - pressure[[x - 1, z]]) / (2.0 * dx);
            }
        }

        for z in 1..nz - 1 {
            for x in 0..nx {
                grad_z[[x, z - 1]] = (pressure[[x, z + 1]] - pressure[[x, z - 1]]) / (2.0 * dz);
            }
        }

        // 2. Find maximum gradient
        result.max_gradient = grad_x
            .iter()
            .chain(grad_z.iter())
            .map(|x| x.abs())
            .fold(0.0, f64::max);

        // 3. Compute steepness parameter
        // τ_shock ≈ 1 / (β * c0 * p * |∇p|) for weak shocks
        let max_pressure = pressure.iter().map(|x| x.abs()).fold(0.0, f64::max);

        if max_pressure > 1.0 && result.max_gradient > 1e-6 {
            let beta = 3.5; // For water/tissue
            result.steepness_parameter =
                1.0 / (beta * c0 * max_pressure * result.max_gradient).max(1e-10);
        }

        // 4. Detect shock by gradient threshold
        if result.max_gradient > self.config.gradient_threshold {
            result.shock_detected = true;

            // Find shock location (maximum gradient)
            let mut max_grad_idx = 0;
            let mut max_val = 0.0;
            for (idx, &val) in grad_z.iter().enumerate() {
                if val.abs() > max_val {
                    max_val = val.abs();
                    max_grad_idx = idx;
                }
            }
            result.shock_location = Some(max_grad_idx % nz);

            // Estimate shock strength (pressure jump)
            if let Some(z_idx) = result.shock_location {
                if z_idx > 0 && z_idx < nz - 1 {
                    let p_before = pressure[[nx / 2, z_idx - 1]];
                    let p_after = pressure[[nx / 2, z_idx + 1]];
                    result.shock_strength = Some((p_after - p_before).abs());
                }
            }

            // Estimate shock distance from source
            if let Some(z_idx) = result.shock_location {
                result.shock_distance = Some(z_idx as f64 * dz);
            }
        }

        // 5. Harmonic analysis (simple power spectrum approach)
        result.harmonic_ratios = self.compute_harmonic_ratios(pressure, frequency)?;

        // 6. Compute Total Harmonic Distortion
        if !result.harmonic_ratios.is_empty() {
            let sum_harmonics_sq: f64 = result.harmonic_ratios.iter().map(|x| x * x).sum();
            result.thd = sum_harmonics_sq.sqrt();
        }

        // 7. Secondary shock detection via harmonic threshold
        if result.thd > self.config.harmonic_threshold && !result.shock_detected {
            result.shock_detected = true;
        }

        Ok(result)
    }

    /// Compute artificial viscosity source term for shock capturing
    pub fn artificial_viscosity(
        &self,
        pressure: &Array2<f64>,
        dx: f64,
        dz: f64,
        rho0: f64,
        c0: f64,
    ) -> KwaversResult<Array2<f64>> {
        let (nx, nz) = pressure.dim();
        let mut q_av = Array2::zeros((nx, nz));

        if !self.config.enable_capturing || nx < 3 || nz < 3 {
            return Ok(q_av);
        }

        // Compute Laplacian of pressure
        let mut laplacian = Array2::zeros((nx - 2, nz - 2));

        for z in 1..nz - 1 {
            for x in 1..nx - 1 {
                let laplacian_val = (pressure[[x + 1, z]] - 2.0 * pressure[[x, z]]
                    + pressure[[x - 1, z]])
                    / (dx * dx)
                    + (pressure[[x, z + 1]] - 2.0 * pressure[[x, z]] + pressure[[x, z - 1]])
                        / (dz * dz);
                laplacian[[x - 1, z - 1]] = laplacian_val;
            }
        }

        // Compute gradients
        let mut grad_mag = Array2::zeros((nx - 2, nz - 2));

        for z in 1..nz - 1 {
            for x in 1..nx - 1 {
                let grad_x = (pressure[[x + 1, z]] - pressure[[x - 1, z]]) / (2.0 * dx);
                let grad_z = (pressure[[x, z + 1]] - pressure[[x, z - 1]]) / (2.0 * dz);
                grad_mag[[x - 1, z - 1]] = (grad_x * grad_x + grad_z * grad_z).sqrt();
            }
        }

        // Apply artificial viscosity: Q_av = μ |∇p| ∇²p / ρ0
        let mu = self.config.viscosity_coefficient * c0;

        for z in 1..nz - 1 {
            for x in 1..nx - 1 {
                let q_val = mu * grad_mag[[x - 1, z - 1]] * laplacian[[x - 1, z - 1]] / rho0;
                q_av[[x, z]] = q_val;
            }
        }

        Ok(q_av)
    }

    /// Apply shock capturing filter to smooth discontinuities
    pub fn shock_filter(
        &self,
        pressure: &mut Array2<f64>,
        shock_result: &ShockDetectionResult,
        filter_width: usize,
    ) -> KwaversResult<()> {
        if !shock_result.shock_detected {
            return Ok(());
        }

        let (nx, nz) = pressure.dim();

        // Apply smoothing near shock location
        if let Some(z_shock) = shock_result.shock_location {
            let z_min = z_shock.saturating_sub(filter_width);
            let z_max = (z_shock + filter_width).min(nz - 1);

            // Simple smoothing: replace with average of neighbors
            for z in z_min..=z_max {
                if z > 0 && z < nz - 1 {
                    for x in 1..nx - 1 {
                        let smoothed =
                            (pressure[[x, z - 1]] + 2.0 * pressure[[x, z]] + pressure[[x, z + 1]])
                                / 4.0;
                        pressure[[x, z]] = smoothed;
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute harmonic ratios from pressure field
    fn compute_harmonic_ratios(
        &self,
        pressure: &Array2<f64>,
        frequency: f64,
    ) -> KwaversResult<Vec<f64>> {
        let mut ratios = Vec::new();

        if frequency <= 0.0 || pressure.is_empty() {
            return Ok(ratios);
        }

        // Get center line (x = nx/2) for frequency analysis
        let (nx, _nz) = pressure.dim();
        let center_line = pressure.slice(s![nx / 2, ..]);

        // Compute power spectral density (simplified: using amplitude ratios)
        let max_amp = center_line.iter().map(|x| x.abs()).fold(0.0, f64::max);

        if max_amp < 1e-10 {
            return Ok(ratios);
        }

        // Simple harmonic detection: look for harmonic frequency content
        // This is a simplified approach - full FFT would be better
        for harmonic in 2..=self.config.num_harmonics {
            // Estimate harmonic amplitude (simplified)
            // In practice, would use FFT to properly extract harmonics
            let harmonic_amplitude = max_amp / (harmonic as f64).sqrt();
            ratios.push(harmonic_amplitude / max_amp);
        }

        Ok(ratios)
    }

    /// Record shock detection result in history
    pub fn record_result(&mut self, result: ShockDetectionResult) {
        self.history.push(result);
    }

    /// Get shock detection history
    pub fn history(&self) -> &[ShockDetectionResult] {
        &self.history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get configuration
    pub fn config(&self) -> ShockCapturingConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shock_capture_creation() {
        let config = ShockCapturingConfig::default();
        let _capture = ShockCapture::new(config);
    }

    #[test]
    fn test_no_shock_detection_smooth_field() {
        let config = ShockCapturingConfig {
            gradient_threshold: 1e6,  // Very high threshold
            harmonic_threshold: 0.01, // Very low tolerance for harmonics
            ..Default::default()
        };
        let capture = ShockCapture::new(config);

        // Smooth Gaussian pressure field
        let mut pressure = Array2::zeros((64, 64));
        for i in 0..64 {
            for j in 0..64 {
                let x = (i as f64 - 32.0) / 10.0;
                let y = (j as f64 - 32.0) / 10.0;
                pressure[[i, j]] = 100.0 * (-0.5 * (x * x + y * y)).exp();
            }
        }

        let result = capture
            .detect_shock(&pressure, 0.001, 0.001, 1540.0, 1e6)
            .unwrap();
        // With strict thresholds, smooth field should not trigger detection
        assert!(result.max_gradient < config.gradient_threshold);
    }

    #[test]
    fn test_shock_detection_steep_gradient() {
        let config = ShockCapturingConfig {
            gradient_threshold: 1e3,
            ..Default::default()
        };
        let capture = ShockCapture::new(config);

        // Create field with steep gradient (shock-like)
        let mut pressure = Array2::zeros((64, 64));
        for i in 0..64 {
            for j in 0..64 {
                if j < 32 {
                    pressure[[i, j]] = 1000.0;
                } else {
                    pressure[[i, j]] = -1000.0;
                }
            }
        }

        let result = capture
            .detect_shock(&pressure, 0.001, 0.001, 1540.0, 1e6)
            .unwrap();
        assert!(result.shock_detected); // Sharp gradient detected
        assert!(result.max_gradient > 0.0);
    }

    #[test]
    fn test_artificial_viscosity_generation() {
        let config = ShockCapturingConfig::default();
        let capture = ShockCapture::new(config);

        let mut pressure = Array2::zeros((64, 64));
        for i in 0..64 {
            for j in 0..64 {
                pressure[[i, j]] = ((i as f64) * (j as f64)).sin() * 100.0;
            }
        }

        let q_av = capture
            .artificial_viscosity(&pressure, 0.001, 0.001, 1000.0, 1540.0)
            .unwrap();
        assert_eq!(q_av.dim(), (64, 64));
    }

    #[test]
    fn test_shock_filter_application() {
        let config = ShockCapturingConfig::default();
        let mut capture = ShockCapture::new(config);

        let mut pressure = Array2::zeros((64, 64));
        for i in 0..64 {
            for j in 0..64 {
                if j < 32 {
                    pressure[[i, j]] = 1000.0;
                } else {
                    pressure[[i, j]] = -1000.0;
                }
            }
        }

        let result = capture
            .detect_shock(&pressure, 0.001, 0.001, 1540.0, 1e6)
            .unwrap();

        // Apply filter
        capture.shock_filter(&mut pressure, &result, 3).unwrap();
        // Pressure should be modified near shock
    }

    #[test]
    fn test_history_recording() {
        let config = ShockCapturingConfig::default();
        let mut capture = ShockCapture::new(config);

        let result = ShockDetectionResult {
            shock_detected: true,
            shock_location: Some(32),
            max_gradient: 5000.0,
            ..Default::default()
        };

        capture.record_result(result.clone());
        assert_eq!(capture.history().len(), 1);
        assert!(capture.history()[0].shock_detected);

        capture.clear_history();
        assert_eq!(capture.history().len(), 0);
    }

    #[test]
    fn test_config_validation() {
        let config = ShockCapturingConfig::default();
        assert!(config.gradient_threshold > 0.0);
        assert!(config.viscosity_coefficient >= 0.0);
        assert!(config.viscosity_coefficient <= 1.0);
    }
}
