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
//! ```text
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
//! ```text
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

use crate::core::constants::SOUND_SPEED_TISSUE;
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::fft::fft_1d_array;
use ndarray::{s, Array1, Array2};
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
            shock_velocity_estimate: SOUND_SPEED_TISSUE, // m/s (soft tissue)
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

        // 5. Harmonic analysis via FFT-based spectral decomposition
        result.harmonic_ratios = self.compute_harmonic_ratios(pressure, frequency, dz)?;

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

    /// Compute harmonic distortion ratios from the FFT spectrum of the axial pressure.
    ///
    /// ## Algorithm — IEEE Std 519-2014, §3.1 (THD via DFT)
    ///
    /// For a pressure waveform p(z) sampled with spacing `dz` over N points:
    ///
    /// 1. Compute the N-point DFT: `P[k] = Σ p[n] exp(−i·2π·k·n/N)`
    /// 2. The bin index of the fundamental: `k₀ = round(f₀ · N · dz)`
    ///    (since the sampling frequency is `fs = 1/dz` Hz and `f₀ = frequency`).
    /// 3. The amplitude of the n-th harmonic: `A_n = |P[n·k₀]| / N`
    /// 4. The n-th harmonic ratio: `r_n = A_n / A₁`
    ///
    /// Physical grounding: nonlinear acoustic propagation transfers energy from
    /// the fundamental to harmonics at rates governed by the Fay/Fubini solution
    /// (Blackstock 1966). FFT-based THD directly measures this energy transfer.
    ///
    /// ## Guard conditions
    /// - Frequency ≤ 0 or empty field → return `[]`
    /// - Fundamental bin = 0 (DC) → return `[]` (unphysical)
    /// - Fundamental amplitude < 1e-10 → return `[]` (no signal)
    /// - n·k₀ ≥ N/2 (Nyquist) → skip that harmonic (aliased)
    ///
    /// ## Reference
    /// - Blackstock, D.T. (1966). J. Acoust. Soc. Am. 39(6), 1019–1026.
    /// - IEEE Std 519-2014, §3.1: Total Harmonic Distortion.
    fn compute_harmonic_ratios(
        &self,
        pressure: &Array2<f64>,
        frequency: f64,
        dz: f64,
    ) -> KwaversResult<Vec<f64>> {
        if frequency <= 0.0 || dz <= 0.0 || pressure.is_empty() {
            return Ok(Vec::new());
        }

        let (nx, nz) = pressure.dim();
        if nz < 4 {
            return Ok(Vec::new());
        }

        // Extract axial centre line: p[z] for x = nx/2
        let centre_line: Array1<f64> = pressure.slice(s![nx / 2, ..]).to_owned();

        // Fundamental bin: k₀ = round(f₀ · N · dz)
        // Derivation: fs = 1/dz → bin for frequency f = f·N/fs = f·N·dz
        let k0_f = frequency * (nz as f64) * dz;
        if k0_f < 0.5 {
            // Fundamental falls at or below DC bin — unphysical
            return Ok(Vec::new());
        }
        let k0 = k0_f.round() as usize;
        if k0 == 0 || k0 >= nz / 2 {
            return Ok(Vec::new());
        }

        // Forward DFT of the centre-line pressure
        let spectrum = fft_1d_array(&centre_line);
        let n_inv = 1.0 / nz as f64;

        // Fundamental amplitude: A₁ = |P[k₀]| / N
        let a1 = spectrum[k0].norm() * n_inv;
        if a1 < 1e-10 {
            return Ok(Vec::new()); // No signal at fundamental
        }

        // Extract harmonic ratios: r_n = A_n / A₁ for n = 2..=num_harmonics
        let mut ratios = Vec::with_capacity(self.config.num_harmonics.saturating_sub(1));
        for harmonic in 2..=self.config.num_harmonics {
            let k_n = k0 * harmonic;
            if k_n >= nz / 2 {
                break; // Beyond Nyquist — no valid data
            }
            let a_n = spectrum[k_n].norm() * n_inv;
            ratios.push(a_n / a1);
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
        let capture = ShockCapture::new(config);

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

    // -----------------------------------------------------------------------
    // Tests for FFT-based compute_harmonic_ratios
    // -----------------------------------------------------------------------

    /// A pure sine at the fundamental has zero energy at all harmonics.
    ///
    /// Proof: sin(2π·f₀·n·dz) has DFT energy only at bin k₀ = round(f₀·N·dz).
    /// All harmonic bins k = 2k₀, 3k₀, … have |P[k]| = 0 (up to floating-point
    /// noise) → all harmonic ratios ≈ 0.
    #[test]
    fn test_harmonic_ratios_pure_sine_is_zero() {
        let config = ShockCapturingConfig {
            num_harmonics: 3,
            ..Default::default()
        };
        let capture = ShockCapture::new(config);

        // Build a 2D pressure field with a pure 100 Hz sine on centre row
        let nz = 256;
        let nx = 5;
        let dz = 1.0 / 1000.0; // 1 mm spacing → fs = 1000 Hz
        let f0 = 100.0_f64;    // fundamental frequency
        let mut pressure = Array2::zeros((nx, nz));
        for z in 0..nz {
            let val = (2.0 * std::f64::consts::PI * f0 * z as f64 * dz).sin() * 1000.0;
            for x in 0..nx {
                pressure[[x, z]] = val;
            }
        }

        let ratios = capture.compute_harmonic_ratios(&pressure, f0, dz).unwrap();

        // All harmonic ratios should be near zero for a pure sine
        for (n, &r) in ratios.iter().enumerate() {
            assert!(
                r < 0.01,
                "harmonic {} ratio should be ~0 for pure sine, got {:.4e}",
                n + 2, r
            );
        }
    }

    /// Synthesised signal with known second harmonic amplitude yields correct ratio.
    ///
    /// Signal: p(z) = A₁·sin(2π·f₀·z) + A₂·sin(2π·2f₀·z)
    /// Expected ratio: r₂ = A₂ / A₁
    #[test]
    fn test_harmonic_ratios_known_second_harmonic() {
        let config = ShockCapturingConfig {
            num_harmonics: 3,
            ..Default::default()
        };
        let capture = ShockCapture::new(config);

        let nz = 512;
        let nx = 5;
        let dz = 1.0 / 5120.0; // fs = 5120 Hz
        let f0 = 200.0_f64;    // fundamental: bin = 200/5120*512 = 20
        let a1 = 1000.0_f64;
        let a2 = 200.0_f64;    // second harmonic: 20% of fundamental

        let mut pressure = Array2::zeros((nx, nz));
        for z in 0..nz {
            let t = z as f64 * dz;
            let val = a1 * (2.0 * std::f64::consts::PI * f0 * t).sin()
                + a2 * (2.0 * std::f64::consts::PI * 2.0 * f0 * t).sin();
            for x in 0..nx {
                pressure[[x, z]] = val;
            }
        }

        let ratios = capture.compute_harmonic_ratios(&pressure, f0, dz).unwrap();

        assert!(!ratios.is_empty(), "should produce at least one ratio");
        let expected_r2 = a2 / a1;
        let measured_r2 = ratios[0];
        let err = (measured_r2 - expected_r2).abs();
        assert!(
            err < 0.02,
            "second harmonic ratio: expected {:.3}, got {:.3} (err={:.2e})",
            expected_r2, measured_r2, err
        );
    }

    /// Empty / zero pressure field returns empty ratios (no division by zero).
    #[test]
    fn test_harmonic_ratios_zero_field_returns_empty() {
        let config = ShockCapturingConfig::default();
        let capture = ShockCapture::new(config);

        let pressure = Array2::zeros((8, 128));
        let ratios = capture.compute_harmonic_ratios(&pressure, 100.0, 1e-4).unwrap();
        assert!(ratios.is_empty(), "zero field should yield empty ratios");
    }

    /// Invalid frequency (zero) returns empty ratios without error.
    #[test]
    fn test_harmonic_ratios_invalid_frequency_returns_empty() {
        let config = ShockCapturingConfig::default();
        let capture = ShockCapture::new(config);

        let mut pressure = Array2::zeros((8, 64));
        pressure[[4, 10]] = 1.0;
        let ratios = capture.compute_harmonic_ratios(&pressure, 0.0, 1e-4).unwrap();
        assert!(ratios.is_empty());
    }
}
