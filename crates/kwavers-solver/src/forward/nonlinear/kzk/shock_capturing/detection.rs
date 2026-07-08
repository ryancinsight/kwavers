//! Shock detection and harmonic analysis.

use super::{ShockCapture, ShockDetectionResult};
use apollo::fft_1d_leto;
use kwavers_core::constants::tissue_acoustics::B_OVER_A_WATER_37C;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array1 as LetoArray1;
use ndarray::Array2;

impl ShockCapture {
    /// Create new shock capture instance
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(config: super::ShockCapturingConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Detect shock formation from pressure field
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
                "Pressure field too small for shock detection".to_owned(),
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

        // 2. Maximum gradient
        result.max_gradient = grad_x
            .iter()
            .chain(grad_z.iter())
            .map(|x| x.abs())
            .fold(0.0, f64::max);

        // 3. Steepness parameter τ_shock ≈ 1 / (β c₀ p |∇p|)
        let max_pressure = pressure.iter().map(|x| x.abs()).fold(0.0, f64::max);

        if max_pressure > 1.0 && result.max_gradient > 1e-6 {
            // β = 1 + B/(2A); for water at body temperature B/A = 5.0 → β = 3.5.
            let beta = 1.0 + B_OVER_A_WATER_37C / 2.0;
            result.steepness_parameter =
                1.0 / (beta * c0 * max_pressure * result.max_gradient).max(1e-10);
        }

        // 4. Detect shock by gradient threshold
        if result.max_gradient > self.config.gradient_threshold {
            result.shock_detected = true;

            let mut max_grad_idx = 0;
            let mut max_val = 0.0;
            for (idx, &val) in grad_z.iter().enumerate() {
                if val.abs() > max_val {
                    max_val = val.abs();
                    max_grad_idx = idx;
                }
            }
            result.shock_location = Some(max_grad_idx % nz);

            if let Some(z_idx) = result.shock_location {
                if z_idx > 0 && z_idx < nz - 1 {
                    let p_before = pressure[[nx / 2, z_idx - 1]];
                    let p_after = pressure[[nx / 2, z_idx + 1]];
                    result.shock_strength = Some((p_after - p_before).abs());
                }
            }

            if let Some(z_idx) = result.shock_location {
                result.shock_distance = Some(z_idx as f64 * dz);
            }
        }

        // 5. Harmonic analysis via FFT
        result.harmonic_ratios = self.compute_harmonic_ratios(pressure, frequency, dz)?;

        // 6. Total Harmonic Distortion
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

    /// Compute harmonic distortion ratios from FFT spectrum of the axial pressure.
    ///
    /// IEEE Std 519-2014, §3.1: `P[k] = Σ p[n] exp(−i·2π·k·n/N)`.
    /// Fundamental bin: `k₀ = round(f₀ · N · dz)`.
    /// Harmonic ratio: `r_n = |P[n·k₀]| / |P[k₀]|`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn compute_harmonic_ratios(
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

        let centre_line = (0..nz).map(|z| pressure[[nx / 2, z]]).collect::<Vec<_>>();

        let k0_f = frequency * (nz as f64) * dz;
        if k0_f < 0.5 {
            return Ok(Vec::new());
        }
        let k0 = k0_f.round() as usize;
        if k0 == 0 || k0 >= nz / 2 {
            return Ok(Vec::new());
        }

        let centre_line = LetoArray1::from_shape_vec([nz], centre_line)
            .expect("shock-detection centre line length must match its Leto shape");
        let spectrum = fft_1d_leto(centre_line.view());
        let n_inv = 1.0 / nz as f64;

        let a1 = spectrum[k0].norm() * n_inv;
        if a1 < 1e-10 {
            return Ok(Vec::new());
        }

        let mut ratios = Vec::with_capacity(self.config.num_harmonics.saturating_sub(1));
        for harmonic in 2..=self.config.num_harmonics {
            let k_n = k0 * harmonic;
            if k_n >= nz / 2 {
                break;
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
    #[must_use]
    pub fn history(&self) -> &[ShockDetectionResult] {
        &self.history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> super::ShockCapturingConfig {
        self.config
    }
}
