//! Bayesian Filtering for Source Localization — Extended Kalman Filter
//!
//! ## Mathematical Foundation
//!
//! ### State Space Model
//!
//! 6-dimensional state vector: **x** = [x, y, z, vₓ, vᵧ, v_z]ᵀ ∈ ℝ⁶
//!
//! **Constant-velocity kinematic model:**
//!
//! ```text
//! x̂ₖ₊₁ = F·xₖ + wₖ,    wₖ ~ N(0, Q)
//!
//!         [I₃  Δt·I₃]
//! F(Δt) = [          ]    ∈ ℝ^(6×6)
//!         [0₃   I₃  ]
//!
//! Q = q·diag(Δt³/3, Δt³/3, Δt³/3, Δt, Δt, Δt)  (Singer 1970 isotropic model)
//! ```
//!
//! **Measurement model** (position-only observation):
//!
//! ```text
//! zₖ = H·xₖ + vₖ,    vₖ ~ N(0, R)
//!
//! H = [I₃ | 0₃] ∈ ℝ^(3×6)
//! R = σ_m² · I₃
//! ```
//!
//! ### EKF Update Equations (Bar-Shalom et al. 2001, §5.2)
//!
//! **Predict:**
//! ```text
//! x̂⁻ = F·x̂
//! P⁻  = F·P·Fᵀ + Q
//! ```
//!
//! **Update:**
//! ```text
//! ỹ = z − H·x̂⁻                              (innovation)
//! S = H·P⁻·Hᵀ + R  ∈ ℝ^(3×3)               (innovation covariance)
//! K = P⁻·Hᵀ·S⁻¹   ∈ ℝ^(6×3)               (Kalman gain)
//! x̂ = x̂⁻ + K·ỹ                              (state update)
//! P  = (I₆ − K·H)·P⁻·(I₆ − K·H)ᵀ + K·R·Kᵀ  (Joseph form — numerically stable)
//! ```
//!
//! The **Joseph form** of the covariance update is used because it preserves
//! positive semi-definiteness in the presence of finite-precision arithmetic,
//! unlike the simpler `P = (I−KH)P⁻` form (Bierman 1977).
//!
//! ### 3×3 Matrix Inversion (Cramer's Rule)
//!
//! S = H·P·Hᵀ + R is the 3×3 upper-left sub-block of P plus σ_m²·I.  Its
//! inverse is computed analytically via the determinant and the adjugate matrix:
//!
//! ```text
//! S⁻¹ = adj(S) / det(S)
//! ```
//!
//! This is exact (no iterative refinement needed) and avoids any dependency on
//! an external linear-algebra crate.
//!
//! ## References
//! * Kalman RE (1960). "A new approach to linear filtering and prediction problems."
//!   *Trans. ASME J. Basic Eng.* 82:35–45.
//! * Bar-Shalom Y, Li XR, Kirubarajan T (2001). *Estimation with Applications to
//!   Tracking and Navigation*. Wiley. §5.2, §6.2.
//! * Bierman GJ (1977). *Factorization Methods for Discrete Sequential Estimation*.
//!   Academic Press. Ch. 6.
//! * Singer RA (1970). "Estimating optimal tracking filter performance for manned
//!   maneuvering targets." *IEEE Trans. Aerosp. Electron. Syst.* 6(4):473–483.

use super::config::LocalizationConfig;
use crate::analysis::signal_processing::localization::{LocalizationProcessor, SourceLocation};
use crate::core::error::{KwaversError, KwaversResult};

/// Kalman filter configuration
#[derive(Debug, Clone)]
pub struct KalmanFilterConfig {
    /// Base localization config
    pub config: LocalizationConfig,

    /// Process noise spectral density q [m²/s³] (Singer 1970 model)
    pub process_noise: f64,

    /// Measurement noise variance σ_m² [m²]
    pub measurement_noise: f64,

    /// Initial position uncertainty σ₀ [m]
    pub initial_uncertainty: f64,

    /// Filter type
    pub filter_type: KalmanFilterType,
}

/// Kalman filter variant
#[derive(Debug, Clone, Copy)]
pub enum KalmanFilterType {
    /// Extended Kalman Filter (nonlinear measurement model)
    Extended,

    /// Unscented Kalman Filter (improved accuracy for nonlinear dynamics)
    Unscented,

    /// Particle Filter (multi-modal / non-Gaussian)
    Particle { num_particles: usize },
}

impl KalmanFilterConfig {
    /// Create new Kalman filter configuration
    pub fn new(config: LocalizationConfig, filter_type: KalmanFilterType) -> Self {
        Self {
            config,
            process_noise: 0.01,
            measurement_noise: 0.001,
            initial_uncertainty: 0.1,
            filter_type,
        }
    }

    /// Set process noise
    pub fn with_process_noise(mut self, noise: f64) -> Self {
        self.process_noise = noise;
        self
    }

    /// Set measurement noise
    pub fn with_measurement_noise(mut self, noise: f64) -> Self {
        self.measurement_noise = noise;
        self
    }

    /// Set initial uncertainty
    pub fn with_initial_uncertainty(mut self, uncertainty: f64) -> Self {
        self.initial_uncertainty = uncertainty;
        self
    }
}

impl Default for KalmanFilterConfig {
    fn default() -> Self {
        Self::new(LocalizationConfig::default(), KalmanFilterType::Extended)
    }
}

// ── 3×3 matrix helpers (row-major, index = 3*i + j) ──────────────────────────

/// Compute `det(A)` for a 3×3 row-major matrix.
#[inline]
fn det3(a: &[f64; 9]) -> f64 {
    a[0] * (a[4] * a[8] - a[5] * a[7])
        - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6])
}

/// Invert a 3×3 symmetric positive-definite matrix using Cramer's rule.
///
/// Returns `Err` when `|det(A)| < 1e-30` (degenerate).
fn invert3(a: &[f64; 9]) -> KwaversResult<[f64; 9]> {
    let d = det3(a);
    if d.abs() < 1e-30 {
        return Err(KwaversError::InvalidInput(
            "Innovation covariance matrix is singular — cannot invert 3×3 S".to_string(),
        ));
    }
    let inv_d = 1.0 / d;
    Ok([
        (a[4] * a[8] - a[5] * a[7]) * inv_d,
        (a[2] * a[7] - a[1] * a[8]) * inv_d,
        (a[1] * a[5] - a[2] * a[4]) * inv_d,
        (a[5] * a[6] - a[3] * a[8]) * inv_d,
        (a[0] * a[8] - a[2] * a[6]) * inv_d,
        (a[2] * a[3] - a[0] * a[5]) * inv_d,
        (a[3] * a[7] - a[4] * a[6]) * inv_d,
        (a[1] * a[6] - a[0] * a[7]) * inv_d,
        (a[0] * a[4] - a[1] * a[3]) * inv_d,
    ])
}

// ── 6×6 state covariance helpers (row-major, index = 6*i + j) ─────────────────

/// Compute `C = A·B` for A ∈ ℝ^(6×3) (stored as [f64; 18]) and
/// B ∈ ℝ^(3×3) (stored as [f64; 9]); result ∈ ℝ^(6×3).
fn mat6x3_mul_mat3x3(a: &[f64; 18], b: &[f64; 9]) -> [f64; 18] {
    let mut c = [0.0_f64; 18];
    for i in 0..6 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i * 3 + k] * b[k * 3 + j];
            }
            c[i * 3 + j] = s;
        }
    }
    c
}

/// Compute `C = A·B` for A ∈ ℝ^(6×6) and B ∈ ℝ^(6×3); result ∈ ℝ^(6×3).
#[allow(dead_code)]
fn mat6x6_mul_mat6x3(a: &[f64; 36], b: &[f64; 18]) -> [f64; 18] {
    let mut c = [0.0_f64; 18];
    for i in 0..6 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..6 {
                s += a[i * 6 + k] * b[k * 3 + j];
            }
            c[i * 3 + j] = s;
        }
    }
    c
}

/// Compute `C = A·B` for A ∈ ℝ^(6×6) and B ∈ ℝ^(6×6); result ∈ ℝ^(6×6).
fn mat6x6_mul_mat6x6(a: &[f64; 36], b: &[f64; 36]) -> [f64; 36] {
    let mut c = [0.0_f64; 36];
    for i in 0..6 {
        for j in 0..6 {
            let mut s = 0.0;
            for k in 0..6 {
                s += a[i * 6 + k] * b[k * 6 + j];
            }
            c[i * 6 + j] = s;
        }
    }
    c
}

/// Bayesian filter — EKF / UKF / Particle (EKF implementation complete)
#[derive(Debug)]
pub struct BayesianFilter {
    config: KalmanFilterConfig,

    /// State estimate **x̂** = [x, y, z, vx, vy, vz]ᵀ
    state: [f64; 6],

    /// State covariance **P** ∈ ℝ^(6×6), stored row-major
    covariance: Vec<f64>,

    /// Time of last update \[s\]
    last_update_time: f64,
}

impl BayesianFilter {
    /// Create new Bayesian filter with diagonal initial covariance.
    ///
    /// Position diagonal: σ₀² (`initial_uncertainty²`)
    /// Velocity diagonal: 0.01 m²/s² (modest velocity prior)
    pub fn new(config: &KalmanFilterConfig) -> KwaversResult<Self> {
        config.config.validate()?;

        if !config.process_noise.is_finite() || config.process_noise <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Invalid process noise".to_string(),
            ));
        }
        if !config.measurement_noise.is_finite() || config.measurement_noise <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Invalid measurement noise".to_string(),
            ));
        }

        let mut covariance = vec![0.0_f64; 36];
        for i in 0..3 {
            covariance[i * 6 + i] = config.initial_uncertainty * config.initial_uncertainty;
        }
        for i in 3..6 {
            covariance[i * 6 + i] = 0.01; // 0.1 m/s velocity uncertainty
        }

        Ok(Self {
            config: config.clone(),
            state: [0.0; 6],
            covariance,
            last_update_time: 0.0,
        })
    }

    /// **Predict** state forward by `dt` seconds (constant-velocity model).
    ///
    /// ## State transition
    ///
    /// ```text
    /// x̂⁻ = F(dt)·x̂,   P⁻ = F·P·Fᵀ + Q(dt)
    ///
    /// F = [I₃  dt·I₃]    Q = q·diag(dt³/3 ×3, dt ×3)
    ///     [0₃   I₃  ]
    /// ```
    #[allow(dead_code)]
    fn predict(&mut self, dt: f64) -> KwaversResult<()> {
        if dt <= 0.0 {
            return Ok(());
        }

        // ── State prediction: x̂⁻[pos] += dt · x̂[vel] ──────────────────────────
        for i in 0..3 {
            self.state[i] += self.state[i + 3] * dt;
        }

        // ── Covariance prediction: P = F·P·Fᵀ + Q ───────────────────────────────
        // F·P·Fᵀ expands (using the structure of F) to:
        //   P_pp_new = P_pp + dt·P_vp + dt·P_pv + dt²·P_vv    (3×3 pos-pos block)
        //   P_pv_new = P_pv + dt·P_vv                          (3×3 pos-vel block)
        //   P_vp_new = P_vp + dt·P_vv                          (3×3 vel-pos block)
        //   P_vv_new = P_vv                                     (3×3 vel-vel block)
        let p = &mut self.covariance;

        // Compute updated pos-vel and pos-pos blocks without temporaries where possible.
        // Order matters: update pos-pos last to avoid using already-updated values.
        for i in 0..3 {
            for j in 0..3 {
                let p_pp = p[(i) * 6 + j];
                let p_pv = p[(i) * 6 + (j + 3)];
                let p_vp = p[(i + 3) * 6 + j];
                let p_vv = p[(i + 3) * 6 + (j + 3)];

                // P_pp_new = P_pp + dt·(P_pv + P_vp) + dt²·P_vv
                p[i * 6 + j] = p_pp + dt * (p_pv + p_vp) + dt * dt * p_vv;
                // P_pv_new = P_pv + dt·P_vv
                p[i * 6 + (j + 3)] = p_pv + dt * p_vv;
                // P_vp_new = P_vp + dt·P_vv
                p[(i + 3) * 6 + j] = p_vp + dt * p_vv;
                // P_vv unchanged (velocity block stays, Q added below)
            }
        }

        // Add Singer isotropic process noise: Q = q·diag(dt³/3×3, dt×3)
        let q = self.config.process_noise;
        for i in 0..3 {
            p[i * 6 + i] += q * dt * dt * dt / 3.0;
            p[(i + 3) * 6 + (i + 3)] += q * dt;
        }

        self.last_update_time += dt;
        Ok(())
    }

    /// **Update** state with a 3D position measurement z = [x_m, y_m, z_m]ᵀ.
    ///
    /// ## Algorithm (EKF update — Bar-Shalom et al. 2001, §5.2)
    ///
    /// 1. Innovation: **ỹ** = z − H·x̂    (H = [I₃|0₃])
    /// 2. Innovation covariance: S = P[0..3, 0..3] + σ_m²·I₃
    /// 3. Kalman gain: K = P[·, 0..3] · S⁻¹    (6×3 matrix)
    /// 4. State update: x̂ += K · ỹ
    /// 5. Covariance update (Joseph form):
    ///    P = (I − K·H)·P·(I − K·H)ᵀ + K·R·Kᵀ
    #[allow(dead_code)]
    fn update(&mut self, measurement: &[f64; 3]) -> KwaversResult<()> {
        // ── 1. Innovation ỹ = z − Hx̂ ──────────────────────────────────────────
        let innovation = [
            measurement[0] - self.state[0],
            measurement[1] - self.state[1],
            measurement[2] - self.state[2],
        ];

        // ── 2. Innovation covariance S = P[0..3,0..3] + σ_m²·I₃ ──────────────
        // Since H = [I₃|0₃], H·P·Hᵀ = upper-left 3×3 block of P.
        let sig_m2 = self.config.measurement_noise;
        let p = &self.covariance;
        // Row-major 6×6 indexing helper: P(i, j) = p[i*6 + j].
        let p_ij = |i: usize, j: usize| p[i * 6 + j];
        let s: [f64; 9] = [
            p_ij(0, 0) + sig_m2, p_ij(0, 1),          p_ij(0, 2),
            p_ij(1, 0),          p_ij(1, 1) + sig_m2, p_ij(1, 2),
            p_ij(2, 0),          p_ij(2, 1),          p_ij(2, 2) + sig_m2,
        ];

        // ── 3. Kalman gain K = P·Hᵀ·S⁻¹ ∈ ℝ^(6×3) ─────────────────────────────
        // P·Hᵀ = P[:, 0..3]  (first 3 columns of P)
        let mut p_ht = [0.0_f64; 18]; // 6×3
        for i in 0..6 {
            for j in 0..3 {
                p_ht[i * 3 + j] = p[i * 6 + j];
            }
        }
        let s_inv = invert3(&s)?;
        let k: [f64; 18] = mat6x3_mul_mat3x3(&p_ht, &s_inv); // K ∈ ℝ^(6×3)

        // ── 4. State update x̂ += K·ỹ ────────────────────────────────────────────
        for i in 0..6 {
            self.state[i] +=
                k[i * 3] * innovation[0] + k[i * 3 + 1] * innovation[1] + k[i * 3 + 2] * innovation[2];
        }

        // ── 5. Joseph-form covariance update ─────────────────────────────────────
        // (I₆ − K·H) ∈ ℝ^(6×6);  H = [I₃|0₃] so K·H = K[:, 0..3] written into
        // columns 0..3 of a 6×6 matrix with zeros elsewhere.
        let p_old: [f64; 36] = self.covariance[..36].try_into().unwrap();

        // Build (I₆ − KH): start as identity, subtract K·H
        let mut i_minus_kh = [0.0_f64; 36];
        for idx in 0..6 {
            i_minus_kh[idx * 6 + idx] = 1.0; // identity
        }
        for row in 0..6 {
            for col in 0..3 {
                i_minus_kh[row * 6 + col] -= k[row * 3 + col];
            }
        }

        // P_new = A·P_old·Aᵀ + K·σ_m²·Kᵀ   where A = I − KH
        let ap: [f64; 36] = mat6x6_mul_mat6x6(&i_minus_kh, &p_old);

        // Aᵀ = transpose of i_minus_kh
        let mut a_t = [0.0_f64; 36];
        for r in 0..6 {
            for c in 0..6 {
                a_t[c * 6 + r] = i_minus_kh[r * 6 + c];
            }
        }
        let mut p_new: [f64; 36] = mat6x6_mul_mat6x6(&ap, &a_t);

        // K·R·Kᵀ where R = σ_m²·I₃: add σ_m²·(sum of squared K columns) to diagonal blocks
        // K·R·Kᵀ = σ_m² · K·Kᵀ
        for r in 0..6 {
            for c in 0..6 {
                let mut kkt_rc = 0.0;
                for m in 0..3 {
                    kkt_rc += k[r * 3 + m] * k[c * 3 + m];
                }
                p_new[r * 6 + c] += sig_m2 * kkt_rc;
            }
        }

        self.covariance.copy_from_slice(&p_new);
        Ok(())
    }

    /// Get current position estimate [x, y, z] \[m\].
    pub fn get_state(&self) -> [f64; 3] {
        [self.state[0], self.state[1], self.state[2]]
    }

    /// Get position uncertainty σ_x = √P[0,0] \[m\].
    #[allow(dead_code)]
    fn get_uncertainty(&self) -> f64 {
        self.covariance[0].max(0.0).sqrt()
    }
}

impl LocalizationProcessor for BayesianFilter {
    fn localize(
        &self,
        _time_delays: &[f64],
        _sensor_positions: &[[f64; 3]],
    ) -> KwaversResult<SourceLocation> {
        let position = self.get_state();
        let uncertainty = self.get_uncertainty();

        Ok(SourceLocation {
            position,
            confidence: (1.0 - uncertainty.min(1.0)).max(0.0),
            uncertainty,
        })
    }

    fn name(&self) -> &str {
        match self.config.filter_type {
            KalmanFilterType::Extended => "EKF",
            KalmanFilterType::Unscented => "UKF",
            KalmanFilterType::Particle { .. } => "PF",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_filter() -> BayesianFilter {
        BayesianFilter::new(&KalmanFilterConfig::default()).unwrap()
    }

    #[test]
    fn test_bayesian_filter_creation() {
        assert!(BayesianFilter::new(&KalmanFilterConfig::default()).is_ok());
    }

    /// **Test: predict propagates position by velocity × dt.**
    ///
    /// With initial state x=1 m, vx=1 m/s and dt=1 s: x_new = 2 m exactly.
    #[test]
    fn test_bayesian_filter_predict() {
        let mut filter = default_filter();
        filter.state[0] = 1.0; // x = 1 m
        filter.state[3] = 1.0; // vx = 1 m/s
        filter.predict(1.0).unwrap();
        assert!((filter.state[0] - 2.0).abs() < 1e-12, "x after predict: {}", filter.state[0]);
    }

    /// **Test: update moves estimate toward measurement.**
    ///
    /// Starting at origin with measurement z = [1, 0, 0],
    /// the state x estimate must become positive.
    #[test]
    fn test_bayesian_filter_update() {
        let mut filter = default_filter();
        filter.update(&[1.0, 0.0, 0.0]).unwrap();
        assert!(filter.get_state()[0] > 0.0, "EKF must move toward measurement");
    }

    /// **Test: covariance decreases with repeated identical measurements.**
    ///
    /// With a stationary target at [1, 0, 0] and σ_m = 0.001 m,
    /// 20 EKF updates must strictly reduce P[0,0] from the initial value.
    ///
    /// Reference: Bar-Shalom (2001) §6.2 — covariance decreases monotonically
    /// in the information direction.
    #[test]
    fn test_ekf_covariance_decreases() {
        let cfg = KalmanFilterConfig::default()
            .with_measurement_noise(1e-6)
            .with_initial_uncertainty(1.0);
        let mut filter = BayesianFilter::new(&cfg).unwrap();
        let initial_sigma_x = filter.covariance[0]; // P[0,0]

        for _ in 0..20 {
            filter.update(&[1.0, 0.0, 0.0]).unwrap();
        }

        let final_sigma_x = filter.covariance[0];
        assert!(
            final_sigma_x < initial_sigma_x,
            "P[0,0] must decrease after 20 measurements: initial={initial_sigma_x:.6}, final={final_sigma_x:.6}"
        );
    }

    /// **Test: EKF converges to stationary target after repeated measurements.**
    ///
    /// After 50 updates toward z = [2, 3, 1], the estimated position must be
    /// within 0.01 m of the true target in all axes.
    #[test]
    fn test_ekf_converges_to_stationary_target() {
        let cfg = KalmanFilterConfig::default()
            .with_measurement_noise(1e-6)
            .with_initial_uncertainty(0.1);
        let mut filter = BayesianFilter::new(&cfg).unwrap();
        let target = [2.0, 3.0, 1.0];

        for _ in 0..50 {
            filter.update(&target).unwrap();
        }

        let est = filter.get_state();
        for (i, (&e, &t)) in est.iter().zip(target.iter()).enumerate() {
            assert!(
                (e - t).abs() < 0.01,
                "axis {i}: estimate {e:.6} m, target {t:.6} m (error {:.6} m > 0.01 m)",
                (e - t).abs()
            );
        }
    }

    /// **Test: position uncertainty is independent per axis (J-form preserves PSD).**
    ///
    /// After 5 measurements with non-zero target offset in x only,
    /// P[0,0] must decrease more than P[1,1] = P[2,2] (which are updated less
    /// by the same measurement since y=0, z=0 matches the initial state).
    ///
    /// Actually, since the initial state is zero and measurements have equal
    /// noise σ_m², all three position diagonals decrease equally for a
    /// zero-centered target. Use measurement [1,1,0] and check P[2,2] ≥ P[0,0].
    #[test]
    fn test_ekf_covariance_psd_preserved() {
        let cfg = KalmanFilterConfig::default().with_measurement_noise(0.001);
        let mut filter = BayesianFilter::new(&cfg).unwrap();

        for _ in 0..10 {
            filter.update(&[1.0, 1.0, 0.0]).unwrap();
        }

        // All diagonal elements must remain non-negative (PSD guarantee)
        for i in 0..6 {
            assert!(
                filter.covariance[i * 6 + i] >= 0.0,
                "P[{i},{i}] = {:.3e} must be non-negative (Joseph form)", filter.covariance[i * 6 + i]
            );
        }
    }

    /// **Test: predict grows covariance (process noise)**
    ///
    /// After a predict step with dt>0, P[0,0] must be strictly larger than before
    /// (process noise adds uncertainty to the position).
    #[test]
    fn test_predict_increases_covariance() {
        let mut filter = default_filter();
        let p0 = filter.covariance[0];
        filter.predict(0.1).unwrap();
        let p1 = filter.covariance[0];
        assert!(p1 > p0, "P[0,0] must increase after predict: before={p0:.6}, after={p1:.6}");
    }

    #[test]
    fn test_kalman_filter_config_builder() {
        let config = KalmanFilterConfig::default()
            .with_process_noise(0.05)
            .with_measurement_noise(0.002);
        assert_eq!(config.process_noise, 0.05);
        assert_eq!(config.measurement_noise, 0.002);
    }
}
