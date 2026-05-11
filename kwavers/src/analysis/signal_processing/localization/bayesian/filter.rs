use super::config::{KalmanFilterConfig, KalmanFilterType};
use super::linalg::{invert3, mat6x3_mul_mat3x3, mat6x6_mul_mat6x6};
use crate::analysis::signal_processing::localization::{LocalizationProcessor, SourceLocation};
use crate::core::error::{KwaversError, KwaversResult};

/// Bayesian filter — EKF / UKF / Particle (EKF implementation complete)
#[derive(Debug)]
pub struct BayesianFilter {
    config: KalmanFilterConfig,
    /// State estimate **x̂** = [x, y, z, vx, vy, vz]ᵀ
    pub(super) state: [f64; 6],
    /// State covariance **P** ∈ ℝ^(6×6), stored row-major
    pub(super) covariance: Vec<f64>,
}

impl BayesianFilter {
    /// Create new Bayesian filter with diagonal initial covariance.
    ///
    /// Position diagonal: σ₀² (`initial_uncertainty²`)
    /// Velocity diagonal: 0.01 m²/s² (modest velocity prior)
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: &KalmanFilterConfig) -> KwaversResult<Self> {
        config.config.validate()?;

        if !config.process_noise.is_finite() || config.process_noise <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Invalid process noise".to_owned(),
            ));
        }
        if !config.measurement_noise.is_finite() || config.measurement_noise <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Invalid measurement noise".to_owned(),
            ));
        }

        let mut covariance = vec![0.0_f64; 36];
        for i in 0..3 {
            covariance[i * 6 + i] = config.initial_uncertainty * config.initial_uncertainty;
        }
        for i in 3..6 {
            covariance[i * 6 + i] = 0.01;
        }

        Ok(Self {
            config: config.clone(),
            state: [0.0; 6],
            covariance,
        })
    }

    /// **Predict** state forward by `dt` seconds (constant-velocity model).
    ///
    /// ```text
    /// x̂⁻ = F(dt)·x̂,   P⁻ = F·P·Fᵀ + Q(dt)
    ///
    /// F = [I₃  dt·I₃]    Q = q·diag(dt³/3 ×3, dt ×3)
    ///     [0₃   I₃  ]
    /// ```
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn predict(&mut self, dt: f64) -> KwaversResult<()> {
        if dt <= 0.0 {
            return Ok(());
        }

        for i in 0..3 {
            self.state[i] += self.state[i + 3] * dt;
        }

        let p = &mut self.covariance;
        for i in 0..3 {
            for j in 0..3 {
                let p_pp = p[(i) * 6 + j];
                let p_pv = p[(i) * 6 + (j + 3)];
                let p_vp = p[(i + 3) * 6 + j];
                let p_vv = p[(i + 3) * 6 + (j + 3)];

                p[i * 6 + j] = (dt * dt).mul_add(p_vv, dt.mul_add(p_pv + p_vp, p_pp));
                p[i * 6 + (j + 3)] = dt.mul_add(p_vv, p_pv);
                p[(i + 3) * 6 + j] = dt.mul_add(p_vv, p_vp);
            }
        }

        let q = self.config.process_noise;
        for i in 0..3 {
            p[i * 6 + i] += q * dt * dt * dt / 3.0;
            p[(i + 3) * 6 + (i + 3)] += q * dt;
        }

        Ok(())
    }

    /// **Update** state with a 3D position measurement z = [x_m, y_m, z_m]ᵀ.
    ///
    /// EKF update — Bar-Shalom et al. 2001, §5.2:
    /// 1. Innovation: **ỹ** = z − H·x̂    (H = [I₃|0₃])
    /// 2. Innovation covariance: S = P[0..3, 0..3] + σ_m²·I₃
    /// 3. Kalman gain: K = P[·, 0..3] · S⁻¹    (6×3 matrix)
    /// 4. State update: x̂ += K · ỹ
    /// 5. Covariance update (Joseph form):
    ///    P = (I − K·H)·P·(I − K·H)ᵀ + K·R·Kᵀ
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn update(&mut self, measurement: &[f64; 3]) -> KwaversResult<()> {
        let innovation = [
            measurement[0] - self.state[0],
            measurement[1] - self.state[1],
            measurement[2] - self.state[2],
        ];

        let sig_m2 = self.config.measurement_noise;
        let p = &self.covariance;
        let p_ij = |i: usize, j: usize| p[i * 6 + j];
        let s: [f64; 9] = [
            p_ij(0, 0) + sig_m2,
            p_ij(0, 1),
            p_ij(0, 2),
            p_ij(1, 0),
            p_ij(1, 1) + sig_m2,
            p_ij(1, 2),
            p_ij(2, 0),
            p_ij(2, 1),
            p_ij(2, 2) + sig_m2,
        ];

        let mut p_ht = [0.0_f64; 18];
        for i in 0..6 {
            for j in 0..3 {
                p_ht[i * 3 + j] = p[i * 6 + j];
            }
        }
        let s_inv = invert3(&s)?;
        let k: [f64; 18] = mat6x3_mul_mat3x3(&p_ht, &s_inv);

        for i in 0..6 {
            self.state[i] += k[i * 3 + 2].mul_add(innovation[2], k[i * 3].mul_add(innovation[0], k[i * 3 + 1] * innovation[1]));
        }

        let p_old: [f64; 36] = self.covariance[..36].try_into().unwrap();

        let mut i_minus_kh = [0.0_f64; 36];
        for idx in 0..6 {
            i_minus_kh[idx * 6 + idx] = 1.0;
        }
        for row in 0..6 {
            for col in 0..3 {
                i_minus_kh[row * 6 + col] -= k[row * 3 + col];
            }
        }

        let ap: [f64; 36] = mat6x6_mul_mat6x6(&i_minus_kh, &p_old);

        let mut a_t = [0.0_f64; 36];
        for r in 0..6 {
            for c in 0..6 {
                a_t[c * 6 + r] = i_minus_kh[r * 6 + c];
            }
        }
        let mut p_new: [f64; 36] = mat6x6_mul_mat6x6(&ap, &a_t);

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use] 
    pub fn get_state(&self) -> [f64; 3] {
        [self.state[0], self.state[1], self.state[2]]
    }

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
