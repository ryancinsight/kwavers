//! Regularization methods for FWI
//! Based on Tikhonov & Arsenin (1977): "Solutions of Ill-posed Problems"

use ndarray::{Array3, Zip};

/// Regularization methods for FWI
#[derive(Debug)]
pub struct Regularizer {
    /// Tikhonov regularization weight
    tikhonov_weight: f64,
    /// Total variation weight
    tv_weight: f64,
    /// Smoothness weight
    smoothness_weight: f64,
}

impl Default for Regularizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Regularizer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tikhonov_weight: 0.01,
            tv_weight: 0.0,
            smoothness_weight: 0.0,
        }
    }

    /// Apply combined regularization to gradient
    pub fn apply_regularization(&self, gradient: &mut Array3<f64>, model: &Array3<f64>) {
        if self.tikhonov_weight > 0.0 {
            self.apply_tikhonov(gradient, model);
        }

        if self.tv_weight > 0.0 {
            self.apply_total_variation(gradient, model);
        }

        if self.smoothness_weight > 0.0 {
            self.apply_smoothness(gradient, model);
        }
    }

    /// Tikhonov (L2) regularization
    /// Penalizes large model values
    fn apply_tikhonov(&self, gradient: &mut Array3<f64>, model: &Array3<f64>) {
        let w = self.tikhonov_weight;
        Zip::from(gradient).and(model).par_for_each(|g, &m| {
            *g += w * m;
        });
    }

    /// Total Variation regularization — Rudin, Osher & Fatemi (1992) Eq. (11).
    ///
    /// Computes the full discrete divergence of the normalised gradient field:
    ///   ∂TV/∂m[i,j,k] = −(fx[i] + fy[i] + fz[i])/W[i]
    ///                   + fx[i−1,j,k]/W[i−1,j,k]   (if i > 0)
    ///                   + fy[i,j−1,k]/W[i,j−1,k]   (if j > 0)
    ///                   + fz[i,j,k−1]/W[i,j,k−1]   (if k > 0)
    ///
    /// Previous code omitted the backward-neighbor contributions, producing
    /// only −(fx+fy+fz)/W — the truncated half of the true functional derivative.
    /// That is the same error fixed in `solver::inverse::seismic::fwi::gradient`
    /// as Bug 57 (Rudin-Osher-Fatemi 1992).
    fn apply_total_variation(&self, gradient: &mut Array3<f64>, model: &Array3<f64>) {
        let (nx, ny, nz) = model.dim();
        let eps2 = 1e-16_f64; // Huber ε² (squared) prevents division by zero

        // Pre-compute forward differences.
        let mut fx = Array3::<f64>::zeros((nx, ny, nz)); // fx[i,j,k] = m[i+1,j,k] − m[i,j,k]
        let mut fy = Array3::<f64>::zeros((nx, ny, nz));
        let mut fz = Array3::<f64>::zeros((nx, ny, nz));

        for i in 0..nx - 1 {
            for j in 0..ny {
                for k in 0..nz {
                    fx[[i, j, k]] = model[[i + 1, j, k]] - model[[i, j, k]];
                }
            }
        }
        for i in 0..nx {
            for j in 0..ny - 1 {
                for k in 0..nz {
                    fy[[i, j, k]] = model[[i, j + 1, k]] - model[[i, j, k]];
                }
            }
        }
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz - 1 {
                    fz[[i, j, k]] = model[[i, j, k + 1]] - model[[i, j, k]];
                }
            }
        }

        // Huber weight W[i,j,k] = √(fx² + fy² + fz² + ε²).
        let mut w = Array3::<f64>::zeros((nx, ny, nz));
        Zip::from(&mut w)
            .and(&fx)
            .and(&fy)
            .and(&fz)
            .par_for_each(|w_val, &dx, &dy, &dz| {
                *w_val = dz.mul_add(dz, dx.mul_add(dx, dy * dy) + eps2).sqrt();
            });

        // Functional derivative: divergence of the normalised gradient field.
        let tv_w = self.tv_weight;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let w_c = w[[i, j, k]];
                    // Negative self-contribution from forward differences leaving (i,j,k).
                    let mut g = -(fx[[i, j, k]] + fy[[i, j, k]] + fz[[i, j, k]]) / w_c;
                    // Positive back-contributions from forward differences ending at (i,j,k).
                    if i > 0 {
                        g += fx[[i - 1, j, k]] / w[[i - 1, j, k]];
                    }
                    if j > 0 {
                        g += fy[[i, j - 1, k]] / w[[i, j - 1, k]];
                    }
                    if k > 0 {
                        g += fz[[i, j, k - 1]] / w[[i, j, k - 1]];
                    }
                    gradient[[i, j, k]] += tv_w * g;
                }
            }
        }
    }

    /// Smoothness (Tikhonov L2 gradient) regularisation — adds ∂J_s/∂m = −w·∇²m.
    ///
    /// J_s = ½w·‖∇m‖²  →  ∂J_s/∂m = −w·∇²m  (integration by parts, zero BCs).
    ///
    /// Previous code computed `∇²(gradient)` and subtracted it from `gradient`.
    /// That is not the functional derivative of any standard regularisation term;
    /// `I − w∇²` is a high-pass filter that *amplifies* high-frequency gradient
    /// components rather than penalising model roughness (Bug 63).
    ///
    /// Fix: compute `∇²m` on the velocity model and add −w·∇²m to the gradient.
    fn apply_smoothness(&self, gradient: &mut Array3<f64>, model: &Array3<f64>) {
        let (nx, ny, nz) = model.dim();
        let w = self.smoothness_weight;

        // ∂J_s/∂m[i,j,k] = −w · ∇²m[i,j,k]
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let laplacian_m = model[[i + 1, j, k]]
                        + model[[i - 1, j, k]]
                        + model[[i, j + 1, k]]
                        + model[[i, j - 1, k]]
                        + model[[i, j, k + 1]]
                        + model[[i, j, k - 1]]
                        - 6.0 * model[[i, j, k]];
                    gradient[[i, j, k]] -= w * laplacian_m;
                }
            }
        }
    }

    /// Set regularization weights
    pub fn set_weights(&mut self, tikhonov: f64, tv: f64, smoothness: f64) {
        self.tikhonov_weight = tikhonov;
        self.tv_weight = tv;
        self.smoothness_weight = smoothness;
    }
}
