//! Gaussian sub-pixel localization (Thompson et al. 2002).
//!
//! # Theorem: Gaussian Sub-pixel Localization
//!
//! **Theorem** (Thompson et al. 2002):
//! For a Gaussian PSF with FWHM and background noise, localization precision is:
//! ```text
//! σ_loc ≈ PSF_FWHM / (2√(2·ln2) · √SNR)
//! ```
//!
//! **Algorithm (Gauss-Newton least squares, 5×5 neighbourhood)**:
//! ```text
//! 1. Threshold envelope at 3·σ_noise; find local maxima on 2D grid
//! 2. For each candidate (x₀, z₀): fit
//!       I(x,z) = A·exp(−[(x−x₀)²+(z−z₀)²]/(2σ²)) + bg
//!    using Gauss-Newton 5×5 neighbourhood
//! 3. Accept if: A > 2·bg AND σ ∈ [0.3λ, 2λ]
//! ```

use super::types::{BubbleDetection, GaussianLocalizationConfig};
use crate::core::error::KwaversResult;
use ndarray::{s, Array2};

/// Sub-pixel microbubble localizer using Gauss-Newton least squares.
#[derive(Debug)]
pub struct GaussianLocalizer {
    config: GaussianLocalizationConfig,
}

impl GaussianLocalizer {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(config: GaussianLocalizationConfig) -> Self {
        Self { config }
    }

    /// Detect and localize microbubbles in a 2D envelope frame.
    ///
    /// # Arguments
    /// * `envelope` — 2D array \[N_z × N_x\], envelope-detected bubble signal
    /// * `frame_idx` — frame index stored in each detection
    ///
    /// # Returns
    /// Vector of sub-pixel localizations passing quality filters.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn localize_frame(
        &self,
        envelope: &Array2<f64>,
        frame_idx: usize,
    ) -> KwaversResult<Vec<BubbleDetection>> {
        let (n_z, n_x) = (envelope.nrows(), envelope.ncols());
        if n_z < 3 || n_x < 3 {
            return Ok(Vec::new());
        }

        let noise_std = noise_std_estimate(envelope);
        let threshold = self.config.threshold_sigma_multiplier * noise_std;

        let mut detections = Vec::new();
        let hw = self.config.fit_half_width;

        for iz in 1..(n_z - 1) {
            for ix in 1..(n_x - 1) {
                let v = envelope[[iz, ix]];
                if v < threshold {
                    continue;
                }
                if !is_local_max(envelope, iz, ix) {
                    continue;
                }
                let iz0 = iz.saturating_sub(hw);
                let iz1 = (iz + hw + 1).min(n_z);
                let ix0 = ix.saturating_sub(hw);
                let ix1 = (ix + hw + 1).min(n_x);
                let patch = envelope.slice(s![iz0..iz1, ix0..ix1]).to_owned();

                if let Some((z_sub, x_sub, amp, sigma, bg)) = gauss_newton_fit_2d(
                    &patch,
                    iz as f64 - iz0 as f64,
                    ix as f64 - ix0 as f64,
                    self.config.max_gauss_newton_iter,
                ) {
                    if amp <= self.config.min_snr_ratio * bg.max(noise_std) {
                        continue;
                    }
                    if sigma < self.config.min_sigma_px || sigma > self.config.max_sigma_px {
                        continue;
                    }
                    detections.push(BubbleDetection {
                        x: ix0 as f64 + x_sub,
                        z: iz0 as f64 + z_sub,
                        amplitude: amp,
                        sigma,
                        background: bg,
                        frame: frame_idx,
                    });
                }
            }
        }

        Ok(detections)
    }
}

/// Noise standard deviation estimate via median absolute deviation.
///
/// σ̂ = MAD(envelope) / 0.6745   (consistent for Gaussian noise)
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
fn noise_std_estimate(a: &Array2<f64>) -> f64 {
    let mut vals: Vec<f64> = a.iter().copied().collect();
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let med = if vals.len().is_multiple_of(2) {
        (vals[vals.len() / 2 - 1] + vals[vals.len() / 2]) / 2.0
    } else {
        vals[vals.len() / 2]
    };
    let mut devs: Vec<f64> = vals.iter().map(|&x| (x - med).abs()).collect();
    devs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mad = if devs.len().is_multiple_of(2) {
        (devs[devs.len() / 2 - 1] + devs[devs.len() / 2]) / 2.0
    } else {
        devs[devs.len() / 2]
    };
    mad / 0.6745
}

/// Returns true iff `envelope[[iz, ix]]` is a strict local maximum in its 3×3 neighbourhood.
fn is_local_max(a: &Array2<f64>, iz: usize, ix: usize) -> bool {
    let center = a[[iz, ix]];
    let (nz, nx) = (a.nrows(), a.ncols());
    for dz in [-1i32, 0, 1] {
        for dx in [-1i32, 0, 1] {
            if dz == 0 && dx == 0 {
                continue;
            }
            let niz = iz as i32 + dz;
            let nix = ix as i32 + dx;
            if niz < 0 || nix < 0 || niz >= nz as i32 || nix >= nx as i32 {
                continue;
            }
            if a[[niz as usize, nix as usize]] >= center {
                return false;
            }
        }
    }
    true
}

/// Gauss-Newton 2D Gaussian fit on a small patch.
///
/// # Theorem
/// Given observed intensities I(zᵢ, xⱼ), fit the model:
/// ```text
/// f(z,x; θ) = A·exp(−[(z−z₀)²+(x−x₀)²]/(2σ²)) + bg
/// ```
/// Parameters θ = (A, z₀, x₀, σ, bg) optimized by Gauss-Newton:
/// ```text
/// θ_{t+1} = θ_t − (JᵀJ)⁻¹ Jᵀ r     where r = I − f(z,x;θ)
/// ```
///
/// Returns `Some((z_center, x_center, amplitude, sigma, background))` in patch
/// coordinates on convergence, `None` on numerical failure.
pub(super) fn gauss_newton_fit_2d(
    patch: &Array2<f64>,
    z0_init: f64,
    x0_init: f64,
    max_iter: usize,
) -> Option<(f64, f64, f64, f64, f64)> {
    let (nz, nx) = (patch.nrows(), patch.ncols());
    let n = nz * nx;

    let bg_init = patch.iter().copied().fold(f64::MAX, f64::min);
    let amp_init = patch.iter().copied().fold(f64::MIN, f64::max) - bg_init;
    if amp_init <= 0.0 {
        return None;
    }

    let mut z0 = z0_init;
    let mut x0 = x0_init;
    let mut amp = amp_init.max(1e-30);
    let mut sigma = 1.0_f64;
    let mut bg = bg_init;

    let mut rss_prev = f64::INFINITY;

    for _iter in 0..max_iter {
        let mut jt_j = [[0.0_f64; 5]; 5];
        let mut jt_r = [0.0_f64; 5];
        let mut rss_curr = 0.0_f64;

        for iz in 0..nz {
            for ix in 0..nx {
                let z = iz as f64;
                let x = ix as f64;
                let dz = z - z0;
                let dx = x - x0;
                let r2 = dz.mul_add(dz, dx * dx) / (2.0 * sigma * sigma);
                let g = (-r2).exp();
                let model = amp * g + bg;
                let resid = patch[[iz, ix]] - model;
                rss_curr += resid * resid;

                let df_amp = g;
                let df_z0 = amp * g * dz / (sigma * sigma);
                let df_x0 = amp * g * dx / (sigma * sigma);
                let df_sigma = amp * g * dz.mul_add(dz, dx * dx) / (sigma * sigma * sigma);
                let df_bg = 1.0;

                let j = [df_amp, df_z0, df_x0, df_sigma, df_bg];
                for p in 0..5 {
                    for q in 0..5 {
                        jt_j[p][q] += j[p] * j[q];
                    }
                    jt_r[p] += j[p] * resid;
                }
            }
        }

        let delta = solve_5x5(&jt_j, &jt_r)?;

        amp += delta[0];
        z0 += delta[1];
        x0 += delta[2];
        sigma += delta[3];
        bg += delta[4];

        amp = amp.max(1e-30);
        sigma = sigma.clamp(0.1, (nz.max(nx)) as f64);
        z0 = z0.clamp(0.0, (nz - 1) as f64);
        x0 = x0.clamp(0.0, (nx - 1) as f64);

        let step_sq: f64 = delta.iter().map(|&d| d * d).sum();
        if step_sq < 1e-12 * n as f64 {
            break;
        }

        if rss_prev.is_finite() && (rss_prev - rss_curr).abs() / (rss_prev + 1e-30) < 1e-10 {
            break;
        }
        rss_prev = rss_curr;
    }

    Some((z0, x0, amp, sigma, bg))
}

/// Gaussian elimination solver for a 5×5 system Aδ = b.
///
/// Returns `None` if the matrix is singular (pivot < 1e-12).
#[allow(clippy::needless_range_loop)]
fn solve_5x5(a: &[[f64; 5]; 5], b: &[f64; 5]) -> Option<[f64; 5]> {
    let mut m = [[0.0_f64; 6]; 5];
    for i in 0..5 {
        for j in 0..5 {
            m[i][j] = a[i][j];
        }
        m[i][5] = b[i];
    }

    for col in 0..5 {
        let mut max_row = col;
        let mut max_val = m[col][col].abs();
        for row in (col + 1)..5 {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        m.swap(col, max_row);

        let pivot = m[col][col];
        for row in (col + 1)..5 {
            let factor = m[row][col] / pivot;
            for k in col..=5 {
                let v = m[col][k];
                m[row][k] -= factor * v;
            }
        }
    }

    let mut x = [0.0_f64; 5];
    for i in (0..5).rev() {
        x[i] = m[i][5];
        for j in (i + 1)..5 {
            x[i] -= m[i][j] * x[j];
        }
        x[i] /= m[i][i];
    }

    Some(x)
}
