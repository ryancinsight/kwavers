//! Microbubble Detection and Sub-pixel Localization for ULM
//!
//! # Overview
//!
//! Implements the two-stage ULM detection pipeline:
//! 1. **SVD Clutter Filter** — separates microbubble signal from tissue clutter
//!    using spatiotemporal singular value decomposition.
//! 2. **Gaussian Sub-pixel Localization** — fits each detected peak to a 2D Gaussian
//!    to achieve sub-pixel centroid precision.
//!
//! # Theorem: Spatiotemporal SVD Clutter Filtering
//!
//! **Theorem** (Demené et al. 2015 §II-B):
//! The IQ data matrix S ∈ ℝ^{N_px × N_t} has the property that tissue clutter
//! concentrates in the first k left singular vectors (correlated over time),
//! while microbubbles decorrelate rapidly (δ-correlated).
//!
//! ```text
//! Decompose:  S = U Σ Vᵀ
//! Tissue:     T = U[:,0:k] diag(Σ[0:k]) Vᵀ[0:k,:]
//! Bubble:     B = S − T   (sparse, transient signal)
//! ```
//!
//! The optimal threshold k is given by the Singular Value Hard Threshold (SVHT)
//! (Gavish & Donoho 2014):
//! ```text
//! τ = ω(β) · σ_median     (the universal threshold)
//! where β = N_px/N_t (aspect ratio)
//! ω(β) = 0.56·β³ − 0.95·β² + 1.82·β + 1.43   (Gavish & Donoho 2014, eq. 11)
//! σ_median = median(σᵢ) / (0.6745 · √(N_t))    (scaled noise estimate)
//! k = max{n : σₙ > τ}
//! ```
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
//!
//! # References
//!
//! - Demené, C., et al. (2015). Spatiotemporal Clutter Filtering of Ultrafast Ultrasound
//!   Data Highly Increases Doppler and fUS Sensitivity.
//!   *IEEE Trans. Med. Imaging* 34(11):2271–2285. DOI: 10.1109/TMI.2015.2428634
//! - Errico, C., et al. (2015). Ultrafast ultrasound localization microscopy for deep
//!   super-resolution vascular imaging. *Nature* 527:499–502. DOI: 10.1038/nature16066
//! - Gavish, M., & Donoho, D. L. (2014). The optimal hard threshold for singular values
//!   is 4/√3. *IEEE Trans. Inf. Theory* 60(8):5040–5053. DOI: 10.1109/TIT.2014.2323359
//! - Thompson, R. E., Larson, D. R., & Webb, W. W. (2002). Precise nanometer localization
//!   analysis for individual fluorescent probes.
//!   *Biophys. J.* 82(5):2775–2783. DOI: 10.1016/S0006-3495(02)75618-X

use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use crate::math::linear_algebra::LinearAlgebra;
use ndarray::{s, Array1, Array2};

// ─── Data types ──────────────────────────────────────────────────────────────

/// One detected and localized microbubble in a single frame.
#[derive(Debug, Clone, PartialEq)]
pub struct BubbleDetection {
    /// Sub-pixel lateral position [pixels or m, same units as input grid]
    pub x: f64,
    /// Sub-pixel axial position [pixels or m]
    pub z: f64,
    /// Fitted Gaussian amplitude [a.u.]
    pub amplitude: f64,
    /// Fitted Gaussian width σ [same units as x/z]
    pub sigma: f64,
    /// Background level [a.u.]
    pub background: f64,
    /// Frame index
    pub frame: usize,
}

/// Configuration for SVD clutter filtering.
#[derive(Debug, Clone)]
pub struct SvdClutterConfig {
    /// Override automatic SVHT threshold with fixed k (0 = automatic).
    pub fixed_clutter_rank: usize,
    /// Safety margin added to SVHT k (default 0).
    pub rank_margin: usize,
}

impl Default for SvdClutterConfig {
    fn default() -> Self {
        Self {
            fixed_clutter_rank: 0,
            rank_margin: 0,
        }
    }
}

/// Configuration for Gaussian localization.
#[derive(Debug, Clone)]
pub struct LocalizationConfig {
    /// Detection threshold: candidate_amplitude > threshold_sigma_multiplier × noise_std
    pub threshold_sigma_multiplier: f64,
    /// Minimum PSF width accepted [pixels]
    pub min_sigma_px: f64,
    /// Maximum PSF width accepted [pixels]
    pub max_sigma_px: f64,
    /// Minimum amplitude-to-background ratio for acceptance
    pub min_snr_ratio: f64,
    /// Half-side of the local neighbourhood used for Gaussian fit (default 2 → 5×5)
    pub fit_half_width: usize,
    /// Maximum Gauss-Newton iterations
    pub max_gauss_newton_iter: usize,
}

impl Default for LocalizationConfig {
    fn default() -> Self {
        Self {
            threshold_sigma_multiplier: 3.0,
            min_sigma_px: 0.3,
            max_sigma_px: 3.0,
            min_snr_ratio: 2.0,
            fit_half_width: 2,
            max_gauss_newton_iter: 20,
        }
    }
}

// ─── SVD Clutter Filter ───────────────────────────────────────────────────────

/// SVD spatiotemporal clutter filter.
///
/// Separates microbubble signal from tissue clutter by projecting out the
/// low-rank tissue subspace in the IQ data matrix.
#[derive(Debug)]
pub struct SvdClutterFilter {
    config: SvdClutterConfig,
}

impl SvdClutterFilter {
    #[must_use]
    pub fn new(config: SvdClutterConfig) -> Self {
        Self { config }
    }

    /// Filter IQ data matrix S \[N_px × N_t\] → bubble-only matrix B \[N_px × N_t\].
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. [U, Σ, Vᵀ] = SVD(S)
    /// 2. τ = SVHT(β, Σ);  k = max{n : σₙ > τ} + rank_margin
    /// 3. T = U[:,0:k] · diag(Σ[0:k]) · Vᵀ[0:k,:]
    /// 4. B = S − T
    /// ```
    ///
    /// Returns `(B, k)` where k is the tissue rank used.
    pub fn filter(&self, iq_data: &Array2<f64>) -> KwaversResult<(Array2<f64>, usize)> {
        let (n_px, n_t) = (iq_data.nrows(), iq_data.ncols());
        if n_px == 0 || n_t == 0 {
            return Err(KwaversError::Numerical(NumericalError::SolverFailed {
                method: "SVD clutter filter".to_string(),
                reason: "empty IQ data matrix".to_string(),
            }));
        }

        let (u, sigma, vt) = LinearAlgebra::svd(iq_data)?;

        let k = if self.config.fixed_clutter_rank > 0 {
            self.config.fixed_clutter_rank.min(sigma.len())
        } else {
            let k_svht = svht_threshold(&sigma, n_px, n_t);
            (k_svht + self.config.rank_margin).min(sigma.len())
        };

        if k == 0 {
            // No clutter detected — return original
            return Ok((iq_data.clone(), 0));
        }

        // Reconstruct tissue: T = U[:,0:k] · diag(Σ[0:k]) · Vᵀ[0:k,:]
        //
        // Note: LinearAlgebra::svd returns (U, Σ, V) where V = V_nalgebra^T
        // (i.e. the returned third matrix has columns = right singular vectors).
        // Reconstruction requires V_k^T = first k rows of V^T = first k columns of V,
        // transposed.  So: T = U_k · diag(Σ_k) · V_k^T  where V_k = vt[:, 0..k].
        let tissue = {
            let u_k = u.slice(s![.., 0..k]).to_owned(); // (n_px, k)
            let sigma_k = sigma.slice(s![0..k]).to_owned(); // (k,)
            let v_k = vt.slice(s![.., 0..k]).to_owned(); // (n_t, k) — first k cols of V
            let vt_k = v_k.t().to_owned(); // (k, n_t) — V_k^T
            // U_k · diag(Σ_k): scale columns of U_k
            let mut us = u_k.clone();
            for (j, &s_j) in sigma_k.iter().enumerate() {
                for i in 0..n_px {
                    us[[i, j]] *= s_j;
                }
            }
            // (U_k · Σ_k) · V^T_k
            let mut result = Array2::<f64>::zeros((n_px, n_t));
            ndarray::linalg::general_mat_mul(1.0, &us, &vt_k, 0.0, &mut result);
            result
        };

        let bubble = iq_data - &tissue;
        Ok((bubble, k))
    }
}

/// Singular Value Hard Threshold (Gavish & Donoho 2014).
///
/// # Theorem (Gavish & Donoho 2014, Theorem 3 + Supplementary S5)
///
/// For an n×m matrix (n ≤ m) corrupted by iid Gaussian noise with unknown variance σ²,
/// β = n/m, the optimal hard threshold (unknown-σ case) is:
/// ```text
/// τ = ω(β) · s_med / μ_MP(β)
/// ```
/// where:
/// - `ω(β) = 0.56·β³ − 0.95·β² + 1.82·β + 1.43`  (SVHT coefficient, eq. 11)
/// - `s_med` = median singular value
/// - `μ_MP(β)` = theoretical median of the Marchenko-Pastur distribution with parameter β
///
/// This formulation follows from the noise estimate:
/// `σ̂ = s_med / (μ_MP(β) · √m)`,
/// giving `τ = ω(β) · σ̂ · √m = ω(β) · s_med / μ_MP(β)`.
///
/// Returns the number of singular values that exceed the threshold (tissue rank k).
fn svht_threshold(sigma: &Array1<f64>, n_rows: usize, n_cols: usize) -> usize {
    let (n, m) = if n_rows <= n_cols {
        (n_rows, n_cols)
    } else {
        (n_cols, n_rows)
    };
    let beta = n as f64 / m as f64;

    // ω(β) polynomial approximation (Gavish & Donoho 2014, eq. 11)
    let omega = 0.56 * beta.powi(3) - 0.95 * beta.powi(2) + 1.82 * beta + 1.43;

    // Median singular value
    let mut s_sorted: Vec<f64> = sigma.iter().copied().collect();
    s_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let s_med = if s_sorted.is_empty() {
        return 0;
    } else if s_sorted.len() % 2 == 0 {
        (s_sorted[s_sorted.len() / 2 - 1] + s_sorted[s_sorted.len() / 2]) / 2.0
    } else {
        s_sorted[s_sorted.len() / 2]
    };

    // Marchenko-Pastur median: τ = ω · s_med / μ_MP(β)
    let mu_mp = mp_median(beta);
    let tau = omega * s_med / mu_mp;

    sigma.iter().filter(|&&sv| sv > tau).count()
}

/// Numerical median of the Marchenko-Pastur distribution with parameter β ∈ (0, 1].
///
/// # Theorem
///
/// The Marchenko-Pastur distribution with parameter β ∈ (0,1] has support
/// [λ₋, λ₊] = [(1−√β)², (1+√β)²] and PDF:
/// ```text
/// f(x) = √((λ₊−x)(x−λ₋)) / (2π·β·x)
/// ```
///
/// # Algorithm
///
/// To avoid endpoint singularities in numerical integration, use the substitution
/// `x = λ₋ + (λ₊−λ₋)·sin²(φ)` for φ ∈ [0, π/2], giving:
/// ```text
/// f(x)·dx/dφ = (λ₊−λ₋)²·sin²(2φ) / (4π·β·x(φ))
/// ```
/// Then bisect on the CDF to find the median.
fn mp_median(beta: f64) -> f64 {
    let lb = (1.0_f64 - beta.sqrt()).powi(2);
    let ub = (1.0_f64 + beta.sqrt()).powi(2);
    let range = ub - lb;

    // CDF from lb to x using sin-substitution: x = lb + range·sin²(φ)
    // Upper limit φ_max = arcsin(√((x−lb)/range))
    let cdf = |x: f64| -> f64 {
        if x <= lb {
            return 0.0;
        }
        if x >= ub {
            return 1.0;
        }
        let phi_max = ((x - lb) / range).sqrt().asin();
        let n = 400usize;
        let dphi = phi_max / n as f64;
        let mut sum = 0.0;
        for k in 0..n {
            let phi = (k as f64 + 0.5) * dphi;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();
            let xv = lb + range * sin_phi * sin_phi;
            // f(x)·(dx/dφ) = range² · 2·sin(φ)·cos(φ) · range·2·sin(φ)·cos(φ) / (2π·β·xv)
            // = range² · 4·sin²(φ)·cos²(φ) / (2π·β·xv) · dphi
            // Simplified: range² · sin²(2φ) / (2π·β·xv) · dphi
            let sin2 = 2.0 * sin_phi * cos_phi; // sin(2φ)
            let integrand =
                range * range * sin2 * sin2 / (2.0 * std::f64::consts::PI * beta * xv);
            sum += integrand * dphi;
        }
        sum
    };

    // Normalize: cdf_total = ∫_{lb}^{ub} f(x) dx ≈ 1 (verify and normalize)
    let cdf_total = cdf(ub - 1e-10);

    // Bisection for x such that cdf(x)/cdf_total = 0.5
    let mut lo = lb;
    let mut hi = ub;
    for _ in 0..52 {
        let mid = (lo + hi) / 2.0;
        if cdf(mid) / cdf_total < 0.5 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

// ─── Gaussian Sub-pixel Localization ─────────────────────────────────────────

/// Sub-pixel microbubble localizer using Gauss-Newton least squares.
#[derive(Debug)]
pub struct GaussianLocalizer {
    config: LocalizationConfig,
}

impl GaussianLocalizer {
    #[must_use]
    pub fn new(config: LocalizationConfig) -> Self {
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
    pub fn localize_frame(
        &self,
        envelope: &Array2<f64>,
        frame_idx: usize,
    ) -> KwaversResult<Vec<BubbleDetection>> {
        let (n_z, n_x) = (envelope.nrows(), envelope.ncols());
        if n_z < 3 || n_x < 3 {
            return Ok(Vec::new());
        }

        // Noise estimate: median absolute deviation → σ_noise = MAD / 0.6745
        let noise_std = noise_std_estimate(envelope);
        let threshold = self.config.threshold_sigma_multiplier * noise_std;

        // Find local maxima above threshold
        let mut detections = Vec::new();
        let hw = self.config.fit_half_width;

        for iz in 1..(n_z - 1) {
            for ix in 1..(n_x - 1) {
                let v = envelope[[iz, ix]];
                if v < threshold {
                    continue;
                }
                // Local maximum test (3×3 neighbourhood)
                if !is_local_max(envelope, iz, ix) {
                    continue;
                }
                // Extract fit patch (clamped to array bounds)
                let iz0 = iz.saturating_sub(hw);
                let iz1 = (iz + hw + 1).min(n_z);
                let ix0 = ix.saturating_sub(hw);
                let ix1 = (ix + hw + 1).min(n_x);
                let patch = envelope.slice(s![iz0..iz1, ix0..ix1]).to_owned();

                match gauss_newton_fit_2d(
                    &patch,
                    iz as f64 - iz0 as f64, // initial z-center in patch coords
                    ix as f64 - ix0 as f64, // initial x-center in patch coords
                    self.config.max_gauss_newton_iter,
                ) {
                    Some((z_sub, x_sub, amp, sigma, bg)) => {
                        // Quality gates
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
                    None => {}
                }
            }
        }

        Ok(detections)
    }
}

/// Noise standard deviation estimate via median absolute deviation.
///
/// σ̂ = MAD(envelope) / 0.6745   (consistent for Gaussian noise)
fn noise_std_estimate(a: &Array2<f64>) -> f64 {
    let mut vals: Vec<f64> = a.iter().copied().collect();
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let med = if vals.len() % 2 == 0 {
        (vals[vals.len() / 2 - 1] + vals[vals.len() / 2]) / 2.0
    } else {
        vals[vals.len() / 2]
    };
    let mut devs: Vec<f64> = vals.iter().map(|&x| (x - med).abs()).collect();
    devs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mad = if devs.len() % 2 == 0 {
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
fn gauss_newton_fit_2d(
    patch: &Array2<f64>,
    z0_init: f64,
    x0_init: f64,
    max_iter: usize,
) -> Option<(f64, f64, f64, f64, f64)> {
    let (nz, nx) = (patch.nrows(), patch.ncols());
    let n = nz * nx;

    // Initial parameter estimates
    let bg_init = patch.iter().copied().fold(f64::MAX, f64::min);
    let amp_init = patch.iter().copied().fold(f64::MIN, f64::max) - bg_init;
    if amp_init <= 0.0 {
        return None;
    }

    let mut z0 = z0_init;
    let mut x0 = x0_init;
    let mut amp = amp_init.max(1e-30);
    let mut sigma = 1.0_f64; // initial guess: 1 pixel
    let mut bg = bg_init;

    for _iter in 0..max_iter {
        // Residuals and Jacobian (θ = [amp, z0, x0, sigma, bg])
        let mut jt_j = [[0.0_f64; 5]; 5];
        let mut jt_r = [0.0_f64; 5];
        let mut rss_prev = 0.0_f64;

        for iz in 0..nz {
            for ix in 0..nx {
                let z = iz as f64;
                let x = ix as f64;
                let dz = z - z0;
                let dx = x - x0;
                let r2 = (dz * dz + dx * dx) / (2.0 * sigma * sigma);
                let g = (-r2).exp();
                let model = amp * g + bg;
                let resid = patch[[iz, ix]] - model;
                rss_prev += resid * resid;

                // Partial derivatives: ∂f/∂θ
                let df_amp = g;
                let df_z0 = amp * g * dz / (sigma * sigma);
                let df_x0 = amp * g * dx / (sigma * sigma);
                let df_sigma = amp * g * (dz * dz + dx * dx) / (sigma * sigma * sigma);
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

        // Solve 5×5 normal equations (JᵀJ) δ = Jᵀr via Gaussian elimination
        let delta = solve_5x5(&jt_j, &jt_r)?;

        amp += delta[0];
        z0 += delta[1];
        x0 += delta[2];
        sigma += delta[3];
        bg += delta[4];

        // Clamp to keep parameters physical
        amp = amp.max(1e-30);
        sigma = sigma.clamp(0.1, (nz.max(nx)) as f64);
        z0 = z0.clamp(0.0, (nz - 1) as f64);
        x0 = x0.clamp(0.0, (nx - 1) as f64);

        let step_sq: f64 = delta.iter().map(|&d| d * d).sum();
        if step_sq < 1e-12 * n as f64 {
            break; // Converged
        }
    }

    Some((z0, x0, amp, sigma, bg))
}

/// Gaussian elimination solver for a 5×5 system Aδ = b.
///
/// Returns `None` if the matrix is singular (pivot < 1e-12).
fn solve_5x5(a: &[[f64; 5]; 5], b: &[f64; 5]) -> Option<[f64; 5]> {
    let mut m = [[0.0_f64; 6]; 5];
    for i in 0..5 {
        for j in 0..5 {
            m[i][j] = a[i][j];
        }
        m[i][5] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..5 {
        // Find pivot
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

    // Back substitution
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

// ─── Combined pipeline ────────────────────────────────────────────────────────

/// Full ULM detection pipeline: SVD filtering + Gaussian localization.
#[derive(Debug)]
pub struct UlmDetector {
    clutter_filter: SvdClutterFilter,
    localizer: GaussianLocalizer,
}

impl UlmDetector {
    #[must_use]
    pub fn new(clutter_cfg: SvdClutterConfig, loc_cfg: LocalizationConfig) -> Self {
        Self {
            clutter_filter: SvdClutterFilter::new(clutter_cfg),
            localizer: GaussianLocalizer::new(loc_cfg),
        }
    }

    /// Process a block of IQ frames and return all bubble detections.
    ///
    /// # Arguments
    /// * `iq_block` — IQ matrix \[N_px × N_t\] (pixels × frames, linearized 2D→1D)
    /// * `n_z` — number of axial pixels (N_px = n_z × n_x)
    /// * `n_x` — number of lateral pixels
    pub fn process_block(
        &self,
        iq_block: &Array2<f64>,
        n_z: usize,
        n_x: usize,
    ) -> KwaversResult<Vec<BubbleDetection>> {
        let (bubble_data, _k) = self.clutter_filter.filter(iq_block)?;
        let n_t = bubble_data.ncols();
        let mut all_detections = Vec::new();

        for t in 0..n_t {
            let frame_col = bubble_data.column(t);
            // Reshape linearized pixel vector to 2D envelope
            if frame_col.len() != n_z * n_x {
                return Err(KwaversError::Numerical(NumericalError::SolverFailed {
                    method: "ULM detect".to_string(),
                    reason: format!(
                        "pixel count {} ≠ n_z×n_x = {}×{}={}",
                        frame_col.len(),
                        n_z,
                        n_x,
                        n_z * n_x
                    ),
                }));
            }
            let envelope = Array2::from_shape_vec(
                (n_z, n_x),
                frame_col.iter().map(|v| v.abs()).collect(),
            )
            .map_err(|e| {
                KwaversError::Numerical(NumericalError::SolverFailed {
                    method: "ULM reshape".to_string(),
                    reason: e.to_string(),
                })
            })?;

            let frame_dets = self.localizer.localize_frame(&envelope, t)?;
            all_detections.extend(frame_dets);
        }

        Ok(all_detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Generate a deterministic pseudo-noise matrix using a simple LCG for portability.
    fn make_noise_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut state = seed;
        Array2::from_shape_fn((rows, cols), |_| {
            // LCG: xₙ₊₁ = (a·xₙ + c) mod m
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map [0, 2^64) to approximately N(0,1) via Box-Muller
            let u1 = (state >> 11) as f64 / (1u64 << 53) as f64 + 1e-30;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
    }

    #[test]
    fn test_svht_noise_only() {
        // For a pure-noise matrix, SVHT should give k ≈ 0 (all singular values below threshold).
        // Gavish & Donoho (2014) Theorem 3 guarantees this for Gaussian noise.
        let n_px = 50usize;
        let n_t = 100usize;
        let noise = make_noise_matrix(n_px, n_t, 42);
        let (_u, sigma, _vt) = LinearAlgebra::svd(&noise).unwrap();

        let k = svht_threshold(&sigma, n_px, n_t);
        // For Gaussian noise, the SVHT should retain very few components
        assert!(
            k <= 5,
            "SVHT on noise should give k≈0, got k={k}"
        );
    }

    #[test]
    fn test_svd_clutter_filter_rank1_tissue() {
        // Create a rank-1 tissue matrix + point source bubble
        let n_px = 20;
        let n_t = 30;

        // Tissue: outer product of fixed spatial pattern × slow temporal trend
        let tissue_spatial: Vec<f64> = (0..n_px).map(|i| (i as f64 * 0.1).sin()).collect();
        let tissue_temporal: Vec<f64> = (0..n_t).map(|t| (t as f64 * 0.05).cos() * 100.0).collect();

        let mut iq = Array2::<f64>::zeros((n_px, n_t));
        for i in 0..n_px {
            for t in 0..n_t {
                iq[[i, t]] = tissue_spatial[i] * tissue_temporal[t];
            }
        }
        // Add a transient bubble at pixel 10, frames 5-8
        for t in 5..8 {
            iq[[10, t]] += 5.0;
        }

        let cfg = SvdClutterConfig {
            fixed_clutter_rank: 1,
            ..Default::default()
        };
        let filter = SvdClutterFilter::new(cfg);
        let (bubble, k) = filter.filter(&iq).unwrap();

        assert_eq!(k, 1, "Tissue rank should be 1");
        // The bubble pixel energy should be higher in bubble signal than in tissue frames
        let bubble_energy: f64 = (5..8).map(|t| bubble[[10, t]].powi(2)).sum();
        let noise_energy: f64 = (15..20).map(|t| bubble[[10, t]].powi(2)).sum();
        assert!(
            bubble_energy > noise_energy * 5.0,
            "Bubble frames energy {bubble_energy:.3} should exceed background {noise_energy:.3}"
        );
    }

    #[test]
    fn test_gaussian_fit_synthetic_peak() {
        // Synthetic 2D Gaussian peak: center should be recovered within 0.1 px
        let (nz, nx) = (11usize, 11usize);
        let (true_z, true_x) = (5.3_f64, 5.7_f64); // sub-pixel offset
        let amp = 10.0_f64;
        let sigma = 1.2_f64;
        let bg = 0.5_f64;

        let mut envelope = Array2::<f64>::zeros((nz, nx));
        for iz in 0..nz {
            for ix in 0..nx {
                let dz = iz as f64 - true_z;
                let dx = ix as f64 - true_x;
                envelope[[iz, ix]] = amp * (-(dz * dz + dx * dx) / (2.0 * sigma * sigma)).exp() + bg;
            }
        }

        let result = gauss_newton_fit_2d(&envelope, true_z.round(), true_x.round(), 50);
        assert!(result.is_some(), "Gauss-Newton fit should converge");
        let (z_fit, x_fit, amp_fit, sigma_fit, _bg_fit) = result.unwrap();

        assert!(
            (z_fit - true_z).abs() < 0.1,
            "z center error {:.4} > 0.1 px",
            (z_fit - true_z).abs()
        );
        assert!(
            (x_fit - true_x).abs() < 0.1,
            "x center error {:.4} > 0.1 px",
            (x_fit - true_x).abs()
        );
        assert!(
            (amp_fit - amp).abs() / amp < 0.05,
            "Amplitude error {:.3}",
            (amp_fit - amp).abs() / amp
        );
        assert!(
            (sigma_fit - sigma).abs() / sigma < 0.05,
            "Sigma error {:.3}",
            (sigma_fit - sigma).abs() / sigma
        );
    }

    #[test]
    fn test_localizer_detects_isolated_peak() {
        // Single strong Gaussian bubble on quiet background
        let (nz, nx) = (20, 20);
        let (tz, tx) = (10.4, 10.6);
        let mut envelope = Array2::<f64>::zeros((nz, nx));
        for iz in 0..nz {
            for ix in 0..nx {
                let dz = iz as f64 - tz;
                let dx = ix as f64 - tx;
                envelope[[iz, ix]] = 8.0 * (-(dz * dz + dx * dx) / 2.0).exp() + 0.1;
            }
        }

        let cfg = LocalizationConfig {
            min_sigma_px: 0.5,
            max_sigma_px: 3.0,
            ..Default::default()
        };
        let localizer = GaussianLocalizer::new(cfg);
        let detections = localizer.localize_frame(&envelope, 0).unwrap();

        assert_eq!(detections.len(), 1, "Should detect exactly one bubble");
        let d = &detections[0];
        assert!(
            (d.z - tz).abs() < 0.2,
            "z localization error {:.3}",
            (d.z - tz).abs()
        );
        assert!(
            (d.x - tx).abs() < 0.2,
            "x localization error {:.3}",
            (d.x - tx).abs()
        );
    }
}
