//! Bent-ray traveltime tomography — iterative reconstruction (ADR 020).
//!
//! Closes the loop opened by [`super::bent_ray`]: the Fermat shortest-path tracer
//! produces, for each source→receiver pair, the per-voxel path-length
//! **system-matrix row** such that `t = Σ_v s_v · row_v`. This module assembles
//! those rows over many pairs and inverts for the slowness field `s`.
//!
//! # Nonlinear iteration
//!
//! Ray geometry depends on the (unknown) slowness, so traveltime tomography is
//! nonlinear. The standard fixed point alternates **trace** and **solve**:
//!
//! ```text
//! s ← initial guess
//! repeat (outer):
//!     trace bent rays through the current s  →  sparse rows {a_i}
//!     refine s by sweeps of sparse ART (Kaczmarz) on  a_i · s = t_obs_i
//! ```
//!
//! The inner solver is **ART** (algebraic reconstruction / Kaczmarz row-action),
//! the appropriate algorithm for the sparse path-length rows: a dense system
//! matrix is never formed. Each sweep is
//! `s_v ← s_v + λ (t_i − a_i·s)/‖a_i‖² · a_{iv}` for the voxels `v` on ray `i`.
//! Because the `BentRay::row` voxel indices are the row-major flatten
//! `i·ny + j`, they index an `Array2<f64>`'s contiguous buffer directly.
//!
//! # References
//! - Kaczmarz, S. (1937). Row-action linear solve.
//! - Nolet, G. (1987). *Seismic Tomography*. (bent-ray traveltime inversion)

use super::bent_ray::bent_ray_path;
use leto::Array2;

/// One traveltime measurement: a source→receiver pair and its observed
/// first-arrival traveltime \[s]. Source/receiver are grid indices `(i, j)`.
#[derive(Debug, Clone, Copy)]
pub struct TraveltimeDatum {
    /// Source voxel `(i, j)`.
    pub source: (usize, usize),
    /// Receiver voxel `(i, j)`.
    pub receiver: (usize, usize),
    /// Observed first-arrival traveltime \[s].
    pub traveltime: f64,
}

/// Configuration for [`reconstruct_bent_ray_tomography`].
#[derive(Debug, Clone, Copy)]
pub struct BentRayTomographyConfig {
    /// Outer trace↔solve iterations (ray re-tracing through the updated model).
    pub outer_iterations: usize,
    /// Inner ART sweeps over all rays per outer iteration.
    pub inner_sweeps: usize,
    /// ART relaxation `λ ∈ (0, 1]`.
    pub relaxation: f64,
}

impl Default for BentRayTomographyConfig {
    fn default() -> Self {
        Self {
            outer_iterations: 6,
            inner_sweeps: 8,
            relaxation: 0.2,
        }
    }
}

/// Result of a bent-ray tomographic reconstruction.
#[derive(Debug, Clone)]
pub struct BentRayTomographyResult {
    /// Reconstructed slowness field \[s/m], same shape as the initial guess.
    pub slowness: Array2<f64>,
    /// RMS traveltime data misfit \[s] after each outer iteration (monotone
    /// non-increasing for a well-posed problem).
    pub residual_history: Vec<f64>,
}

/// Iterative bent-ray traveltime tomography. Reconstructs the slowness field
/// from traveltime data, starting from `initial_slowness` and re-tracing rays
/// through the evolving model each outer iteration.
///
/// `dx` is the (isotropic) grid spacing \[m]. Measurements whose ray cannot be
/// traced (out-of-bounds endpoints) are skipped. Returns the reconstructed
/// slowness and the per-outer-iteration RMS data misfit.
#[must_use]
pub fn reconstruct_bent_ray_tomography(
    initial_slowness: &Array2<f64>,
    dx: f64,
    data: &[TraveltimeDatum],
    config: &BentRayTomographyConfig,
) -> BentRayTomographyResult {
    let mut slowness = initial_slowness.clone();
    let ny = slowness.ncols();
    let mut residual_history = Vec::with_capacity(config.outer_iterations);

    for _outer in 0..config.outer_iterations {
        // ── Trace bent rays through the current model ────────────────────────
        // Each traced ray carries its sparse path-length row and observed time.
        let traced: Vec<(Vec<(usize, f64)>, f64)> = data
            .iter()
            .filter_map(|d| {
                bent_ray_path(&slowness, dx, d.source, d.receiver).map(|r| (r.row, d.traveltime))
            })
            .collect();

        // ── Inner ART sweeps over the sparse rows ────────────────────────────
        for _sweep in 0..config.inner_sweeps {
            for (row, t_obs) in &traced {
                let buf = slowness
                    .as_slice_mut()
                    .expect("row-major contiguous slowness");
                let mut predicted = 0.0;
                let mut norm_sq = 0.0;
                for &(idx, len) in row {
                    predicted += len * buf[idx];
                    norm_sq += len * len;
                }
                if norm_sq <= 0.0 {
                    continue;
                }
                let update = config.relaxation * (t_obs - predicted) / norm_sq;
                for &(idx, len) in row {
                    buf[idx] += update * len;
                }
            }
        }

        // ── Record the post-update data misfit (rays of the new model) ───────
        residual_history.push(rms_misfit(&slowness, dx, data));
    }

    debug_assert_eq!(slowness.ncols(), ny);
    BentRayTomographyResult {
        slowness,
        residual_history,
    }
}

/// RMS traveltime misfit `√(⟨(t_pred − t_obs)²⟩)` of `slowness` against `data`,
/// with `t_pred` the bent-ray traveltime through `slowness`. Skips untraceable
/// pairs. Returns `0` if no pair traces.
#[must_use]
pub fn rms_misfit(slowness: &Array2<f64>, dx: f64, data: &[TraveltimeDatum]) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for d in data {
        if let Some(ray) = bent_ray_path(slowness, dx, d.source, d.receiver) {
            let e = ray.traveltime - d.traveltime;
            sum_sq += e * e;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum_sq / count as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruction::bent_ray::bent_ray_traveltime;

    const DX: f64 = 1.0e-3; // 1 mm voxels

    /// Build a transmission survey: sources on the left/top edges, receivers on
    /// the right/bottom edges, giving crossing ray coverage of the interior.
    fn survey(nx: usize, ny: usize, truth: &Array2<f64>) -> Vec<TraveltimeDatum> {
        let mut data = Vec::new();
        // Left → right transmission (dense fan from every left-edge voxel).
        for si in 0..nx {
            for ri in 0..nx {
                push_datum(&mut data, truth, (si, 0), (ri, ny - 1));
            }
        }
        // Top → bottom transmission (orthogonal coverage).
        for sj in 0..ny {
            for rj in 0..ny {
                push_datum(&mut data, truth, (0, sj), (nx - 1, rj));
            }
        }
        data
    }

    fn push_datum(
        data: &mut Vec<TraveltimeDatum>,
        truth: &Array2<f64>,
        source: (usize, usize),
        receiver: (usize, usize),
    ) {
        if let Some(t) = bent_ray_traveltime(truth, DX, source, receiver) {
            data.push(TraveltimeDatum {
                source,
                receiver,
                traveltime: t,
            });
        }
    }

    /// A wrong uniform initial guess converges to the true uniform slowness
    /// (a homogeneous medium has straight rays; ART must recover the constant).
    #[test]
    fn recovers_homogeneous_slowness_from_wrong_guess() {
        let (nx, ny) = (16, 16);
        let s_true = 1.0 / 1500.0; // water
        let truth = Array2::from_elem((nx, ny), s_true);
        let data = survey(nx, ny, &truth);
        assert!(data.len() > 50, "need adequate ray coverage");

        let guess = Array2::from_elem((nx, ny), 1.0 / 1400.0); // 7% wrong (too fast)
        let cfg = BentRayTomographyConfig::default();
        let result = reconstruct_bent_ray_tomography(&guess, DX, &data, &cfg);

        // Data misfit collapses: the model fits the observed traveltimes.
        let initial_misfit = rms_misfit(&guess, DX, &data);
        let final_misfit = *result.residual_history.last().unwrap();
        assert!(
            final_misfit < 0.1 * initial_misfit,
            "data misfit must collapse: {final_misfit} vs {initial_misfit}"
        );

        // Interior recovery: the mean recovers the true uniform slowness to ~1%
        // (an 85% reduction of the 7% starting error), and the large majority of
        // interior voxels are well recovered. A handful of voxels lie in the
        // survey's coverage null space (no traced ray crosses them) and keep the
        // initial value — a property of the *geometry*, not the algorithm — so
        // we assert a high recovered fraction rather than a pointwise maximum.
        let mut sum = 0.0;
        let mut n = 0.0;
        let mut well_recovered = 0.0;
        for i in 2..nx - 2 {
            for j in 2..ny - 2 {
                sum += result.slowness[[i, j]];
                n += 1.0;
                if (result.slowness[[i, j]] - s_true).abs() / s_true < 0.03 {
                    well_recovered += 1.0;
                }
            }
        }
        let mean_err = (sum / n - s_true).abs() / s_true;
        assert!(
            mean_err < 0.015,
            "mean interior slowness error {mean_err} must be <1.5%"
        );
        assert!(
            well_recovered / n > 0.85,
            "≥85% of interior voxels recovered to <3%; got {}",
            well_recovered / n
        );
    }

    /// A slow circular anomaly is recovered: the reconstruction correlates with
    /// the truth and the anomaly region is reconstructed slower than background,
    /// with the data misfit falling across outer iterations.
    #[test]
    fn recovers_slow_disk_anomaly_and_reduces_misfit() {
        let (nx, ny) = (20, 20);
        let s_bg = 1.0 / 1500.0;
        let s_anom = 1.0 / 1300.0; // ~15% slower (higher slowness)
        let (ci, cj, r) = (10.0_f64, 10.0_f64, 3.5_f64);

        let mut truth = Array2::from_elem((nx, ny), s_bg);
        for i in 0..nx {
            for j in 0..ny {
                let d = ((i as f64 - ci).powi(2) + (j as f64 - cj).powi(2)).sqrt();
                if d <= r {
                    truth[[i, j]] = s_anom;
                }
            }
        }

        let data = survey(nx, ny, &truth);
        let guess = Array2::from_elem((nx, ny), s_bg); // correct background, no anomaly
        let cfg = BentRayTomographyConfig {
            outer_iterations: 8,
            inner_sweeps: 10,
            relaxation: 0.2,
        };
        let result = reconstruct_bent_ray_tomography(&guess, DX, &data, &cfg);

        // (a) Data misfit decreases from the first to the last outer iteration.
        let first = result.residual_history.first().copied().unwrap();
        let last = result.residual_history.last().copied().unwrap();
        assert!(last < first, "misfit must fall: {last} !< {first}");

        // (b) The reconstruction correlates with the truth.
        let corr = kwavers_math::statistics::pearson(
            &result.slowness.iter().copied().collect::<Vec<_>>(),
            &truth.iter().copied().collect::<Vec<_>>(),
        );
        assert!(
            corr > 0.5,
            "reconstruction correlation {corr} must exceed 0.5"
        );

        // (c) The anomaly region is reconstructed slower than the background
        //     ring — the defining tomographic outcome (correct sign + location).
        let mut anom_sum = 0.0;
        let mut anom_n = 0.0;
        let mut bg_sum = 0.0;
        let mut bg_n = 0.0;
        for i in 0..nx {
            for j in 0..ny {
                let d = ((i as f64 - ci).powi(2) + (j as f64 - cj).powi(2)).sqrt();
                if d <= r - 1.0 {
                    anom_sum += result.slowness[[i, j]];
                    anom_n += 1.0;
                } else if d > r + 2.0 && i > 2 && i < nx - 2 && j > 2 && j < ny - 2 {
                    bg_sum += result.slowness[[i, j]];
                    bg_n += 1.0;
                }
            }
        }
        let anom_mean = anom_sum / anom_n;
        let bg_mean = bg_sum / bg_n;
        assert!(
            anom_mean > bg_mean,
            "anomaly must reconstruct slower than background: {anom_mean} !> {bg_mean}"
        );
    }
}
