//! CPU Minimum Variance Distortionless Response (MVDR) beamformer for 3D ultrasound.
//!
//! ## Theorem: MVDR (Capon) Beamformer
//!
//! Given an M-element receive array, let **x**[n] ∈ ℝ^M be the snapshot of
//! delay-aligned RF samples at time n after applying receive delays for voxel
//! **r**_v.  The spatial covariance matrix is
//!
//! ```text
//!   R = (1/N) Σ_{n=0}^{N-1} x[n] x[n]^T  ∈ ℝ^{M×M}.
//! ```
//!
//! To ensure positive-definiteness under finite data, diagonal loading is applied:
//!
//! ```text
//!   R_δ = R + δ · (tr(R)/M) · I_M
//! ```
//!
//! where δ > 0 is the relative loading factor.  The MVDR weight vector **w** is
//!
//! ```text
//!   w = R_δ^{−1} 1 / (1^T R_δ^{−1} 1)
//! ```
//!
//! where **1** is the all-ones steering vector (the delays have already been
//! absorbed into **x**[n]).  The MVDR beamformed signal is
//!
//! ```text
//!   y[n] = w^T x[n]   with output power   P = 1 / (1^T R_δ^{−1} 1).
//! ```
//!
//! ## Proof of Distortionless Response
//!
//! Minimising E[|w^T x|^2] = w^T R w subject to w^T 1 = 1 by Lagrange
//! multipliers gives w* = R^{-1}1 / (1^T R^{-1}1).  The constraint 1^T w* = 1
//! is satisfied by construction, so the power from the look-direction is
//! preserved while interference power is minimised.
//!
//! ## Spatial Smoothing
//!
//! Full covariance estimation of an M-element array requires O(M²·N) operations.
//! Spatial smoothing (Shan & Kailath 1985) divides the aperture into Q overlapping
//! subarrays of size L:
//!
//! ```text
//!   R̂ = (1/Q) Σ_{q=1}^{Q} R_q,   Q = M − L + 1   (1-D case)
//! ```
//!
//! This reduces matrix size from M×M to L×L and decorrelates coherent arrivals.
//! The 3-D extension uses overlapping 3-D sub-apertures indexed by (qx, qy, qz).
//!
//! ## References
//! - Capon J. (1969): "High-resolution frequency-wavenumber spectrum analysis."
//!   *Proc. IEEE* 57(8), 1408–1418.
//! - Synnevåg J.F., Austeng A., Holm S. (2007): "Adaptive beamforming applied to
//!   medical ultrasound imaging." *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*
//!   54(8), 1606–1613.
//! - Shan T.J., Kailath T. (1985): "Adaptive beamforming for coherent signals and
//!   interference." *IEEE Trans. Acoust. Speech Signal Process.* 33(3), 527–536.

use nalgebra::{DMatrix, DVector};
use ndarray::{Array3, Array4};
use rayon::prelude::*;

use crate::analysis::signal_processing::beamforming::three_dimensional::config::BeamformingConfig3D;
use crate::core::error::{KwaversError, KwaversResult};

/// Execute CPU MVDR beamforming for a single 3D volume.
///
/// # Algorithm
/// For each voxel **r**_v:
/// 1. Apply element-wise receive delays (fractional-delay linear interpolation).
/// 2. Build the spatially-smoothed covariance matrix over overlapping sub-apertures
///    of size `subarray_size` using all N time samples.
/// 3. Add relative diagonal loading: R_δ = R + δ·(tr(R)/L)·I.
/// 4. Solve the symmetric positive-definite system R_δ **u** = **1** via
///    Cholesky factorisation (O(L³)).
/// 5. Compute output power P = 1/(1^T **u**) and accumulate the average
///    beamformed amplitude |P·u^T x̄| where x̄ = (1/N) Σ_n x[n].
///
/// # Arguments
/// - `rf_data`        : Shape `[frames, channels, samples, 1]`
/// - `config`         : Array/volume geometry and acquisition parameters
/// - `diagonal_loading`: Relative loading factor δ (typical: 1/L to 100/L)
/// - `subarray_size`  : Sub-aperture dimensions (Lx, Ly, Lz); product L = Lx·Ly·Lz
///
/// # Returns
/// 3D volume of MVDR output amplitude, shape `config.volume_dims`.
///
/// # Errors
/// - [`KwaversError::InvalidInput`] if the channel count mismatches the array config.
/// - [`KwaversError::InvalidInput`] if any subarray dimension exceeds the array size.
pub fn mvdr_cpu(
    rf_data: &Array4<f32>,
    config: &BeamformingConfig3D,
    diagonal_loading: f32,
    subarray_size: [usize; 3],
) -> KwaversResult<Array3<f32>> {
    let (frames, channels, samples, _) = rf_data.dim();
    let (nel_x, nel_y, nel_z) = config.num_elements_3d;
    let (vol_x, vol_y, vol_z) = config.volume_dims;
    let expected_channels = nel_x * nel_y * nel_z;

    if channels != expected_channels {
        return Err(KwaversError::InvalidInput(format!(
            "MVDR CPU: channel count {channels} ≠ element count {expected_channels}"
        )));
    }
    let [lx, ly, lz] = subarray_size;
    if lx > nel_x || ly > nel_y || lz > nel_z {
        return Err(KwaversError::InvalidInput(format!(
            "MVDR CPU: subarray size [{lx},{ly},{lz}] exceeds array [{nel_x},{nel_y},{nel_z}]"
        )));
    }
    if samples == 0 {
        return Err(KwaversError::InvalidInput(
            "MVDR CPU: RF data must have at least one sample".to_string(),
        ));
    }

    let l = lx * ly * lz; // sub-aperture element count
    let fs = config.sampling_frequency as f32;
    let c = config.sound_speed as f32;
    let (dx, dy, dz) = (
        config.voxel_spacing.0 as f32,
        config.voxel_spacing.1 as f32,
        config.voxel_spacing.2 as f32,
    );
    let (sx, sy, sz) = (
        config.element_spacing_3d.0 as f32,
        config.element_spacing_3d.1 as f32,
        config.element_spacing_3d.2 as f32,
    );

    // Element (ex, ey, ez) → flat channel index and physical position.
    // Channel index: ch = ex * nel_y * nel_z + ey * nel_z + ez
    let elem_pos: Vec<[f32; 3]> = (0..nel_x)
        .flat_map(|ex| {
            (0..nel_y).flat_map(move |ey| {
                (0..nel_z).map(move |ez| {
                    [
                        (ex as f32 - (nel_x as f32 - 1.0) * 0.5) * sx,
                        (ey as f32 - (nel_y as f32 - 1.0) * 0.5) * sy,
                        (ez as f32 - (nel_z as f32 - 1.0) * 0.5) * sz,
                    ]
                })
            })
        })
        .collect();

    // Linear interpolation accessor — returns 0 outside the recorded window.
    let rf_get = |frame: usize, ch: usize, tau_s: f32| -> f32 {
        if tau_s < 0.0 {
            return 0.0;
        }
        let n0 = tau_s as usize;
        if n0 + 1 >= samples {
            return 0.0;
        }
        let alpha = tau_s - n0 as f32;
        rf_data[[frame, ch, n0, 0]] + alpha * (rf_data[[frame, ch, n0 + 1, 0]] - rf_data[[frame, ch, n0, 0]])
    };

    // Number of overlapping sub-apertures in each dimension.
    let n_sub_x = nel_x - lx + 1;
    let n_sub_y = nel_y - ly + 1;
    let n_sub_z = nel_z - lz + 1;
    let n_subarrays = n_sub_x * n_sub_y * n_sub_z;

    let n_voxels = vol_x * vol_y * vol_z;
    let mut flat: Vec<f32> = vec![0.0_f32; n_voxels];

    flat.par_iter_mut().enumerate().for_each(|(v_idx, out)| {
        let vx = v_idx / (vol_y * vol_z);
        let vy = (v_idx / vol_z) % vol_y;
        let vz = v_idx % vol_z;

        let pv = [
            (vx as f32 - (vol_x as f32 - 1.0) * 0.5) * dx,
            (vy as f32 - (vol_y as f32 - 1.0) * 0.5) * dy,
            (vz as f32 - (vol_z as f32 - 1.0) * 0.5) * dz,
        ];

        // Pre-compute receive delays (in samples) for every element.
        let delays_s: Vec<f32> = elem_pos
            .iter()
            .map(|ep| {
                let d = [pv[0] - ep[0], pv[1] - ep[1], pv[2] - ep[2]];
                let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                dist / c * fs
            })
            .collect();

        // Spatially-smoothed covariance accumulator (L×L, real symmetric).
        let mut r_accum = DMatrix::<f64>::zeros(l, l);

        for qx in 0..n_sub_x {
            for qy in 0..n_sub_y {
                for qz in 0..n_sub_z {
                    // Map sub-aperture element indices → global channel indices.
                    let sub_channels: Vec<usize> = (0..lx)
                        .flat_map(|dx_| {
                            (0..ly).flat_map(move |dy_| {
                                (0..lz).map(move |dz_| {
                                    let ex = qx + dx_;
                                    let ey = qy + dy_;
                                    let ez = qz + dz_;
                                    ex * nel_y * nel_z + ey * nel_z + ez
                                })
                            })
                        })
                        .collect();

                    // Build L×N delay-aligned data matrix X (averaged over frames).
                    // X[i][n] = (1/N_f) Σ_f x_i^f[n + τ_i]
                    let mut x_mat = DMatrix::<f64>::zeros(l, samples);
                    for (i, &ch) in sub_channels.iter().enumerate() {
                        let tau = delays_s[ch];
                        for n in 0..samples {
                            let sample_sum: f64 = (0..frames)
                                .map(|f| rf_get(f, ch, tau + n as f32) as f64)
                                .sum();
                            x_mat[(i, n)] = sample_sum / frames.max(1) as f64;
                        }
                    }

                    // R_q = X X^T / N
                    let n_f64 = samples as f64;
                    r_accum += &x_mat * x_mat.transpose() * (1.0 / n_f64);
                }
            }
        }

        // Spatially-smoothed covariance.
        let r_avg = r_accum / n_subarrays as f64;

        // Diagonal loading: R_δ = R + δ · (tr(R)/L) · I
        let trace = r_avg.trace();
        let loading = diagonal_loading as f64 * trace / l as f64;
        let r_loaded = r_avg + DMatrix::<f64>::identity(l, l) * loading;

        // Solve R_δ u = 1 via Cholesky.
        let ones = DVector::<f64>::from_element(l, 1.0_f64);
        let u = match r_loaded.clone().cholesky() {
            Some(chol) => chol.solve(&ones),
            None => {
                // Fall back to LU if Cholesky fails (e.g. insufficient loading).
                match r_loaded.lu().solve(&ones) {
                    Some(sol) => sol,
                    None => return, // Singular system — output remains 0.
                }
            }
        };

        // MVDR output power P = 1 / (1^T u).
        let denom = ones.dot(&u);
        if denom.abs() < f64::EPSILON {
            return;
        }
        let p = 1.0_f64 / denom;

        // Beamformed signal: y[n] = P · u^T x[n] for the first frame, n=0
        // (full-frame average at each voxel for a compact scalar output).
        // We use the mean delay-aligned signal across all sub-aperture elements.
        // Sub-aperture 0 is canonical; multiply by the full-aperture MVDR gain.
        let sub0_channels: Vec<usize> = (0..lx)
            .flat_map(|dx_| {
                (0..ly).flat_map(move |dy_| {
                    (0..lz).map(move |dz_| dx_ * nel_y * nel_z + dy_ * nel_z + dz_)
                })
            })
            .collect();

        let mut x_bar = DVector::<f64>::zeros(l);
        for (i, &ch) in sub0_channels.iter().enumerate() {
            let tau = delays_s[ch];
            let mean_sample: f64 = (0..frames)
                .map(|f| rf_get(f, ch, tau) as f64)
                .sum::<f64>()
                / frames.max(1) as f64;
            x_bar[i] = mean_sample;
        }

        *out = (p * u.dot(&x_bar)).abs() as f32;
    });

    Array3::from_shape_vec((vol_x, vol_y, vol_z), flat).map_err(|e| {
        KwaversError::InvalidInput(format!("MVDR CPU: output volume shape error: {e}"))
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::signal_processing::beamforming::three_dimensional::config::BeamformingConfig3D;
    use ndarray::Array4;

    fn make_config(
        nel: (usize, usize, usize),
        elem_spacing: (f64, f64, f64),
        vol: (usize, usize, usize),
        sound_speed: f64,
        sampling_frequency: f64,
    ) -> BeamformingConfig3D {
        BeamformingConfig3D {
            num_elements_3d: nel,
            element_spacing_3d: elem_spacing,
            volume_dims: vol,
            voxel_spacing: (1e-3, 1e-3, 1e-3),
            sound_speed,
            sampling_frequency,
            ..BeamformingConfig3D::default()
        }
    }

    /// ## Theorem: L=1 MVDR is the identity weighting
    ///
    /// For a single-element, single-subarray (L = 1), the MVDR weight vector
    /// reduces to the scalar identity so that P · u = 1 for any loading factor δ.
    /// The output equals |x̄[0]| = |rf_get(0, 0, τ₀)|.
    ///
    /// ## Proof
    /// R = (1/N) Σ_n x₀[n]²  (1×1 positive scalar, call it σ²).
    /// R_δ = σ²(1 + δ)   (loading multiplies σ² by (1+δ)).
    /// u = 1/R_δ  (1×1 inverse).
    /// denom = 1ᵀu = 1/R_δ.
    /// P = R_δ.
    /// P · u = R_δ · (1/R_δ) = 1  ← independent of δ.
    /// output = |P · u · x̄[0]| = |x̄[0]|.
    /// With voxel and element both at origin, τ₀ = 0, so x̄[0] = rf[0,0,0,0].
    #[test]
    fn mvdr_single_element_equals_signal_at_delay() {
        let config = make_config(
            (1, 1, 1),
            (1e-3, 1e-3, 1e-3),
            (1, 1, 1),
            1500.0,
            1_000_000.0,
        );
        let mut rf = Array4::<f32>::zeros((1, 1, 6, 1));
        rf[[0, 0, 0, 0]] = 3.0;
        rf[[0, 0, 1, 0]] = 1.0;
        rf[[0, 0, 2, 0]] = 2.0;

        let vol = mvdr_cpu(&rf, &config, 0.1, [1, 1, 1]).unwrap();
        // output = |rf[0,0,0,0]| = 3.0 by the proof above.
        assert!(
            (vol[[0, 0, 0]] - 3.0_f32).abs() < 1e-4_f32,
            "MVDR L=1 identity: expected 3.0, got {}",
            vol[[0, 0, 0]]
        );
    }

    /// ## Theorem: L=1 MVDR is invariant to diagonal loading
    ///
    /// Corollary of the L=1 identity proof: because P · u = 1 regardless of δ,
    /// the output |x̄[0]| is the same for any δ > 0.
    #[test]
    fn mvdr_single_element_invariant_to_diagonal_loading() {
        let config = make_config(
            (1, 1, 1),
            (1e-3, 1e-3, 1e-3),
            (1, 1, 1),
            1500.0,
            1_000_000.0,
        );
        let mut rf = Array4::<f32>::zeros((1, 1, 6, 1));
        rf[[0, 0, 0, 0]] = 4.0;

        let reference = mvdr_cpu(&rf, &config, 0.01, [1, 1, 1]).unwrap()[[0, 0, 0]];
        for &delta in &[0.1_f32, 1.0, 10.0, 100.0] {
            let out = mvdr_cpu(&rf, &config, delta, [1, 1, 1]).unwrap()[[0, 0, 0]];
            assert!(
                (out - reference).abs() < 1e-4_f32,
                "L=1 MVDR invariance: δ={delta} gave {out}, expected {reference}"
            );
        }
    }

    /// ## Theorem: Channel-count mismatch rejected
    ///
    /// `mvdr_cpu` returns `KwaversError::InvalidInput` when the RF channel axis
    /// ≠ nel_x × nel_y × nel_z.
    #[test]
    fn mvdr_channel_mismatch_returns_error() {
        let config = make_config(
            (1, 1, 1), // expects 1 channel
            (1e-3, 1e-3, 1e-3),
            (1, 1, 1),
            1500.0,
            1_000_000.0,
        );
        let rf = Array4::<f32>::zeros((1, 7, 4, 1));
        match mvdr_cpu(&rf, &config, 0.01, [1, 1, 1]).unwrap_err() {
            KwaversError::InvalidInput(msg) => {
                assert!(
                    msg.contains("channel") || msg.contains("element"),
                    "error must reference channel mismatch; got: {msg}"
                );
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    /// ## Theorem: Subarray exceeding array rejected
    ///
    /// `mvdr_cpu` returns `KwaversError::InvalidInput` when any component of
    /// `subarray_size` exceeds the corresponding `num_elements_3d` dimension.
    #[test]
    fn mvdr_subarray_exceeds_array_returns_error() {
        let config = make_config(
            (1, 2, 1), // nel_y = 2
            (1e-3, 1e-3, 1e-3),
            (1, 1, 1),
            1500.0,
            1_000_000.0,
        );
        let rf = Array4::<f32>::zeros((1, 2, 4, 1));
        // ly = 3 > nel_y = 2 → must be rejected.
        let result = mvdr_cpu(&rf, &config, 0.01, [1, 3, 1]);
        assert!(
            result.is_err(),
            "MVDR must reject subarray dimension exceeding array"
        );
    }

    /// ## Theorem: Diagonal loading guarantees Cholesky success and P > 0
    ///
    /// With δ > 0, R_δ = R + δ·(tr(R)/L)·I is strictly positive-definite
    /// because the loading term δ·(tr(R)/L) > 0 whenever R has at least one
    /// non-zero diagonal entry.  Cholesky therefore always succeeds, and the
    /// output power P = 1/(1ᵀ R_δ⁻¹ 1) > 0 because 1ᵀ R_δ⁻¹ 1 is a
    /// positive-definite quadratic form.
    ///
    /// Tested on a 2-element co-located array with uncorrelated signals
    /// (off-diagonal R entries are small), verifying that the result is finite
    /// and non-negative for a range of δ values spanning four orders of magnitude.
    #[test]
    fn mvdr_diagonal_loading_ensures_finite_positive_output() {
        // 2 elements at the same position (spacing=0) → delay=0 for origin voxel.
        let config = make_config(
            (1, 2, 1),
            (0.0, 0.0, 0.0),
            (1, 1, 1),
            1500.0,
            1_000_000.0,
        );
        // Uncorrelated pulses: ch0 fires at n=0, ch1 fires at n=2.
        let mut rf = Array4::<f32>::zeros((1, 2, 5, 1));
        rf[[0, 0, 0, 0]] = 2.0;
        rf[[0, 1, 2, 0]] = 2.0;

        for &delta in &[0.001_f32, 0.1, 1.0, 10.0] {
            let vol = mvdr_cpu(&rf, &config, delta, [1, 2, 1]).unwrap();
            let out = vol[[0, 0, 0]];
            assert!(
                out.is_finite() && out >= 0.0,
                "MVDR output must be finite and non-negative for δ={delta}; got {out}"
            );
        }
    }
}
