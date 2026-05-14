//! `KernelCube` — bilinear interpolator across a sparse `(f0, pnp)`
//! grid of cached [`FocalKernel`]s.

use ndarray::{Array3, Zip};

use crate::core::error::{KwaversError, KwaversResult};

use super::{kernel::FocalKernel, placement::place_kernel_at_focus, resample::resample_trilinear};

/// Bilinear interpolator across a sparse `(f0, pnp)` kernel sweep.
///
/// Constructed from a `Vec<FocalKernel>` whose `(f0, pnp_realised)`
/// pairs form a Cartesian grid. After construction, [`Self::query`]
/// returns a normalized (`env.max() == 1`) per-voxel focal envelope on
/// the planner grid for any `(f0, pnp)` query — corner, edge, interior
/// or clamped-outside.
///
/// Physics
/// -------
///
/// * The **`pnp` dimension is degenerate** in the linear-water regime
///   (`B/A = 0`): the kernel field shape is invariant under amplitude
///   scaling, and [`Self::query`] normalizes by the global max anyway.
///   The `pnp` parameter is retained on the API for symmetry with
///   `f0` but does not drive shape selection.
///
/// * The **`f0` dimension is real**: focal-spot dimensions scale with
///   wavelength (Penttinen 1976). Two kernels at neighbouring `f0`
///   sweep points are linearly blended after resampling each to the
///   planner grid; the result is then re-normalized by global max.
///
/// * Queries **outside** the sweep clamp to the nearest corner — no
///   extrapolation, since the focal-spot ratio breaks for wavelength
///   ratios further than the sweep spacing.
#[derive(Debug)]
pub struct KernelCube {
    kernels: Vec<FocalKernel>,
    f0_values: Vec<f64>,
    pnp_values: Vec<f64>,
}

impl KernelCube {
    /// Construct from a flat list of kernels. The `(f0, pnp_realised)`
    /// pairs are extracted, deduplicated, and sorted to form the
    /// interpolation axes. Returns an error if any combination of
    /// `(f0, pnp)` axis values is missing from the input set
    /// (the cube must be Cartesian-complete).
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn new(kernels: Vec<FocalKernel>) -> KwaversResult<Self> {
        if kernels.is_empty() {
            return Err(KwaversError::InvalidInput(
                "KernelCube::new requires at least one kernel".to_owned(),
            ));
        }
        let mut f0_set: Vec<f64> = kernels.iter().map(|k| k.f0).collect();
        let mut pnp_set: Vec<f64> = kernels.iter().map(|k| k.pnp_realised).collect();
        f0_set.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pnp_set.sort_by(|a, b| a.partial_cmp(b).unwrap());
        f0_set.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
        pnp_set.dedup_by(|a, b| (*a - *b).abs() < 1e3);

        // Cartesian-completeness check: for every (f0, pnp) combo, at
        // least one kernel within tolerance must exist.
        for f0 in &f0_set {
            for pnp in &pnp_set {
                let exists = kernels
                    .iter()
                    .any(|k| (k.f0 - *f0).abs() < 1e-3 && (k.pnp_realised - *pnp).abs() < 1e3);
                if !exists {
                    return Err(KwaversError::InvalidInput(format!(
                        "KernelCube::new missing kernel for (f0={}, pnp={})",
                        f0, pnp
                    )));
                }
            }
        }

        Ok(Self {
            kernels,
            f0_values: f0_set,
            pnp_values: pnp_set,
        })
    }

    /// Sorted unique `f0` axis values (Hz).
    #[must_use]
    pub fn f0_axis(&self) -> &[f64] {
        &self.f0_values
    }

    /// Sorted unique `pnp` axis values (Pa).
    #[must_use]
    pub fn pnp_axis(&self) -> &[f64] {
        &self.pnp_values
    }

    /// Number of cached kernels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// `true` when no kernels are cached. (Trivially `false` after
    /// successful construction; provided for API symmetry.)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Returns the bracket `(f0_lo, f0_hi, alpha)` for blending. When
    /// the query lies outside the sweep, both ends collapse to the
    /// nearest corner with `alpha = 0` (no extrapolation).
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    fn bracket_f0(&self, f0: f64) -> (f64, f64, f64) {
        let lo = self.f0_values[0];
        let hi = *self.f0_values.last().unwrap();
        if f0 <= lo {
            return (lo, lo, 0.0);
        }
        if f0 >= hi {
            return (hi, hi, 0.0);
        }
        for w in self.f0_values.windows(2) {
            if w[0] <= f0 && f0 <= w[1] {
                return (w[0], w[1], (f0 - w[0]) / (w[1] - w[0]));
            }
        }
        (lo, lo, 0.0) // unreachable but defensive
    }

    /// Find any kernel matching `(f0, *)` — caller picks any pnp since
    /// the field shape is amplitude-invariant under linear water.
    /// # Panics
    /// - Panics if `KernelCube::new validates Cartesian completeness`.
    ///
    fn pick_kernel_at_f0(&self, f0: f64) -> &FocalKernel {
        self.kernels
            .iter()
            .find(|k| (k.f0 - f0).abs() < 1e-3)
            .expect("KernelCube::new validates Cartesian completeness")
    }

    /// Build a normalized envelope at a single `f0` sweep point on the
    /// caller-specified planner grid.
    fn envelope_at_f0(
        &self,
        f0: f64,
        target_shape: (usize, usize, usize),
        target_focus_idx: (usize, usize, usize),
        target_dx_m: f64,
    ) -> Array3<f64> {
        let kernel = self.pick_kernel_at_f0(f0);
        let resampled = resample_trilinear(kernel, target_dx_m);
        let mut placed = place_kernel_at_focus(&resampled, target_shape, target_focus_idx);
        let peak = placed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if peak > 0.0 {
            placed.mapv_inplace(|v| v / peak);
        }
        placed
    }

    /// Query a normalized focal envelope (`env.max() == 1`) at any
    /// `(f0, pnp)` within the sweep. The `pnp` argument is accepted
    /// for API symmetry but is **not** used to select shape (the
    /// linear-water regime makes envelope shape amplitude-invariant);
    /// the caller's source-amplitude scaling carries absolute Pa.
    ///
    /// `f0` outside the sweep clamps to the nearest corner.
    #[must_use]
    pub fn query(
        &self,
        f0: f64,
        _pnp: f64,
        target_shape: (usize, usize, usize),
        target_focus_idx: (usize, usize, usize),
        target_dx_m: f64,
    ) -> Array3<f64> {
        let (f0_lo, f0_hi, alpha) = self.bracket_f0(f0);
        let env_lo = self.envelope_at_f0(f0_lo, target_shape, target_focus_idx, target_dx_m);
        if alpha == 0.0 {
            return env_lo;
        }
        let env_hi = self.envelope_at_f0(f0_hi, target_shape, target_focus_idx, target_dx_m);
        // In-place blend into `env_lo` (consumed by ownership) — saves
        // ~2×N³×f64 of intermediate allocation vs the naive
        // `&env_lo * (1-α) + &env_hi * α`. Parallel Zip splits the
        // work across cores; the per-element fused multiply-add is
        // the entire inner loop.
        let mut blend = env_lo;
        let inv_alpha = 1.0 - alpha;
        Zip::from(&mut blend).and(&env_hi).par_for_each(|lo, &hi| {
            *lo = (*lo).mul_add(inv_alpha, hi * alpha);
        });
        let peak = blend.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if peak > 0.0 {
            blend.mapv_inplace(|v| v / peak);
        }
        blend
    }
}
