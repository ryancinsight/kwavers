//! `FocalKernel` — a per-voxel focal-pressure field with geometry +
//! source metadata, captured from a single PSTD pulse.

use leto::Array3;

/// A cached focal-pressure kernel from a single focused-bowl PSTD pulse.
///
/// The `field` array stores the per-voxel **peak rarefactional pressure**
/// (positive Pa, equal to `-p_min` from the PSTD recorder) over the
/// captured time window. It encodes real diffraction, sidelobes, and
/// depth-of-focus; treatment planners read it instead of an analytical
/// Gaussian focal-envelope approximation.
#[derive(Debug, Clone)]
pub struct FocalKernel {
    /// Peak rarefactional pressure per voxel (Pa, positive).
    pub field: Array3<f64>,
    /// Grid spacing of the field array (m), isotropic.
    pub dx_m: f64,
    /// Grid index of the focal voxel within `field`.
    pub focus_idx: (usize, usize, usize),
    /// Source centre frequency (Hz).
    pub f0: f64,
    /// Realised peak rarefactional pressure at the focal voxel (Pa).
    /// May differ from the calibration target by a few %.
    pub pnp_realised: f64,
    /// Source drive pressure at the bowl surface (Pa).
    pub source_pa: f64,
    /// Penttinen 1976 lateral focal FWHM (m).
    pub fwhm_lat_m: f64,
    /// Penttinen 1976 axial focal FWHM (m).
    pub fwhm_ax_m: f64,
}

impl FocalKernel {
    /// Construct a `FocalKernel` directly. The caller is responsible for
    /// providing a non-negative `field` (peak rarefactional pressure is
    /// non-negative by definition) and for ensuring `focus_idx` is in
    /// bounds. Both invariants are debug-asserted.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        field: Array3<f64>,
        dx_m: f64,
        focus_idx: (usize, usize, usize),
        f0: f64,
        pnp_realised: f64,
        source_pa: f64,
        fwhm_lat_m: f64,
        fwhm_ax_m: f64,
    ) -> Self {
        debug_assert!(dx_m > 0.0, "FocalKernel::new requires dx_m > 0");
        debug_assert!(f0 > 0.0, "FocalKernel::new requires f0 > 0");
        let (nx, ny, nz) = field.dim();
        debug_assert!(
            focus_idx.0 < nx && focus_idx.1 < ny && focus_idx.2 < nz,
            "FocalKernel::new focus_idx out of bounds"
        );
        Self {
            field,
            dx_m,
            focus_idx,
            f0,
            pnp_realised,
            source_pa,
            fwhm_lat_m,
            fwhm_ax_m,
        }
    }

    /// Shape `(nx, ny, nz)` of the field array.
    #[must_use]
    pub fn shape(&self) -> (usize, usize, usize) {
        self.field.dim()
    }

    /// Peak rarefactional pressure at the focal voxel (Pa, positive).
    /// Guaranteed `≥ 0` because the field array stores peak rarefactional
    /// values only.
    #[must_use]
    pub fn focal_pressure(&self) -> f64 {
        self.field[self.focus_idx]
    }

    /// Linearly rescale the field by a multiplicative factor. Used to
    /// derive a kernel at a different target `pnp` without rerunning
    /// the wave solver — exact in the linear-water regime (B/A = 0).
    pub fn scale_in_place(&mut self, factor: f64) {
        self.field.mapv_inplace(|p| p * factor);
        self.pnp_realised *= factor;
        self.source_pa *= factor;
    }
}
