//! Core hybrid PSTD/FDTD solver implementation

use kwavers_domain::field::wave::WaveFields;
use kwavers_grid::Grid;
use crate::forward::fdtd::FdtdSolver;
use crate::forward::hybrid::adaptive_selection::AdaptiveSelector;
use crate::forward::hybrid::config::HybridConfig;
use crate::forward::hybrid::coupling::CouplingInterface;
use crate::forward::hybrid::domain_decomposition::{DomainDecomposer, DomainRegion};
use crate::forward::hybrid::metrics::{HybridMetrics, HybridValidationResults};
use crate::forward::pstd::PSTDSolver;
use ndarray::Array3;

mod construction;
mod interface_impl;
mod stepping;
mod update;

const HYBRID_BLEND_WIDTH: usize = 5;

/// PSTD blending weight for a hybrid transition region.
///
/// ## Theorem
/// The raised-cosine partition
/// `w(d) = 0.5 * (1 - cos(pi * d / W))`, clamped to `1` for `d >= W`,
/// satisfies `w(0)=0`, `w(W)=1`, and `0 <= w <= 1`. Therefore the blended
/// field `w * pstd + (1-w) * fdtd` is a convex combination that preserves
/// boundedness across the FDTD/PSTD interface while transitioning smoothly from
/// the finite-difference boundary state to the spectral interior state.
#[inline]
fn hybrid_pstd_weight(distance_from_boundary: f64, blend_width: usize) -> f64 {
    if blend_width == 0 || distance_from_boundary >= blend_width as f64 {
        return 1.0;
    }
    0.5 * (1.0 - (std::f64::consts::PI * distance_from_boundary / blend_width as f64).cos())
}

/// Hybrid PSTD/FDTD solver combining spectral and finite-difference methods
#[derive(Debug)]
pub struct HybridSolver {
    /// Configuration
    pub(super) config: HybridConfig,

    /// Computational grid
    pub(super) grid: Grid,

    /// PSTD solver for smooth regions
    pub(super) pstd_solver: PSTDSolver,

    /// FDTD solver for discontinuous regions
    pub(super) fdtd_solver: FdtdSolver,

    // Unified Fields
    pub(super) fields: WaveFields,

    /// Reusable source-mask workspace for update-time injection.
    pub(super) source_mask_scratch: Array3<f64>,

    /// Domain decomposer
    pub(super) decomposer: DomainDecomposer,

    /// Adaptive selector for method choice
    pub(super) selector: AdaptiveSelector,

    /// Coupling interface manager
    pub(super) coupling: CouplingInterface,

    /// Current domain regions
    pub(super) regions: Vec<DomainRegion>,

    /// Performance metrics
    pub(super) metrics: HybridMetrics,

    /// Validation results
    pub(super) validation_results: HybridValidationResults,

    /// Time step counter
    pub(super) time_step: usize,
}

#[cfg(test)]
mod tests {
    use super::{hybrid_pstd_weight, HYBRID_BLEND_WIDTH};

    #[test]
    fn hybrid_blend_weight_transitions_from_fdtd_boundary_to_pstd_interior() {
        assert_eq!(hybrid_pstd_weight(0.0, HYBRID_BLEND_WIDTH), 0.0);
        assert!((hybrid_pstd_weight(2.5, HYBRID_BLEND_WIDTH) - 0.5).abs() < 1e-12);
        assert_eq!(
            hybrid_pstd_weight(HYBRID_BLEND_WIDTH as f64, HYBRID_BLEND_WIDTH),
            1.0
        );
        assert_eq!(
            hybrid_pstd_weight((HYBRID_BLEND_WIDTH + 1) as f64, HYBRID_BLEND_WIDTH),
            1.0
        );
    }
}
