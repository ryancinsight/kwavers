// adaptive_selection/selector.rs - Main adaptive method selector

use super::criteria::SelectionCriteria;
use super::metrics::{ComputationalMetrics, MaterialMetrics, SpectralMetrics};
use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array3, Array4};

/// Method selection result
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectedMethod {
    Spectral,
    FiniteDifference,
    DiscontinuousGalerkin,
}

/// Adaptive method selector
#[derive(Debug)]
pub struct AdaptiveMethodSelector {
    criteria: SelectionCriteria,
    previous_selection: Option<Array3<SelectedMethod>>,
}

impl AdaptiveMethodSelector {
    /// Create new selector
    #[must_use]
    pub fn new(criteria: SelectionCriteria) -> Self {
        Self {
            criteria,
            previous_selection: None,
        }
    }

    /// Select optimal method for each region
    pub fn select_methods(
        &mut self,
        fields: &Array4<f64>,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<Array3<SelectedMethod>> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut selection = Array3::from_elem((nx, ny, nz), SelectedMethod::Spectral);

        // Analyze field properties
        let pressure_field = fields.index_axis(ndarray::Axis(0), 0);

        // Compute metrics for different regions
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let method = self.select_for_point(pressure_field, (i, j, k), grid, dt);
                    selection[[i, j, k]] = method;
                }
            }
        }

        // Apply hysteresis to prevent oscillation
        if let Some(ref prev) = self.previous_selection {
            self.apply_hysteresis(&mut selection, prev);
        }

        self.previous_selection = Some(selection.clone());
        Ok(selection)
    }

    /// Select method for a single point
    fn select_for_point(
        &self,
        field: ndarray::ArrayView3<f64>,
        position: (usize, usize, usize),
        grid: &Grid,
        dt: f64,
    ) -> SelectedMethod {
        // Extract local region around point
        let region = self.extract_region(field, position);

        // Compute metrics
        let spectral = SpectralMetrics::compute(region.view(), grid);
        let material = MaterialMetrics::compute(
            region.view(),
            region.view(),
            (1, 1, 1), // Center of extracted region
        );
        let computational = ComputationalMetrics::compute(grid, dt, 1500.0); // Assume typical sound speed

        // Weighted score for each method
        let spectral_score = self.compute_spectral_score(&spectral, &material, &computational);
        let fd_score = self.compute_fd_score(&spectral, &material, &computational);
        let dg_score = self.compute_dg_score(&spectral, &material, &computational);

        // Select method with highest score
        if spectral_score >= fd_score && spectral_score >= dg_score {
            SelectedMethod::Spectral
        } else if fd_score >= dg_score {
            SelectedMethod::FiniteDifference
        } else {
            SelectedMethod::DiscontinuousGalerkin
        }
    }

    /// Extract local region around a point
    fn extract_region(
        &self,
        field: ndarray::ArrayView3<f64>,
        position: (usize, usize, usize),
    ) -> Array3<f64> {
        let (i, j, k) = position;
        let (nx, ny, nz) = field.dim();
        const REGION_SIZE: usize = 3;

        // Extract 3x3x3 region centered at position
        let mut region = Array3::zeros((REGION_SIZE, REGION_SIZE, REGION_SIZE));

        for dk in 0..REGION_SIZE {
            for dj in 0..REGION_SIZE {
                for di in 0..REGION_SIZE {
                    let gi = (i + di).saturating_sub(1).min(nx - 1);
                    let gj = (j + dj).saturating_sub(1).min(ny - 1);
                    let gk = (k + dk).saturating_sub(1).min(nz - 1);

                    region[[di, dj, dk]] = field[[gi, gj, gk]];
                }
            }
        }

        region
    }

    /// Compute score for spectral method
    fn compute_spectral_score(
        &self,
        spectral: &SpectralMetrics,
        material: &MaterialMetrics,
        computational: &ComputationalMetrics,
    ) -> f64 {
        let mut score = 0.0;

        // Spectral methods excel with smooth fields
        score += self.criteria.smoothness_weight * (1.0 - spectral.smoothness);

        // Good for homogeneous media
        score += self.criteria.material_weight * material.homogeneity;

        // Requires good resolution
        score += self.criteria.efficiency_weight * computational.grid_resolution_quality;

        // Low frequency content preferred
        score += self.criteria.frequency_weight * (1.0 - spectral.frequency_content);

        score
    }

    /// Compute score for finite difference method
    fn compute_fd_score(
        &self,
        _spectral: &SpectralMetrics,
        material: &MaterialMetrics,
        computational: &ComputationalMetrics,
    ) -> f64 {
        let mut score = 0.0;

        // FD handles moderate smoothness well
        score += self.criteria.smoothness_weight * 0.5;

        // Works with heterogeneous media
        score += self.criteria.material_weight * (1.0 - material.homogeneity * 0.5);

        // Good stability
        score += self.criteria.efficiency_weight * computational.stability_margin;

        // Moderate frequency handling
        score += self.criteria.frequency_weight * 0.5;

        score
    }

    /// Compute score for discontinuous Galerkin method
    fn compute_dg_score(
        &self,
        spectral: &SpectralMetrics,
        material: &MaterialMetrics,
        computational: &ComputationalMetrics,
    ) -> f64 {
        let mut score = 0.0;

        // DG excels with discontinuities
        score += self.criteria.smoothness_weight * spectral.smoothness;

        // Good for interfaces
        score += self.criteria.material_weight * material.interface_proximity;

        // Memory intensive
        score += self.criteria.efficiency_weight * computational.memory_efficiency * 0.5;

        // Handles high frequencies
        score += self.criteria.frequency_weight * spectral.frequency_content;

        score
    }

    /// Apply hysteresis to prevent oscillation
    /// Only switches method if previous method differs AND hysteresis threshold exceeded
    /// Per Persson & Peraire (2006): "Sub-Cell Shock Capturing for Discontinuous Galerkin Methods"
    fn apply_hysteresis(
        &self,
        selection: &mut Array3<SelectedMethod>,
        previous: &Array3<SelectedMethod>,
    ) {
        let threshold = self.criteria.hysteresis_factor;

        // Hysteresis prevents method switching unless significant change warranted
        // threshold < 0.5: Allow more switching (responsive)
        // threshold > 0.5: Prevent switching (stable)
        // threshold = 0.0: No hysteresis (fully responsive)
        // threshold = 1.0: Maximum hysteresis (very stable)

        for ((i, j, k), current) in selection.indexed_iter_mut() {
            if *current != previous[[i, j, k]] {
                // Prevent switching if hysteresis threshold indicates stability preference
                // In proper implementation, would compare score differences to threshold
                // Current: Conservative approach - keep previous if threshold > 0.5
                if threshold > 0.5 {
                    *current = previous[[i, j, k]];
                }
                // Future: Store scores and use: if |score_new - score_old| < threshold { keep old }
            }
        }
    }

    /// Update internal metrics
    pub fn update_metrics(&mut self, _fields: &Array4<f64>) {
        // Metrics are computed on-demand in select_methods
    }
}

/// Type alias for convenience
pub type AdaptiveSelector = AdaptiveMethodSelector;
