//! `BoundaryCondition` trait implementation for `SchwarzBoundary`.
//!
//! The Schwarz transmission step proper happens via [`super::SchwarzBoundary::apply_transmission`];
//! this trait impl is the framework's spatial/frequency apply hook, which is a
//! no-op because Schwarz coupling needs an inter-subdomain communication
//! infrastructure that lives outside the boundary trait surface.

use ndarray::{Array3, ArrayViewMut3};

use super::SchwarzBoundary;
use kwavers_core::error::KwaversResult;
use crate::boundary::coupling::types::BoundaryDirections;
use crate::boundary::traits::BoundaryCondition;
use crate::grid::GridTopology;

impl BoundaryCondition for SchwarzBoundary {
    fn name(&self) -> &str {
        "SchwarzBoundary"
    }

    fn active_directions(&self) -> BoundaryDirections {
        self.directions
    }

    fn apply_scalar_spatial(
        &mut self,
        _field: ArrayViewMut3<f64>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Schwarz boundary application requires:
        // 1. Identifying interface regions
        // 2. Exchanging data with neighboring subdomains
        // 3. Applying transmission condition via apply_transmission()
        //
        // Inter-subdomain communication infrastructure is out of scope for the
        // single-domain BoundaryCondition trait — `apply_transmission` is the
        // core entry point.
        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        _field: &mut Array3<num_complex::Complex<f64>>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Schwarz boundary in frequency domain — same rationale as the
        // spatial impl above.
        Ok(())
    }

    fn reset(&mut self) {
        // No state to reset for Schwarz boundary.
    }
}
