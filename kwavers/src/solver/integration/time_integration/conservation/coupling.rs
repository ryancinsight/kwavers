//! Conservative coupling interface for multi-rate integration.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;
use std::collections::HashMap;

/// Conservative coupling interface for multi-rate integration
pub trait ConservativeCoupling {
    /// Apply conservative coupling between fast and slow components
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_conservative_coupling(
        &self,
        high_frequency_fields: &mut HashMap<String, Array3<f64>>,
        low_frequency_fields: &mut HashMap<String, Array3<f64>>,
        dt: f64,
        grid: &Grid,
    ) -> KwaversResult<()>;

    /// Compute flux corrections for conservation
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_flux_corrections(
        &self,
        fields: &HashMap<String, Array3<f64>>,
        grid: &Grid,
    ) -> KwaversResult<HashMap<String, Array3<f64>>>;
}
