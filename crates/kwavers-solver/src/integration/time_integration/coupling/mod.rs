//! Coupling strategies for multi-rate time integration
//!
//! This module provides different strategies for coupling physics
//! components that evolve at different time scales.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use leto::Array3;
use std::collections::HashMap;
use std::fmt::Debug;

mod averaging;
mod predictor_corrector;
mod subcycling;

pub use averaging::AveragingStrategy;
pub use predictor_corrector::PredictorCorrectorStrategy;
pub use subcycling::SubcyclingStrategy;

/// Trait for time coupling strategies
pub trait TimeCoupling: Send + Sync + Debug {
    /// Advance the coupled system
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn advance_coupled_system(
        &self,
        fields: &mut HashMap<String, Array3<f64>>,
        physics_components: &HashMap<String, Box<dyn crate::plugin::Plugin>>,
        subcycles: &HashMap<String, usize>,
        global_dt: f64,
        grid: &Grid,
    ) -> KwaversResult<()>;
}
