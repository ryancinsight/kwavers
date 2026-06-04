//! Hemispherical Sparse Array Transducer Implementation
//!
//! Implements hemispherical phased arrays with sparse element control for
//! increased treatment envelope and improved steering efficiency.
//! Based on transcranial MR-guided focused-ultrasound hemispherical-array
//! literature.
//!
//! References:
//! - Clement & Hynynen (2002): "A non-invasive method for focusing through the skull"
//! - Pernot et al. (2003): "High power transcranial beam steering for ultrasonic brain therapy"
//! - Aubry et al. (2003): "Experimental demonstration of noninvasive transskull adaptive focusing"
//! - Hertzberg et al. (2010): "Ultrasound focusing using magnetic resonance acoustic radiation force imaging"
//! - Jones et al. (2019): "Transcranial MR-guided focused ultrasound: A review of the technology"

use kwavers_core::constants::numerical::{TWO_PI};
mod constants;
mod element;
mod geometry;
mod sparse;
mod steering;
mod validation;

// Physical constants needed for array calculations
pub use kwavers_core::constants::C_WATER;

// Explicit re-exports of hemispherical array components
pub use element::{ElementConfiguration, ElementState};
pub use geometry::{ElementPlacement, HemisphereGeometry};
pub use sparse::{ElementSelection, SparseArrayOptimizer};
pub use steering::{FocalPoint, SteeringController};
pub use validation::{ArrayValidator, HemisphericalArrayMetrics};

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_signal::{Signal, SineWave};
use kwavers_source::Source;
use ndarray::Array3;
use std::sync::Arc;

/// Hemispherical array transducer
#[derive(Debug, Clone)]
pub struct HemisphericalArray {
    elements: Vec<ElementConfiguration>,
    steering: SteeringController,
    sparse_optimizer: Option<SparseArrayOptimizer>,
    signal: Arc<dyn Signal>,
}

impl HemisphericalArray {
    /// Create new hemispherical array
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(radius: f64, num_elements: usize, frequency: f64) -> KwaversResult<Self> {
        let geometry = HemisphereGeometry::new(radius)?;
        let elements = ElementPlacement::generate_elements(&geometry, num_elements)?;
        let steering = SteeringController::new(frequency);
        let signal = Arc::new(SineWave::new(frequency, 1.0, 0.0));

        Ok(Self {
            elements,
            steering,
            sparse_optimizer: None,
            signal,
        })
    }

    /// Enable sparse array optimization
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn with_sparse_optimization(mut self, density_factor: f64) -> KwaversResult<Self> {
        self.sparse_optimizer = Some(SparseArrayOptimizer::new(density_factor)?);
        Ok(self)
    }

    /// Set focal point
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn set_focus(&mut self, focal_point: FocalPoint) -> KwaversResult<()> {
        self.steering.set_focus(focal_point, &self.elements)
    }

    /// Get active elements for current configuration
    #[must_use]
    pub fn get_active_elements(&self) -> Vec<usize> {
        self.elements
            .iter()
            .enumerate()
            .filter_map(|(i, e)| if e.is_active() { Some(i) } else { None })
            .collect()
    }
}

impl Source for HemisphericalArray {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        self.create_mask_into(grid, &mut mask);
        mask
    }

    fn create_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.dim(), (grid.nx, grid.ny, grid.nz));
        mask.fill(0.0);

        for element in &self.elements {
            if !element.is_active() {
                continue;
            }

            // Convert element position to grid indices
            if let Some((i, j, k)) = grid.position_to_indices(
                element.position[0],
                element.position[1],
                element.position[2],
            ) {
                if i < grid.nx && j < grid.ny && k < grid.nz {
                    mask[[i, j, k]] = element.amplitude;
                }
            }
        }
    }

    fn amplitude(&self, t: f64) -> f64 {
        // Return base amplitude, phase is handled in mask
        (TWO_PI * 650e3 * t).sin()
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.elements
            .iter()
            .filter(|e| e.is_active())
            .map(|e| (e.position[0], e.position[1], e.position[2]))
            .collect()
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }
}
