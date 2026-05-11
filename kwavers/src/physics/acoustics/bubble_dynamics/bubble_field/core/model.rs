use std::collections::HashMap;

use crate::physics::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
use crate::physics::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;

use super::constants::{DEFAULT_COUPLING_THRESHOLD, DEFAULT_GRID_SPACING, DEFAULT_RHO_LIQUID};

/// Single bubble or bubble cloud field.
///
/// Stores all bubble states keyed by 3-D grid index, advances them through time
/// using the Keller-Miksis ODE, and accounts for secondary Bjerknes pressure
/// coupling between neighbouring bubbles.
#[derive(Debug)]
pub struct BubbleField {
    /// Bubble states indexed by grid position.
    pub bubbles: HashMap<(usize, usize, usize), BubbleState>,
    /// Keller-Miksis ODE solver (shared parameters across all bubbles).
    pub(super) solver: KellerMiksisModel,
    /// Default bubble parameters for cloud generation.
    pub bubble_parameters: BubbleParameters,
    /// Grid dimensions (Nx, Ny, Nz).
    pub grid_shape: (usize, usize, usize),
    /// Physical grid spacing (dx, dy, dz) (m).
    pub grid_spacing: (f64, f64, f64),
    /// Liquid density for secondary Bjerknes pressure [kg m⁻³].
    pub rho_liquid: f64,
    /// R/d threshold below which coupling contribution is skipped.
    pub coupling_threshold: f64,
    /// Time history for selected bubbles.
    pub time_history: Vec<f64>,
    pub radius_history: Vec<Vec<f64>>,
    pub temperature_history: Vec<Vec<f64>>,
}

impl BubbleField {
    /// Create a new bubble field with default 1 mm isotropic grid spacing.
    #[must_use]
    pub fn new(grid_shape: (usize, usize, usize), params: BubbleParameters) -> Self {
        Self::with_spacing(grid_shape, params, DEFAULT_GRID_SPACING)
    }

    /// Create a new bubble field with explicit physical grid spacing.
    #[must_use]
    pub fn with_spacing(
        grid_shape: (usize, usize, usize),
        params: BubbleParameters,
        spacing: (f64, f64, f64),
    ) -> Self {
        Self {
            bubbles: HashMap::new(),
            solver: KellerMiksisModel::new(params.clone()),
            bubble_parameters: params,
            grid_shape,
            grid_spacing: spacing,
            rho_liquid: DEFAULT_RHO_LIQUID,
            coupling_threshold: DEFAULT_COUPLING_THRESHOLD,
            time_history: Vec::new(),
            radius_history: Vec::new(),
            temperature_history: Vec::new(),
        }
    }

    /// Add a single bubble at a grid position.
    pub fn add_bubble(&mut self, i: usize, j: usize, k: usize, state: BubbleState) {
        self.bubbles.insert((i, j, k), state);
    }

    /// Add bubble at center of grid.
    pub fn add_center_bubble(&mut self, params: &BubbleParameters) {
        let center = (
            self.grid_shape.0 / 2,
            self.grid_shape.1 / 2,
            self.grid_shape.2 / 2,
        );
        let state = BubbleState::new(params);
        self.add_bubble(center.0, center.1, center.2, state);
    }
}
