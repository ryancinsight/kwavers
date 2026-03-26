use super::super::adaptive_integration::integrate_bubble_dynamics_adaptive;
use super::super::bubble_state::{BubbleParameters, BubbleState};
use super::super::keller_miksis::KellerMiksisModel;
use ndarray::Array3;
use std::collections::HashMap;

/// Single bubble or bubble cloud field
#[derive(Debug)]
pub struct BubbleField {
    /// Bubble states indexed by grid position
    pub bubbles: HashMap<(usize, usize, usize), BubbleState>,
    /// Solver for bubble dynamics
    solver: KellerMiksisModel,
    /// Default bubble parameters
    pub bubble_parameters: BubbleParameters,
    /// Grid dimensions
    pub grid_shape: (usize, usize, usize),
    /// Time history for selected bubbles
    pub time_history: Vec<f64>,
    pub radius_history: Vec<Vec<f64>>,
    pub temperature_history: Vec<Vec<f64>>,
}

impl BubbleField {
    /// Create new bubble field
    #[must_use]
    pub fn new(grid_shape: (usize, usize, usize), params: BubbleParameters) -> Self {
        Self {
            bubbles: HashMap::new(),
            solver: KellerMiksisModel::new(params.clone()),
            bubble_parameters: params,
            grid_shape,
            time_history: Vec::new(),
            radius_history: Vec::new(),
            temperature_history: Vec::new(),
        }
    }

    /// Add a single bubble at grid position
    pub fn add_bubble(&mut self, i: usize, j: usize, k: usize, state: BubbleState) {
        self.bubbles.insert((i, j, k), state);
    }

    /// Add bubble at center of grid
    pub fn add_center_bubble(&mut self, params: &BubbleParameters) {
        let center = (
            self.grid_shape.0 / 2,
            self.grid_shape.1 / 2,
            self.grid_shape.2 / 2,
        );
        let state = BubbleState::new(params);
        self.add_bubble(center.0, center.1, center.2, state);
    }

    /// Update all bubbles for one time step
    pub fn update(
        &mut self,
        pressure_field: &Array3<f64>,
        dp_dt_field: &Array3<f64>,
        dt: f64,
        t: f64,
    ) {
        // Update each bubble
        for ((i, j, k), state) in &mut self.bubbles {
            let p_acoustic = pressure_field[[*i, *j, *k]];
            let dp_dt = dp_dt_field[[*i, *j, *k]];

            // Use adaptive integration (no Mutex needed anymore)
            if let Err(e) =
                integrate_bubble_dynamics_adaptive(&self.solver, state, p_acoustic, dp_dt, dt, t)
            {
                eprintln!("Bubble dynamics integration failed at position ({i}, {j}, {k}): {e:?}");
            }
        }

        // Record history for tracking
        self.record_history(t);
    }

    /// Record time history of bubble states
    fn record_history(&mut self, t: f64) {
        self.time_history.push(t);

        // Initialize history vectors if needed
        if self.radius_history.is_empty() {
            for _ in 0..self.bubbles.len() {
                self.radius_history.push(Vec::new());
                self.temperature_history.push(Vec::new());
            }
        }

        // Record each bubble's state
        for (idx, (_, state)) in self.bubbles.iter().enumerate() {
            self.radius_history[idx].push(state.radius);
            self.temperature_history[idx].push(state.temperature);
        }
    }

    /// Get bubble state fields for physics modules
    #[must_use]
    pub fn get_state_fields(&self) -> crate::domain::field::BubbleStateFields {
        let shape = self.grid_shape;
        let mut fields = crate::domain::field::BubbleStateFields::new(shape);

        for ((i, j, k), state) in &self.bubbles {
            fields.radius[[*i, *j, *k]] = state.radius;
            fields.temperature[[*i, *j, *k]] = state.temperature;
            fields.pressure[[*i, *j, *k]] = state.pressure_internal;
            fields.velocity[[*i, *j, *k]] = state.wall_velocity;
            fields.is_collapsing[[*i, *j, *k]] = f64::from(i32::from(state.is_collapsing));
            fields.compression_ratio[[*i, *j, *k]] = state.compression_ratio;
        }

        fields
    }

    /// Get statistics about bubble field
    #[must_use]
    pub fn get_statistics(&self) -> BubbleFieldStats {
        let mut stats = BubbleFieldStats::default();

        for state in self.bubbles.values() {
            stats.total_bubbles += 1;
            if state.is_collapsing {
                stats.collapsing_bubbles += 1;
            }
            stats.max_temperature = stats.max_temperature.max(state.temperature);
            stats.max_compression = stats.max_compression.max(state.compression_ratio);
            stats.total_collapses += state.collapse_count;
        }

        stats
    }
}

/// Statistics about bubble field
#[derive(Debug, Default)]
pub struct BubbleFieldStats {
    pub total_bubbles: usize,
    pub collapsing_bubbles: usize,
    pub max_temperature: f64,
    pub max_compression: f64,
    pub total_collapses: u32,
}
