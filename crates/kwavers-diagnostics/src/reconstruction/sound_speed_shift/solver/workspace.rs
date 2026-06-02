//! Workspace lifecycle for speed-shift inverse solves.

use std::mem::size_of;

use super::super::types::SoundSpeedShiftWorkspace;

impl SoundSpeedShiftWorkspace {
    /// Construct an empty reusable workspace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the total retained `f64` capacity across all work vectors.
    #[must_use]
    pub fn allocated_slots(&self) -> usize {
        self.rhs.capacity()
            + self.diagonal.capacity()
            + self.solution.capacity()
            + self.normal_solution.capacity()
            + self.residual.capacity()
            + self.preconditioned.capacity()
            + self.direction.capacity()
            + self.normal_direction.capacity()
            + self.row.capacity()
            + self.laplacian.capacity()
            + self.prediction.capacity()
            + self.previous_solution.capacity()
            + self.power_vector.capacity()
            + self.power_normal.capacity()
            + self.objective_history.capacity()
    }

    /// Return retained workspace memory in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.allocated_slots() * size_of::<f64>()
    }

    /// Zero active numeric buffers while preserving allocations.
    pub fn clear(&mut self) {
        self.rhs.fill(0.0);
        self.diagonal.fill(0.0);
        self.solution.fill(0.0);
        self.normal_solution.fill(0.0);
        self.residual.fill(0.0);
        self.preconditioned.fill(0.0);
        self.direction.fill(0.0);
        self.normal_direction.fill(0.0);
        self.row.fill(0.0);
        self.laplacian.fill(0.0);
        self.prediction.fill(0.0);
        self.previous_solution.fill(0.0);
        self.power_vector.fill(0.0);
        self.power_normal.fill(0.0);
        self.objective_history.clear();
    }

    pub(super) fn prepare(&mut self, rows: usize, cols: usize) {
        resize_zero(&mut self.rhs, cols);
        resize_zero(&mut self.diagonal, cols);
        resize_zero(&mut self.solution, cols);
        resize_zero(&mut self.normal_solution, cols);
        resize_zero(&mut self.residual, cols);
        resize_zero(&mut self.preconditioned, cols);
        resize_zero(&mut self.direction, cols);
        resize_zero(&mut self.normal_direction, cols);
        resize_zero(&mut self.row, rows);
        resize_zero(&mut self.laplacian, cols);
        resize_zero(&mut self.prediction, rows);
        resize_zero(&mut self.previous_solution, cols);
        resize_zero(&mut self.power_vector, cols);
        resize_zero(&mut self.power_normal, cols);
        self.objective_history.clear();
    }
}

fn resize_zero(values: &mut Vec<f64>, len: usize) {
    values.resize(len, 0.0);
    values.fill(0.0);
}
