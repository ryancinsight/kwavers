//! `VelocityMap` — output of the velocity mapping algorithm.

use ndarray::Array2;

/// Output of the velocity mapping algorithm.
#[derive(Debug)]
pub struct VelocityMap {
    /// Mean lateral velocity component ⟨v_x⟩ (m/s).  NaN where count < min_count.
    pub vx: Array2<f64>,
    /// Mean axial velocity component ⟨v_z⟩ (m/s).  NaN where count < min_count.
    pub vz: Array2<f64>,
    /// Velocity magnitude (speed) (m/s).  NaN where count < min_count.
    pub speed: Array2<f64>,
    /// Flow direction (rad) ∈ (−π, π].  NaN where count < min_count.
    pub direction: Array2<f64>,
    /// Wall shear stress proxy μ · ‖∇speed‖ (Pa).
    /// NaN at boundaries and cells with NaN neighbors.
    pub wall_shear_stress: Array2<f64>,
    /// Number of velocity estimates accumulated per cell.
    pub count: Array2<u32>,
}
