//! Types for acoustic wave physics domain.

use crate::solver::inverse::pinn::ml::physics::BoundaryPosition;
use std::collections::HashMap;

/// Acoustic wave problem type
#[derive(Debug, Clone, PartialEq)]
pub enum AcousticProblemType {
    /// Linear acoustic wave equation
    Linear,
    /// Nonlinear acoustic wave equation (Kuznetsov)
    Nonlinear,
}

/// Acoustic boundary condition specification
#[derive(Debug, Clone)]
pub struct AcousticBoundarySpec {
    /// Boundary position
    pub position: BoundaryPosition,
    /// Boundary condition type
    pub condition_type: AcousticBoundaryType,
    /// Boundary parameters
    pub parameters: HashMap<String, f64>,
}

/// Acoustic boundary condition types
#[derive(Debug, Clone)]
pub enum AcousticBoundaryType {
    /// Sound-soft (pressure = 0)
    SoundSoft,
    /// Sound-hard (normal velocity = 0)
    SoundHard,
    /// Absorbing boundary
    Absorbing,
    /// Impedance boundary
    Impedance,
}
