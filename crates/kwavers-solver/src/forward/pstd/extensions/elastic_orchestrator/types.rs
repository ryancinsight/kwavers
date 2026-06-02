//! Public data types for the ElasticPSTD orchestrator.

use ndarray::{Array1, Array2, Array3};

/// Velocity-source injection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElasticPstdSourceMode {
    /// `vx[mask] += signal[step]` per active mask point.
    Additive,
    /// `vx[mask] = signal[step]` per active mask point.
    Dirichlet,
}

/// Velocity-source spec — boolean mask + per-axis 1-D time signals.
///
/// At least one of `ux/uy/uz` must be `Some`. The signal length must equal
/// the simulation's `time_steps` argument.
#[derive(Debug, Clone)]
pub struct ElasticPstdVelocitySource {
    pub mask: Array3<bool>,
    pub ux: Option<Array1<f64>>,
    pub uy: Option<Array1<f64>>,
    pub uz: Option<Array1<f64>>,
    pub mode: ElasticPstdSourceMode,
}

/// Recorded sensor traces — one row per active sensor mask point, columns
/// indexed by step.
#[derive(Debug)]
pub struct ElasticPstdSensorData {
    pub vx: Option<Array2<f64>>,
    pub vy: Option<Array2<f64>>,
    pub vz: Option<Array2<f64>>,
}

/// Build context that wires medium properties into the orchestrator.
#[derive(Debug)]
pub struct ElasticPstdMedium {
    pub lame_lambda: Array3<f64>,
    pub lame_mu: Array3<f64>,
    pub density: Array3<f64>,
}
