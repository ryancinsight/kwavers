//! Fundamental wave equation validation
//! 
//! References:
//! - Pierce (1989): "Acoustics: An Introduction"
//! - Morse & Ingard (1968): "Theoretical Acoustics"

use crate::grid::Grid;
use crate::error::KwaversResult;
use ndarray::Array3;
use super::ValidationMetrics;

/// Validator for fundamental wave equations
pub trait WaveEquationValidator {
    /// Validate against 1D wave equation analytical solution
    /// u(x,t) = A*sin(kx - ωt) where ω = ck
    fn validate_1d_wave(&self, grid: &Grid, time: f64) -> KwaversResult<ValidationMetrics>;
    
    /// Validate standing wave with analytical solution
    /// p(x,t) = 2A*sin(kx)*cos(ωt)
    fn validate_standing_wave(&self, grid: &Grid, time: f64) -> KwaversResult<ValidationMetrics>;
    
    /// Validate plane wave propagation
    fn validate_plane_wave(&self, grid: &Grid, time: f64) -> KwaversResult<ValidationMetrics>;
}

/// D'Alembert solution for 1D wave equation
pub fn dalembert_solution(x: f64, t: f64, c: f64, initial_condition: impl Fn(f64) -> f64) -> f64 {
    0.5 * (initial_condition(x - c * t) + initial_condition(x + c * t))
}

/// Green's function solution for point source
pub fn greens_function_3d(r: f64, t: f64, c: f64, source_time: impl Fn(f64) -> f64) -> f64 {
    if t > r / c {
        source_time(t - r / c) / (4.0 * std::f64::consts::PI * r)
    } else {
        0.0
    }
}

/// Compute L2 norm error between numerical and analytical solutions
pub fn compute_l2_error(numerical: &Array3<f64>, analytical: &Array3<f64>) -> f64 {
    let diff = numerical - analytical;
    let sum_sq: f64 = diff.iter().map(|x| x * x).sum();
    (sum_sq / diff.len() as f64).sqrt()
}

/// Compute L-infinity norm error
pub fn compute_linf_error(numerical: &Array3<f64>, analytical: &Array3<f64>) -> f64 {
    let diff = numerical - analytical;
    diff.iter().map(|x| x.abs()).fold(0.0, f64::max)
}