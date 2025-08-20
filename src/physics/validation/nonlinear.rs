//! Nonlinear acoustics validation
//! 
//! References:
//! - Hamilton & Blackstock (1998): "Nonlinear Acoustics"
//! - Kuznetsov (1971): "Equations of nonlinear acoustics"

use crate::grid::Grid;
use crate::error::KwaversResult;
use super::ValidationMetrics;

/// Validator for nonlinear acoustic phenomena
pub trait NonlinearValidator {
    /// Validate second harmonic generation (Fubini solution)
    /// For plane wave: p2/p1 = (βkx)/2 where β is nonlinearity parameter
    fn validate_second_harmonic(&self, grid: &Grid, propagation_distance: f64) -> KwaversResult<ValidationMetrics>;
    
    /// Validate shock formation distance
    /// x_shock = ρ₀c₀³/(βωp₀) for plane wave
    fn validate_shock_formation(&self, amplitude: f64, frequency: f64) -> KwaversResult<ValidationMetrics>;
    
    /// Validate Burgers equation solution
    fn validate_burgers(&self, grid: &Grid, time: f64) -> KwaversResult<ValidationMetrics>;
}

/// Fubini solution for weakly nonlinear plane wave
pub fn fubini_solution(x: f64, k: f64, beta: f64, p0: f64, n: usize) -> f64 {
    // Harmonic amplitude: An = (2/nπσ) * Jn(nσ)
    // where σ = βkxM₀, M₀ = p₀/(ρ₀c₀²)
    let sigma = beta * k * x * p0;
    
    (1..=n).map(|harmonic| {
        let n = harmonic as f64;
        let bessel = bessel_jn(harmonic as i32, sigma);
        2.0 * bessel / (n * std::f64::consts::PI * sigma)
    }).sum()
}

/// Mendousse solution for nonlinear absorption
pub fn mendousse_solution(x: f64, alpha: f64, beta: f64, k: f64, p0: f64) -> f64 {
    let g = beta * k * p0 / (2.0 * alpha);
    p0 * f64::exp(-alpha * x) * (1.0 + g * x).sqrt()
}

// Placeholder for Bessel function - should use proper implementation
fn bessel_jn(_n: i32, _x: f64) -> f64 {
    // This should use a proper Bessel function implementation
    // For now, return approximation for small arguments
    1.0
}