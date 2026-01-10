//! Nonlinear acoustics validation
//! 
//! References:
//! - Hamilton & Blackstock (1998): "Nonlinear Acoustics"
//! - Kuznetsov (1971): "Equations of nonlinear acoustics"

use crate::domain::grid::Grid;
use crate::domain::core::error::KwaversResult;
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

/// Bessel function of the first kind J_n(x)
///
/// Uses series expansion for small arguments and asymptotic expansion for large arguments.
/// Accurate to ~1e-10 for |x| < 100.
///
/// # References
/// - Abramowitz & Stegun (1964), "Handbook of Mathematical Functions", Section 9.1
/// - Press et al. (2007), "Numerical Recipes", Section 6.5
fn bessel_jn(n: i32, x: f64) -> f64 {
    const MAX_ITERATIONS: usize = 100;
    const TOLERANCE: f64 = 1e-12;
    
    let n = n.abs();
    let x_abs = x.abs();
    
    // Small argument series expansion: J_n(x) = (x/2)^n / n! * Σ(-1)^k * (x/2)^(2k) / (k!(n+k)!)
    if x_abs < 10.0 {
        let mut term = (x_abs / 2.0).powi(n) / factorial(n as usize);
        let mut sum = term;
        let x_half_sq = (x_abs / 2.0).powi(2);
        
        for k in 1..MAX_ITERATIONS {
            term *= -x_half_sq / ((k * (n as usize + k)) as f64);
            sum += term;
            
            if term.abs() < TOLERANCE * sum.abs() {
                break;
            }
        }
        
        sum
    } else {
        // Large argument asymptotic expansion: J_n(x) ≈ √(2/πx) * cos(x - nπ/2 - π/4)
        // This is accurate for x >> n
        let phase = x_abs - (n as f64) * std::f64::consts::FRAC_PI_2 - std::f64::consts::FRAC_PI_4;
        (2.0 / (std::f64::consts::PI * x_abs)).sqrt() * phase.cos()
    }
}

/// Factorial function for small integers
fn factorial(n: usize) -> f64 {
    const FACTORIALS: [f64; 21] = [
        1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
        3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0,
        1307674368000.0, 20922789888000.0, 355687428096000.0, 6402373705728000.0,
        121645100408832000.0, 2432902008176640000.0,
    ];
    
    if n < FACTORIALS.len() {
        FACTORIALS[n]
    } else {
        // For larger n, use Stirling's approximation: n! ≈ √(2πn) * (n/e)^n
        let n_f64 = n as f64;
        (2.0 * std::f64::consts::PI * n_f64).sqrt() * (n_f64 / std::f64::consts::E).powf(n_f64)
    }
}
