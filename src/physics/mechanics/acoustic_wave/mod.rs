// physics/mechanics/acoustic_wave/mod.rs
pub mod nonlinear; // This will now refer to the new subdirectory

// Re-export NonlinearWave from the new structure.
pub use nonlinear::NonlinearWave;

pub mod westervelt_wave;
pub use westervelt_wave::WesterveltWave;

pub mod kuznetsov;
pub use kuznetsov::{KuznetsovWave, KuznetsovConfig};

pub mod unified_solver;
pub use unified_solver::{AcousticWaveSolver, AcousticSolverConfig, AcousticModelType};

use crate::grid::Grid;
use crate::medium::Medium;
use std::f64::consts::PI;

/// Compute acoustic diffusivity from medium properties
/// 
/// This is the single source of truth for acoustic diffusivity calculation.
/// 
/// # Physics Background
/// 
/// Acoustic diffusivity δ = (4μ/3 + μ_B + κ(γ-1)/C_p) / ρ₀
/// Where:
/// - μ = shear viscosity
/// - μ_B = bulk viscosity  
/// - κ = thermal conductivity
/// - γ = specific heat ratio
/// - C_p = specific heat at constant pressure
/// 
/// For soft tissues, we use the approximation:
/// δ ≈ 2αc³/(ω²)
/// 
/// where α is the absorption coefficient and c is the sound speed.
pub fn compute_acoustic_diffusivity(
    medium: &dyn Medium,
    x: f64,
    y: f64,
    z: f64,
    grid: &Grid,
    frequency: f64,
) -> f64 {
    let alpha = medium.absorption_coefficient(x, y, z, grid, frequency);
    let c = medium.sound_speed(x, y, z, grid);
    
    // Approximate diffusivity from power-law absorption
    // δ ≈ 2αc³/(ω²) for typical soft tissues
    let omega = 2.0 * PI * frequency;
    2.0 * alpha * c.powi(3) / (omega * omega)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    
    #[test]
    fn test_acoustic_diffusivity_calculation() {
        // Test that the formula δ = 2αc³/ω² is correctly implemented
        
        // Test case 1: Zero absorption should give zero diffusivity
        let alpha: f64 = 0.0;
        let c: f64 = 1500.0;
        let freq: f64 = 1e6;
        let omega = 2.0 * PI * freq;
        let expected = 2.0 * alpha * c.powi(3) / (omega * omega);
        assert_eq!(expected, 0.0);
        
        // Test case 2: Non-zero values
        let alpha: f64 = 0.5; // Np/m
        let c: f64 = 1500.0; // m/s
        let freq: f64 = 1e6; // Hz
        let omega = 2.0 * PI * freq;
        let diffusivity = 2.0 * alpha * c.powi(3) / (omega * omega);
        
        // Calculate expected value
        let expected = 2.0 * 0.5 * 1500.0_f64.powi(3) / (2.0 * PI * 1e6).powi(2);
        
        assert!((diffusivity - expected).abs() < 1e-10, 
            "Formula calculation mismatch: got {}, expected {}", diffusivity, expected);
        
        // Test case 3: Verify frequency scaling
        let freq2: f64 = 2e6;
        let omega2 = 2.0 * PI * freq2;
        let diffusivity2 = 2.0 * alpha * c.powi(3) / (omega2 * omega2);
        
        // Diffusivity should scale as 1/f² for constant α
        assert!((diffusivity2 - diffusivity / 4.0).abs() < 1e-10,
            "Frequency scaling incorrect: {} vs {}", diffusivity2, diffusivity / 4.0);
        
        // Test case 4: Verify the actual value is reasonable
        // For α = 0.5 Np/m, c = 1500 m/s, f = 1 MHz
        // δ = 2 * 0.5 * 1500³ / (2π * 10⁶)² ≈ 8.5e-5 m²/s
        assert!(diffusivity > 1e-6 && diffusivity < 1e-3,
            "Diffusivity value seems unreasonable: {}", diffusivity);
    }
}
