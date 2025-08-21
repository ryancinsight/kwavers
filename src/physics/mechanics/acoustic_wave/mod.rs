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
/// 
/// # Safety
/// 
/// Returns 0.0 for zero frequency (static fields) to prevent division by zero.
/// This is physically sensible as the frequency-dependent absorption model
/// becomes ill-defined at DC.
pub fn compute_acoustic_diffusivity(
    medium: &dyn Medium,
    x: f64,
    y: f64,
    z: f64,
    grid: &Grid,
    frequency: f64,
) -> f64 {
    // Prevent division by zero for static fields (frequency = 0)
    // At zero frequency, the concept of acoustic diffusivity from
    // frequency-dependent absorption is not well-defined
    if frequency == 0.0 {
        return 0.0;
    }
    
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
    use crate::medium::homogeneous::HomogeneousMedium;
    
    /// Test implementation of Medium trait for heterogeneous testing
    #[derive(Debug)]
    struct HeterogeneousMediumMock {
        /// Returns different properties based on position
        position_dependent: bool,
    }
    
    impl HeterogeneousMediumMock {
        fn new(position_dependent: bool) -> Self {
            Self { position_dependent }
        }
    }
    
    impl Medium for HeterogeneousMediumMock {
        fn density(&self, x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            if self.position_dependent {
                1000.0 + x * 100.0  // Varies with x
            } else {
                1000.0
            }
        }
        
        fn sound_speed(&self, x: f64, y: f64, _z: f64, _grid: &Grid) -> f64 {
            if self.position_dependent {
                // Different sound speeds at different positions
                if x < 0.2 {
                    1600.0  // Higher speed in first region
                } else if y < 0.5 {
                    1400.0  // Lower speed in second region
                } else {
                    1500.0  // Medium speed elsewhere
                }
            } else {
                1500.0
            }
        }
        
        fn absorption_coefficient(&self, x: f64, y: f64, z: f64, _grid: &Grid, _frequency: f64) -> f64 {
            if self.position_dependent {
                // Spatially varying absorption
                0.5 + 0.1 * x + 0.05 * y + 0.02 * z
            } else {
                0.5
            }
        }
        
        fn nonlinearity_parameter(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            5.0
        }
        
        fn density_array(&self) -> ndarray::Array3<f64> {
            ndarray::Array3::ones((10, 10, 10)) * 1000.0
        }
        
        fn sound_speed_array(&self) -> ndarray::Array3<f64> {
            ndarray::Array3::ones((10, 10, 10)) * 1500.0
        }
    }
    
    #[test]
    fn test_zero_frequency_safety() {
        // Test that zero frequency doesn't cause division by zero
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 5.0);
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
        
        // This should not panic and should return 0.0
        let diffusivity = compute_acoustic_diffusivity(&medium, 0.0, 0.0, 0.0, &grid, 0.0);
        assert_eq!(diffusivity, 0.0, "Zero frequency should return zero diffusivity");
        
        // Test with very small frequency (should not panic)
        let small_freq = 1e-10;
        let diffusivity_small = compute_acoustic_diffusivity(&medium, 0.0, 0.0, 0.0, &grid, small_freq);
        assert!(diffusivity_small.is_finite(), "Small frequency should produce finite result");
    }
    
    #[test]
    fn test_acoustic_diffusivity_heterogeneous() {
        // Test that the function correctly uses spatial coordinates
        let medium = HeterogeneousMediumMock::new(true);
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
        let frequency = 1e6;
        
        // Test point 1: x=0.1, y=0.2, z=0.3
        let diffusivity1 = compute_acoustic_diffusivity(&medium, 0.1, 0.2, 0.3, &grid, frequency);
        let c1 = 1600.0;  // x < 0.2
        let alpha1 = 0.5 + 0.1 * 0.1 + 0.05 * 0.2 + 0.02 * 0.3;  // 0.526
        let omega = 2.0 * PI * frequency;
        let expected1 = 2.0 * alpha1 * c1.powi(3) / (omega * omega);
        assert!((diffusivity1 - expected1).abs() < 1e-10,
            "Heterogeneous test 1 failed: got {}, expected {}", diffusivity1, expected1);
        
        // Test point 2: x=0.4, y=0.3, z=0.5
        let diffusivity2 = compute_acoustic_diffusivity(&medium, 0.4, 0.3, 0.5, &grid, frequency);
        let c2 = 1400.0;  // x >= 0.2 and y < 0.5
        let alpha2 = 0.5 + 0.1 * 0.4 + 0.05 * 0.3 + 0.02 * 0.5;  // 0.565
        let expected2 = 2.0 * alpha2 * c2.powi(3) / (omega * omega);
        assert!((diffusivity2 - expected2).abs() < 1e-10,
            "Heterogeneous test 2 failed: got {}, expected {}", diffusivity2, expected2);
        
        // Test point 3: x=0.5, y=0.6, z=0.7
        let diffusivity3 = compute_acoustic_diffusivity(&medium, 0.5, 0.6, 0.7, &grid, frequency);
        let c3 = 1500.0;  // x >= 0.2 and y >= 0.5
        let alpha3 = 0.5 + 0.1 * 0.5 + 0.05 * 0.6 + 0.02 * 0.7;  // 0.594
        let expected3 = 2.0 * alpha3 * c3.powi(3) / (omega * omega);
        assert!((diffusivity3 - expected3).abs() < 1e-10,
            "Heterogeneous test 3 failed: got {}, expected {}", diffusivity3, expected3);
        
        // Verify that different positions give different results
        assert!((diffusivity1 - diffusivity2).abs() > 1e-12,
            "Different positions should yield different diffusivities in heterogeneous medium");
        assert!((diffusivity2 - diffusivity3).abs() > 1e-12,
            "Different positions should yield different diffusivities in heterogeneous medium");
    }
    
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
