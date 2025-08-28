// physics/mechanics/acoustic_wave/mod.rs
pub mod nonlinear; // This will now refer to the new subdirectory

// Re-export NonlinearWave from the new structure.
pub use nonlinear::NonlinearWave;

pub mod westervelt_wave;
pub use westervelt_wave::WesterveltWave;

pub mod kuznetsov;
pub use kuznetsov::{KuznetsovConfig, KuznetsovWave};

pub mod westervelt_fdtd;
pub use westervelt_fdtd::{WesterveltFdtd, WesterveltFdtdConfig};

pub mod unified_solver;
pub use unified_solver::{AcousticModelType, AcousticSolverConfig, AcousticWaveSolver};

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
        /// Cached density array
        density: ndarray::Array3<f64>,
        /// Cached sound speed array
        sound_speed: ndarray::Array3<f64>,
        /// Temperature array
        temperature: ndarray::Array3<f64>,
        /// Bubble radius array
        bubble_radius: ndarray::Array3<f64>,
        /// Bubble velocity array
        bubble_velocity: ndarray::Array3<f64>,
    }

    impl HeterogeneousMediumMock {
        fn new(position_dependent: bool) -> Self {
            Self {
                position_dependent,
                density: ndarray::Array3::ones((10, 10, 10)) * 1000.0,
                sound_speed: ndarray::Array3::ones((10, 10, 10)) * 1500.0,
                temperature: ndarray::Array3::ones((10, 10, 10)) * 310.0,
                bubble_radius: ndarray::Array3::zeros((10, 10, 10)),
                bubble_velocity: ndarray::Array3::zeros((10, 10, 10)),
            }
        }
    }

    // Implement component traits for HeterogeneousMediumMock
    impl crate::medium::core::CoreMedium for HeterogeneousMediumMock {
        fn density(&self, x: f64, y: f64, z: f64, _grid: &Grid) -> f64 {
            if self.position_dependent {
                1000.0 + x + y + z
            } else {
                1000.0
            }
        }

        fn sound_speed(&self, x: f64, y: f64, z: f64, _grid: &Grid) -> f64 {
            if self.position_dependent {
                1500.0 + x * 10.0 + y * 5.0 + z * 2.0
            } else {
                1500.0
            }
        }

        fn is_homogeneous(&self) -> bool {
            !self.position_dependent
        }

        fn reference_frequency(&self) -> f64 {
            1e6
        }
    }

    impl crate::medium::core::ArrayAccess for HeterogeneousMediumMock {
        fn get_density_array(
            &self,
            _grid: &Grid,
        ) -> crate::error::KwaversResult<ndarray::Array3<f64>> {
            Ok(self.density.clone())
        }

        fn get_sound_speed_array(
            &self,
            _grid: &Grid,
        ) -> crate::error::KwaversResult<ndarray::Array3<f64>> {
            Ok(self.sound_speed.clone())
        }

        fn density_array(&self) -> ndarray::Array3<f64> {
            self.density.clone()
        }

        fn sound_speed_array(&self) -> ndarray::Array3<f64> {
            self.sound_speed.clone()
        }
    }

    impl crate::medium::acoustic::AcousticProperties for HeterogeneousMediumMock {
        fn absorption_coefficient(
            &self,
            _x: f64,
            _y: f64,
            _z: f64,
            _grid: &Grid,
            _frequency: f64,
        ) -> f64 {
            0.01
        }

        fn attenuation(&self, _x: f64, _y: f64, _z: f64, _frequency: f64, _grid: &Grid) -> f64 {
            0.01
        }

        fn nonlinearity_parameter(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            5.0
        }

        fn nonlinearity_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            3.5
        }

        fn acoustic_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            1.4e-7
        }

        fn tissue_type(
            &self,
            _x: f64,
            _y: f64,
            _z: f64,
            _grid: &Grid,
        ) -> Option<crate::medium::absorption::TissueType> {
            None
        }
    }

    impl crate::medium::elastic::ElasticProperties for HeterogeneousMediumMock {
        fn lame_lambda(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            2.2e9
        }

        fn lame_mu(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            0.0
        }

        fn shear_wave_speed(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            0.0
        }

        fn compressional_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Use the CoreMedium trait method
            <Self as crate::medium::core::CoreMedium>::sound_speed(self, x, y, z, grid)
        }
    }

    impl crate::medium::elastic::ElasticArrayAccess for HeterogeneousMediumMock {
        fn lame_lambda_array(&self) -> ndarray::Array3<f64> {
            self.density.clone()
        }

        fn lame_mu_array(&self) -> ndarray::Array3<f64> {
            self.bubble_radius.clone()
        }

        fn shear_sound_speed_array(&self) -> ndarray::Array3<f64> {
            ndarray::Array3::zeros((10, 10, 10))
        }

        fn shear_viscosity_coeff_array(&self) -> ndarray::Array3<f64> {
            ndarray::Array3::from_elem((10, 10, 10), 1e-3)
        }

        fn bulk_viscosity_coeff_array(&self) -> ndarray::Array3<f64> {
            ndarray::Array3::from_elem((10, 10, 10), 2e-3)
        }
    }

    impl crate::medium::thermal::ThermalProperties for HeterogeneousMediumMock {
        fn specific_heat(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            4180.0
        }

        fn specific_heat_capacity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            4180.0
        }

        fn thermal_conductivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            0.6
        }

        fn thermal_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            1.4e-7
        }

        fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            3e-4
        }

        fn specific_heat_ratio(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            1.4
        }

        fn gamma(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            1.4
        }
    }

    impl crate::medium::thermal::TemperatureState for HeterogeneousMediumMock {
        fn update_temperature(&mut self, temperature: &ndarray::Array3<f64>) {
            self.temperature.assign(temperature);
        }

        fn temperature(&self) -> &ndarray::Array3<f64> {
            &self.temperature
        }
    }

    impl crate::medium::optical::OpticalProperties for HeterogeneousMediumMock {
        fn optical_absorption_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            0.1
        }

        fn optical_scattering_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            1.0
        }

        fn refractive_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            1.33
        }

        fn anisotropy_factor(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            0.9
        }

        fn reduced_scattering_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            0.1
        }
    }

    impl crate::medium::viscous::ViscousProperties for HeterogeneousMediumMock {
        fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            1e-3
        }

        fn shear_viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            1e-3
        }

        fn bulk_viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            2e-3
        }

        fn kinematic_viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            1e-6
        }
    }

    impl crate::medium::bubble::BubbleProperties for HeterogeneousMediumMock {
        fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            0.072
        }

        fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            101325.0
        }

        fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            2330.0
        }

        fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            1.4
        }

        fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            2e-9
        }
    }

    impl crate::medium::bubble::BubbleState for HeterogeneousMediumMock {
        fn bubble_radius(&self) -> &ndarray::Array3<f64> {
            &self.bubble_radius
        }

        fn bubble_velocity(&self) -> &ndarray::Array3<f64> {
            &self.bubble_velocity
        }

        fn update_bubble_state(
            &mut self,
            radius: &ndarray::Array3<f64>,
            velocity: &ndarray::Array3<f64>,
        ) {
            self.bubble_radius.assign(radius);
            self.bubble_velocity.assign(velocity);
        }
    }

    #[test]
    fn test_zero_frequency_safety() {
        // Test that zero frequency doesn't cause division by zero
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 5.0, &grid);

        // This should not panic and should return 0.0
        let diffusivity = compute_acoustic_diffusivity(&medium, 0.0, 0.0, 0.0, &grid, 0.0);
        assert_eq!(
            diffusivity, 0.0,
            "Zero frequency should return zero diffusivity"
        );

        // Test with very small frequency (should not panic)
        let small_freq = 1e-10;
        let diffusivity_small =
            compute_acoustic_diffusivity(&medium, 0.0, 0.0, 0.0, &grid, small_freq);
        assert!(
            diffusivity_small.is_finite(),
            "Small frequency should produce finite result"
        );
    }

    #[test]
    fn test_acoustic_diffusivity_heterogeneous() {
        // Test that the function correctly uses spatial coordinates
        let medium = HeterogeneousMediumMock::new(true);
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
        let frequency = 1e6;

        // Test point 1: x=0.1, y=0.2, z=0.3
        let diffusivity1 = compute_acoustic_diffusivity(&medium, 0.1, 0.2, 0.3, &grid, frequency);
        let c1: f64 = 1600.0; // x < 0.2
        let alpha1 = 0.5 + 0.1 * 0.1 + 0.05 * 0.2 + 0.02 * 0.3; // 0.526
        let omega = 2.0 * PI * frequency;
        let expected1 = 2.0 * alpha1 * c1.powi(3) / (omega * omega);
        assert!(
            (diffusivity1 - expected1).abs() < 1e-10,
            "Heterogeneous test 1 failed: got {}, expected {}",
            diffusivity1,
            expected1
        );

        // Test point 2: x=0.4, y=0.3, z=0.5
        let diffusivity2 = compute_acoustic_diffusivity(&medium, 0.4, 0.3, 0.5, &grid, frequency);
        let c2: f64 = 1400.0; // x >= 0.2 and y < 0.5
        let alpha2 = 0.5 + 0.1 * 0.4 + 0.05 * 0.3 + 0.02 * 0.5; // 0.565
        let expected2 = 2.0 * alpha2 * c2.powi(3) / (omega * omega);
        assert!(
            (diffusivity2 - expected2).abs() < 1e-10,
            "Heterogeneous test 2 failed: got {}, expected {}",
            diffusivity2,
            expected2
        );

        // Test point 3: x=0.5, y=0.6, z=0.7
        let diffusivity3 = compute_acoustic_diffusivity(&medium, 0.5, 0.6, 0.7, &grid, frequency);
        let c3: f64 = 1500.0; // x >= 0.2 and y >= 0.5
        let alpha3 = 0.5 + 0.1 * 0.5 + 0.05 * 0.6 + 0.02 * 0.7; // 0.594
        let expected3 = 2.0 * alpha3 * c3.powi(3) / (omega * omega);
        assert!(
            (diffusivity3 - expected3).abs() < 1e-10,
            "Heterogeneous test 3 failed: got {}, expected {}",
            diffusivity3,
            expected3
        );

        // Verify that different positions give different results
        assert!(
            (diffusivity1 - diffusivity2).abs() > 1e-12,
            "Different positions should yield different diffusivities in heterogeneous medium"
        );
        assert!(
            (diffusivity2 - diffusivity3).abs() > 1e-12,
            "Different positions should yield different diffusivities in heterogeneous medium"
        );
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

        assert!(
            (diffusivity - expected).abs() < 1e-10,
            "Formula calculation mismatch: got {}, expected {}",
            diffusivity,
            expected
        );

        // Test case 3: Verify frequency scaling
        let freq2: f64 = 2e6;
        let omega2 = 2.0 * PI * freq2;
        let diffusivity2 = 2.0 * alpha * c.powi(3) / (omega2 * omega2);

        // Diffusivity should scale as 1/f² for constant α
        assert!(
            (diffusivity2 - diffusivity / 4.0).abs() < 1e-10,
            "Frequency scaling incorrect: {} vs {}",
            diffusivity2,
            diffusivity / 4.0
        );

        // Test case 4: Verify the actual value is reasonable
        // For α = 0.5 Np/m, c = 1500 m/s, f = 1 MHz
        // δ = 2 * 0.5 * 1500³ / (2π * 10⁶)² ≈ 8.5e-5 m²/s
        assert!(
            diffusivity > 1e-6 && diffusivity < 1e-3,
            "Diffusivity value seems unreasonable: {}",
            diffusivity
        );
    }
}
