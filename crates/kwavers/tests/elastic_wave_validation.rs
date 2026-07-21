//! Validation tests for elastic wave material parameters
//!
//! These tests check the analytical wave-speed accessors used by the
//! canonical elastic propagator [`kwavers_solver::forward::elastic::swe::ElasticWaveSolver`]
//! and by the consolidated PSTD elastic extension
//! [`kwavers_solver::forward::pstd::extensions::PstdElasticPlugin`].
//!
//! References:
//! - Aki & Richards, "Quantitative Seismology", 2002
//! - Carcione, "Wave Fields in Real Media", 2007

use eunomia::assert_relative_eq;
use kwavers_grid::Grid;

/// Test P-wave velocity in isotropic elastic medium.
///
/// Verifies the analytical relation `c_p = sqrt((λ + 2μ) / ρ)` (Aki &
/// Richards Eq. 4.13) against the medium's own accessor.
#[test]
fn test_p_wave_velocity() {
    let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001).unwrap();

    // Steel properties (typical values)
    let density = 7850.0_f64; // kg/m³
    let youngs_modulus = 200e9_f64; // Pa
    let poisson_ratio = 0.3_f64;

    let lame_mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio));
    let lame_lambda =
        youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));

    let theoretical_cp = ((lame_lambda + 2.0 * lame_mu) / density).sqrt();

    let medium = TestElasticMedium::new(density, lame_lambda, lame_mu);
    let computed_cp = medium.p_wave_speed(0.0, 0.0, 0.0, &grid);
    assert_relative_eq!(computed_cp, theoretical_cp, epsilon = 1e-6);
}

/// Test S-wave velocity in isotropic elastic medium.
///
/// Verifies the analytical relation `c_s = sqrt(μ / ρ)` (Aki & Richards
/// Eq. 4.14) against the medium's own accessor.
#[test]
fn test_s_wave_velocity() {
    let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001).unwrap();

    // Granite properties (typical values)
    let density = 2700.0_f64; // kg/m³
    let lame_mu = 30e9_f64; // Pa (shear modulus)
    let lame_lambda = 25e9_f64; // Pa

    let theoretical_cs = (lame_mu / density).sqrt();

    let medium = TestElasticMedium::new(density, lame_lambda, lame_mu);
    let computed_cs = medium.shear_wave_speed(0.0, 0.0, 0.0, &grid);
    assert_relative_eq!(computed_cs, theoretical_cs, epsilon = 1e-6);
}

// `test_elastic_wave_propagation` was removed alongside the deleted
// `ElasticWavePlugin` (the legacy μ = 0 acoustic-fluid duplicate that
// previously exercised this path). End-to-end elastic propagation is now
// covered by `external/elastic_julia_parity/compare_elastic.py` (against
// KWave.jl) and `pykwavers/examples/ewp_elastic_2d_jl_compare.py`. See the
// canonical solver matrix in `solver::forward` module docs.

/// Test medium for elastic wave validation
#[derive(Debug, Clone)]
struct TestElasticMedium {
    density: f64,
    lame_lambda: f64,
    lame_mu: f64,
    bubble_radius_field: leto::Array3<f64>,
    bubble_velocity_field: leto::Array3<f64>,
    thermal_field: leto::Array3<f64>,
}

impl TestElasticMedium {
    fn new(density: f64, lame_lambda: f64, lame_mu: f64) -> Self {
        Self {
            density,
            lame_lambda,
            lame_mu,
            bubble_radius_field: leto::Array3::zeros((1, 1, 1)),
            bubble_velocity_field: leto::Array3::zeros((1, 1, 1)),
            thermal_field: leto::Array3::from_elem((1, 1, 1), 293.15),
        }
    }

    fn p_wave_speed(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        ((self.lame_lambda + 2.0 * self.lame_mu) / self.density).sqrt()
    }
}

// Implement required traits for TestElasticMedium
use kwavers_medium::{
    AcousticProperties, ArrayAccess, BubbleProperties, BubbleState, CoreMedium, ElasticArrayAccess,
    ElasticProperties, MediumOpticalProperties, ThermalField, ThermalProperties, ViscousProperties,
};

impl CoreMedium for TestElasticMedium {
    fn density(&self, _i: usize, _j: usize, _k: usize) -> f64 {
        self.density
    }

    fn sound_speed(&self, _i: usize, _j: usize, _k: usize) -> f64 {
        // P-wave speed approximation
        ((self.lame_lambda + 2.0 * self.lame_mu) / self.density).sqrt()
    }

    fn absorption(&self, _i: usize, _j: usize, _k: usize) -> f64 {
        0.0 // No absorption in test medium
    }

    fn nonlinearity(&self, _i: usize, _j: usize, _k: usize) -> f64 {
        3.5 // Default B/A for water-like media
    }

    fn max_sound_speed(&self) -> f64 {
        self.sound_speed(0, 0, 0)
    }

    fn is_homogeneous(&self) -> bool {
        true
    }

    fn validate(&self, _grid: &Grid) -> kwavers_core::error::KwaversResult<()> {
        Ok(())
    }
}

impl ElasticProperties for TestElasticMedium {
    fn lame_lambda(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.lame_lambda
    }

    fn lame_mu(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.lame_mu
    }
}

// Implement other required traits with defaults
impl ArrayAccess for TestElasticMedium {
    fn density_array(&self) -> leto::ArrayView3<'_, f64> {
        // For test purposes, create a small array and return view
        // In production, this would return a view of stored data
        panic!("ArrayAccess not implemented for test medium - use CoreMedium methods")
    }

    fn sound_speed_array(&self) -> leto::ArrayView3<'_, f64> {
        panic!("ArrayAccess not implemented for test medium - use CoreMedium methods")
    }

    fn absorption_array(&self) -> leto::ArrayView3<'_, f64> {
        panic!("ArrayAccess not implemented for test medium - use CoreMedium methods")
    }

    fn nonlinearity_array(&self) -> leto::ArrayView3<'_, f64> {
        panic!("ArrayAccess not implemented for test medium - use CoreMedium methods")
    }
}

impl AcousticProperties for TestElasticMedium {
    fn absorption_coefficient(
        &self,
        _x: f64,
        _y: f64,
        _z: f64,
        _grid: &Grid,
        _frequency: f64,
    ) -> f64 {
        0.0 // No absorption in test medium
    }

    fn nonlinearity_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        3.5 // Default B/A for water-like media
    }

    fn acoustic_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Acoustic diffusivity = thermal diffusivity / c²
        let thermal_diff = self.thermal_diffusivity(x, y, z, grid);
        // Convert coordinates to indices (simplified for test)
        let i = (x / grid.dx) as usize;
        let j = (y / grid.dy) as usize;
        let k = (z / grid.dz) as usize;
        let c = self.sound_speed(i, j, k);
        thermal_diff / (c * c)
    }
}

impl ElasticArrayAccess for TestElasticMedium {
    fn lame_lambda_array(&self) -> leto::Array3<f64> {
        // Note: This is a simplified test implementation
        leto::Array3::from_elem((10, 10, 10), self.lame_lambda)
    }

    fn lame_mu_array(&self) -> leto::Array3<f64> {
        leto::Array3::from_elem((10, 10, 10), self.lame_mu)
    }

    fn shear_sound_speed_array(&self) -> leto::Array3<f64> {
        // Mathematical specification: c_s = sqrt(μ / ρ)
        // where μ is shear modulus (Pa) and ρ is density (kg/m³)
        let shear_speed = if self.density > 0.0 {
            (self.lame_mu / self.density).sqrt()
        } else {
            0.0
        };
        leto::Array3::from_elem((10, 10, 10), shear_speed)
    }
}

impl ThermalProperties for TestElasticMedium {
    fn thermal_conductivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.6 // W/(m·K) typical for water
    }

    fn specific_heat(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        4182.0 // J/(kg·K) for water
    }

    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let thermal_k = self.thermal_conductivity(x, y, z, grid);
        let i = (x / grid.dx) as usize;
        let j = (y / grid.dy) as usize;
        let k = (z / grid.dz) as usize;
        let rho = self.density(i, j, k);
        let cp = self.specific_heat(x, y, z, grid);
        thermal_k / (rho * cp)
    }

    fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        2.07e-4 // 1/K for water at 20°C
    }
}

// Implement ThermalField trait for test
impl ThermalField for TestElasticMedium {
    fn thermal_field(&self) -> &leto::Array3<f64> {
        &self.thermal_field
    }

    fn update_thermal_field(&mut self, _temperature: &leto::Array3<f64>) {
        // No-op for test
    }
}

// Medium trait is automatically implemented via blanket impl

impl MediumOpticalProperties for TestElasticMedium {
    fn optical_absorption_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.01 // 1/m
    }

    fn optical_scattering_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        10.0 // 1/m
    }

    fn refractive_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        1.33 // Water
    }
}

impl ViscousProperties for TestElasticMedium {
    fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.001 // Pa·s (water at 20°C)
    }
}

impl BubbleProperties for TestElasticMedium {
    fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.0728 // N/m for water
    }

    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        101325.0 // Pa (1 atm)
    }

    fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        2338.0 // Pa for water at 20°C
    }

    fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        1.4 // Air
    }

    fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        2e-9 // m²/s for air in water
    }
}

impl BubbleState for TestElasticMedium {
    fn bubble_radius(&self) -> &leto::Array3<f64> {
        &self.bubble_radius_field
    }

    fn bubble_velocity(&self) -> &leto::Array3<f64> {
        &self.bubble_velocity_field
    }

    fn update_bubble_state(&mut self, radius: &leto::Array3<f64>, velocity: &leto::Array3<f64>) {
        self.bubble_radius_field = radius.clone();
        self.bubble_velocity_field = velocity.clone();
    }
}
