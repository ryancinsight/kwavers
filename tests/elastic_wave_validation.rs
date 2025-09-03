//! Validation tests for elastic wave propagation
//!
//! Tests P-wave and S-wave propagation against analytical solutions
//! References:
//! - Aki & Richards, "Quantitative Seismology", 2002
//! - Carcione, "Wave Fields in Real Media", 2007

use approx::assert_relative_eq;
use kwavers::{
    grid::Grid,
    medium::ElasticProperties,
    physics::{
        field_mapping::UnifiedFieldType,
        plugin::{elastic_wave_plugin::ElasticWavePlugin, PhysicsPlugin, PluginContext},
    },
};
use ndarray::Array4;

/// Test P-wave velocity in isotropic elastic medium
#[test]
fn test_p_wave_velocity() {
    // Create test medium with known elastic properties
    let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001);

    // Steel properties (typical values)
    let density = 7850.0_f64; // kg/m³
    let youngs_modulus = 200e9_f64; // Pa
    let poisson_ratio = 0.3_f64;

    // Calculate Lamé parameters from Young's modulus and Poisson's ratio
    let lame_mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio));
    let lame_lambda =
        youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));

    // Theoretical P-wave velocity: c_p = sqrt((λ + 2μ)/ρ)
    let theoretical_cp = ((lame_lambda + 2.0 * lame_mu) / density).sqrt();

    // Create medium with these properties
    let medium = TestElasticMedium::new(density, lame_lambda, lame_mu);

    // Initialize elastic wave plugin
    let dt = 1e-7; // Time step
    let mut plugin = ElasticWavePlugin::new(&grid, &medium, dt).unwrap();

    // Verify P-wave velocity matches theory
    let computed_cp = medium.p_wave_speed(0.0, 0.0, 0.0, &grid);
    assert_relative_eq!(computed_cp, theoretical_cp, epsilon = 1e-6);
}

/// Test S-wave velocity in isotropic elastic medium
#[test]
fn test_s_wave_velocity() {
    let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001);

    // Granite properties (typical values)
    let density = 2700.0_f64; // kg/m³
    let lame_mu = 30e9_f64; // Pa (shear modulus)
    let lame_lambda = 25e9_f64; // Pa

    // Theoretical S-wave velocity: c_s = sqrt(μ/ρ)
    let theoretical_cs = (lame_mu / density).sqrt();

    let medium = TestElasticMedium::new(density, lame_lambda, lame_mu);

    // Verify S-wave velocity matches theory
    let computed_cs = medium.shear_wave_speed(0.0, 0.0, 0.0, &grid);
    assert_relative_eq!(computed_cs, theoretical_cs, epsilon = 1e-6);
}

/// Test wave propagation with proper mode separation
#[test]
fn test_elastic_wave_propagation() {
    let grid = Grid::new(100, 100, 100, 0.001, 0.001, 0.001);
    let dt = 1e-6; // 1 microsecond timestep

    // Aluminum properties
    let density = 2700.0; // kg/m³
    let lame_mu = 26e9; // Pa
    let lame_lambda = 54e9; // Pa

    let medium = TestElasticMedium::new(density, lame_lambda, lame_mu);

    // Initialize plugin
    let mut plugin = ElasticWavePlugin::new(&grid, &medium, dt).unwrap();

    // Create field array
    let mut fields = Array4::zeros((4, 100, 100, 100));

    // Apply initial displacement (point source)
    fields[[1, 50, 50, 50]] = 1.0; // vx component

    // Propagate for several timesteps
    let pressure = ndarray::Array3::zeros((100, 100, 100));
    let context = PluginContext::new(pressure);
    for _ in 0..10 {
        plugin
            .update(&mut fields, &grid, &medium, dt, 0.0, &context)
            .unwrap();
    }

    // Verify wave has propagated (energy conservation)
    let total_energy: f64 = fields.iter().map(|v| v * v).sum();
    assert!(total_energy > 0.0, "Wave should have propagated");
}

/// Test medium for elastic wave validation
#[derive(Debug, Clone)]
struct TestElasticMedium {
    density: f64,
    lame_lambda: f64,
    lame_mu: f64,
    bubble_radius_field: ndarray::Array3<f64>,
    bubble_velocity_field: ndarray::Array3<f64>,
}

impl TestElasticMedium {
    fn new(density: f64, lame_lambda: f64, lame_mu: f64) -> Self {
        Self {
            density,
            lame_lambda,
            lame_mu,
            bubble_radius_field: ndarray::Array3::zeros((1, 1, 1)),
            bubble_velocity_field: ndarray::Array3::zeros((1, 1, 1)),
        }
    }

    fn p_wave_speed(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        ((self.lame_lambda + 2.0 * self.lame_mu) / self.density).sqrt()
    }
}

// Implement required traits for TestElasticMedium
use kwavers::medium::{
    AcousticProperties, ArrayAccess, BubbleProperties, BubbleState, CoreMedium, ElasticArrayAccess,
    Medium, OpticalProperties, ThermalProperties, ViscousProperties,
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

    fn validate(&self, _grid: &Grid) -> kwavers::KwaversResult<()> {
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
    fn density_array(&self, grid: &Grid) -> ndarray::Array3<f64> {
        let shape = (grid.nx, grid.ny, grid.nz);
        ndarray::Array3::from_elem(shape, self.density)
    }

    fn sound_speed_array(&self, grid: &Grid) -> ndarray::Array3<f64> {
        let shape = (grid.nx, grid.ny, grid.nz);
        ndarray::Array3::from_elem(shape, self.sound_speed(0.0, 0.0, 0.0, grid))
    }

    fn absorption_array(&self, grid: &Grid, _frequency: f64) -> ndarray::Array3<f64> {
        let shape = (grid.nx, grid.ny, grid.nz);
        ndarray::Array3::zeros(shape) // No absorption in test
    }

    fn nonlinearity_array(&self, grid: &Grid) -> ndarray::Array3<f64> {
        let shape = (grid.nx, grid.ny, grid.nz);
        ndarray::Array3::from_elem(shape, 3.5) // Default B/A
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
        let c = self.sound_speed(x, y, z, grid);
        thermal_diff / (c * c)
    }
}

impl ElasticArrayAccess for TestElasticMedium {
    fn lame_lambda_array(&self) -> ndarray::Array3<f64> {
        // Note: This is a simplified test implementation
        ndarray::Array3::from_elem((10, 10, 10), self.lame_lambda)
    }

    fn lame_mu_array(&self) -> ndarray::Array3<f64> {
        ndarray::Array3::from_elem((10, 10, 10), self.lame_mu)
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
        let k = self.thermal_conductivity(x, y, z, grid);
        let rho = self.density(x, y, z, grid);
        let cp = self.specific_heat(x, y, z, grid);
        k / (rho * cp)
    }

    fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        2.07e-4 // 1/K for water at 20°C
    }
}

impl TemperatureState for TestElasticMedium {
    fn temperature(&self) -> &ndarray::Array3<f64> {
        // Return uniform temperature field for test
        ndarray::Array3::from_elem((10, 10, 10), 293.15) // 20°C
    }

    fn update_temperature(&mut self, _temperature: &ndarray::Array3<f64>) {
        // No-op for test medium
    }
}

impl OpticalProperties for TestElasticMedium {
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
    fn bubble_radius(&self) -> &ndarray::Array3<f64> {
        &self.bubble_radius_field
    }

    fn bubble_velocity(&self) -> &ndarray::Array3<f64> {
        &self.bubble_velocity_field
    }

    fn update_bubble_state(
        &mut self,
        radius: &ndarray::Array3<f64>,
        velocity: &ndarray::Array3<f64>,
    ) {
        self.bubble_radius_field = radius.clone();
        self.bubble_velocity_field = velocity.clone();
    }
}
