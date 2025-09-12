//! Heterogeneous medium implementation with spatially varying properties

use crate::error::{KwaversError, KwaversResult, ValidationError};
use crate::grid::Grid;
use crate::medium::{
    absorption::PowerLawAbsorption,
    acoustic::AcousticProperties,
    bubble::{BubbleProperties, BubbleState},
    core::{ArrayAccess, CoreMedium},
    elastic::{ElasticArrayAccess, ElasticProperties},
    optical::OpticalProperties,
    thermal::{ThermalField, ThermalProperties},
    viscous::ViscousProperties,
};
use log::debug;
use ndarray::{Array3, ArrayView3, ArrayViewMut3};

// Import physical constants
use super::constants::{MIN_PHYSICAL_DENSITY, MIN_PHYSICAL_SOUND_SPEED};

/// Medium with spatially varying properties
///
/// Note: The Clone derive is kept but should be used sparingly due to the
/// large memory footprint of this struct. Consider using Arc for sharing.
#[derive(Debug, Clone)]
pub struct HeterogeneousMedium {
    /// Whether to use trilinear interpolation for point queries
    pub use_trilinear_interpolation: bool,
    pub density: Array3<f64>,
    pub sound_speed: Array3<f64>,
    pub viscosity: Array3<f64>,
    pub surface_tension: Array3<f64>,
    pub ambient_pressure: f64,
    pub vapor_pressure: Array3<f64>,
    pub polytropic_index: Array3<f64>,
    pub specific_heat: Array3<f64>,
    pub thermal_conductivity: Array3<f64>,
    pub thermal_expansion: Array3<f64>,
    pub gas_diffusion_coeff: Array3<f64>,
    pub thermal_diffusivity: Array3<f64>,
    pub mu_a: Array3<f64>,
    pub mu_s_prime: Array3<f64>,
    pub temperature: Array3<f64>,
    pub bubble_radius: Array3<f64>,
    pub bubble_velocity: Array3<f64>,
    pub alpha0: Array3<f64>,
    pub delta: Array3<f64>,
    pub b_a: Array3<f64>,
    pub absorption: Array3<f64>,
    pub nonlinearity: Array3<f64>,
    pub reference_frequency: f64,
    // Viscoelastic properties
    pub shear_sound_speed: Array3<f64>,
    pub shear_viscosity_coeff: Array3<f64>,
    pub bulk_viscosity_coeff: Array3<f64>,
    // Elastic properties
    pub lame_lambda: Array3<f64>,
    pub lame_mu: Array3<f64>,
}

impl HeterogeneousMedium {
    /// Helper to get grid indices with fallback to boundary values
    #[inline]
    fn get_indices(&self, x: f64, y: f64, z: f64, grid: &Grid) -> (usize, usize, usize) {
        grid.position_to_indices(x, y, z).unwrap_or_else(|| {
            // Clamp to grid boundaries if out of bounds
            let ix = ((x / grid.dx).floor() as usize).min(grid.nx - 1);
            let iy = ((y / grid.dy).floor() as usize).min(grid.ny - 1);
            let iz = ((z / grid.dz).floor() as usize).min(grid.nz - 1);
            (ix, iy, iz)
        })
    }

    /// Perform trilinear interpolation on a field
    #[allow(dead_code)]
    #[inline]
        fn trilinear_interpolate(
        &self,
        field: &Array3<f64>,
        x: f64,
        y: f64,
        z: f64,
        grid: &Grid,
    ) -> f64 {
        // Find the base index and fractional distances
        let x_pos = (x / grid.dx).max(0.0);
        let y_pos = (y / grid.dy).max(0.0);
        let z_pos = (z / grid.dz).max(0.0);

        let i = (x_pos.floor() as usize).min(grid.nx.saturating_sub(2));
        let j = (y_pos.floor() as usize).min(grid.ny.saturating_sub(2));
        let k = (z_pos.floor() as usize).min(grid.nz.saturating_sub(2));

        let dx = (x_pos - i as f64).clamp(0.0, 1.0);
        let dy = (y_pos - j as f64).clamp(0.0, 1.0);
        let dz = (z_pos - k as f64).clamp(0.0, 1.0);

        // Get the values at the 8 corner points of the cell
        let c000 = field[[i, j, k]];
        let c100 = field[[i + 1, j, k]];
        let c010 = field[[i, j + 1, k]];
        let c110 = field[[i + 1, j + 1, k]];
        let c001 = field[[i, j, k + 1]];
        let c101 = field[[i + 1, j, k + 1]];
        let c011 = field[[i, j + 1, k + 1]];
        let c111 = field[[i + 1, j + 1, k + 1]];

        // Perform trilinear interpolation
        // Interpolate along x
        let c00 = c000 * (1.0 - dx) + c100 * dx;
        let c10 = c010 * (1.0 - dx) + c110 * dx;
        let c01 = c001 * (1.0 - dx) + c101 * dx;
        let c11 = c011 * (1.0 - dx) + c111 * dx;

        // Interpolate along y
        let c0 = c00 * (1.0 - dy) + c10 * dy;
        let c1 = c01 * (1.0 - dy) + c11 * dy;

        // Interpolate along z
        c0 * (1.0 - dz) + c1 * dz
    }

    /// Get value from field using either nearest neighbor or trilinear interpolation
    #[allow(dead_code)]
    #[inline]
        fn get_field_value(&self, field: &Array3<f64>, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if self.use_trilinear_interpolation {
            self.trilinear_interpolate(field, x, y, z, grid)
        } else {
            let (ix, iy, iz) = self.get_indices(x, y, z, grid);
            field[[ix, iy, iz]]
        }
    }

    /// Create a heterogeneous tissue medium
    pub fn tissue(grid: &Grid) -> Self {
        let density = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1050.0);
        let sound_speed = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1540.0);
        let viscosity = Array3::from_elem((grid.nx, grid.ny, grid.nz), 2.5e-3);
        let surface_tension = Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.06);
        let ambient_pressure = 1.013e5;
        let vapor_pressure = Array3::from_elem((grid.nx, grid.ny, grid.nz), 2.338e3);
        let polytropic_index = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.4);
        let specific_heat = Array3::from_elem((grid.nx, grid.ny, grid.nz), 3630.0);
        let thermal_conductivity = Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.52);
        let thermal_expansion = Array3::from_elem((grid.nx, grid.ny, grid.nz), 3.0e-4);
        let gas_diffusion_coeff = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.8e-9);
        let thermal_diffusivity = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.35e-7);
        let mu_a = Array3::from_elem((grid.nx, grid.ny, grid.nz), 10.0);
        let mu_s_prime = Array3::from_elem((grid.nx, grid.ny, grid.nz), 100.0);
        let b_a = Array3::from_elem((grid.nx, grid.ny, grid.nz), 7.0);
        let reference_frequency = 180000.0;

        let temperature = Array3::from_elem((grid.nx, grid.ny, grid.nz), 310.15); // 37°C
        let bubble_radius = Array3::from_elem((grid.nx, grid.ny, grid.nz), 10e-6);
        let bubble_velocity = Array3::zeros((grid.nx, grid.ny, grid.nz));

        let alpha0 = Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.5);
        let delta = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.1);

        // Initialize viscoelastic fields with tissue-appropriate values
        let shear_sound_speed = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(i, j, _k)| {
            let base_speed = 3.0; // m/s (typical for muscle tissue)
            let variation =
                0.5 * ((i as f64 / grid.nx as f64).sin() + (j as f64 / grid.ny as f64).cos());
            (base_speed + variation).max(1.0).min(8.0)
        });

        let shear_viscosity_coeff =
            Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(i, j, k)| {
                let center_x = grid.nx as f64 / 2.0;
                let center_y = grid.ny as f64 / 2.0;
                let center_z = grid.nz as f64 / 2.0;
                let dist_from_center = ((i as f64 - center_x).powi(2)
                    + (j as f64 - center_y).powi(2)
                    + (k as f64 - center_z).powi(2))
                .sqrt();
                let max_dist = (center_x.powi(2) + center_y.powi(2) + center_z.powi(2)).sqrt();
                let normalized_dist = (dist_from_center / max_dist).min(1.0);
                1.0 + 2.0 * normalized_dist // Range: 1.0-3.0 Pa·s
            });

        let bulk_viscosity_coeff = shear_viscosity_coeff.mapv(|shear_visc| shear_visc * 3.0);

        // Initialize elastic fields
        let default_density: f64 = 1050.0;
        let default_sound_speed: f64 = 1540.0;
        let default_bulk_modulus = default_density * default_sound_speed.powi(2);
        let lame_lambda = Array3::from_elem((grid.nx, grid.ny, grid.nz), default_bulk_modulus);
        let lame_mu = Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.0);

        debug!(
            "Initialized HeterogeneousMedium: grid {}x{}x{}, freq = {:.2e}",
            grid.nx, grid.ny, grid.nz, reference_frequency
        );

        // Compute absorption and nonlinearity from parameters
        let freq_ratio: f64 = reference_frequency / 1e6;
        let absorption = alpha0.mapv(|a0| a0 * freq_ratio.powf(1.0));
        let nonlinearity = b_a.clone();

        Self {
            use_trilinear_interpolation: false, // Default to nearest neighbor for performance
            density,
            sound_speed,
            viscosity,
            surface_tension,
            ambient_pressure,
            vapor_pressure,
            polytropic_index,
            specific_heat,
            thermal_conductivity,
            thermal_expansion,
            gas_diffusion_coeff,
            thermal_diffusivity,
            mu_a,
            mu_s_prime,
            temperature,
            bubble_radius,
            bubble_velocity,
            alpha0,
            delta,
            b_a,
            absorption,
            nonlinearity,
            reference_frequency,
            shear_sound_speed,
            shear_viscosity_coeff,
            bulk_viscosity_coeff,
            lame_lambda,
            lame_mu,
        }
    }
}

// Core medium properties
impl CoreMedium for HeterogeneousMedium {
    fn density(&self, i: usize, j: usize, k: usize) -> f64 {
        self.density[[i, j, k]].max(MIN_PHYSICAL_DENSITY)
    }

    fn sound_speed(&self, i: usize, j: usize, k: usize) -> f64 {
        self.sound_speed[[i, j, k]].max(MIN_PHYSICAL_SOUND_SPEED)
    }

    fn reference_frequency(&self) -> f64 {
        self.reference_frequency
    }

    fn absorption(&self, i: usize, j: usize, k: usize) -> f64 {
        self.absorption[[i, j, k]]
    }

    fn nonlinearity(&self, i: usize, j: usize, k: usize) -> f64 {
        self.nonlinearity[[i, j, k]]
    }

    fn max_sound_speed(&self) -> f64 {
        crate::medium::max_sound_speed(&self.sound_speed)
    }

    fn is_homogeneous(&self) -> bool {
        false
    }

    fn validate(&self, grid: &Grid) -> KwaversResult<()> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let expected_shape = [nx, ny, nz];

        if self.density.shape() != expected_shape {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: format!("{expected_shape:?}"),
                    actual: format!("{:?}", self.density.shape()),
                },
            ));
        }

        Ok(())
    }
}

// Array-based access
impl ArrayAccess for HeterogeneousMedium {
    fn density_array(&self) -> ArrayView3<'_, f64> {
        self.density.view()
    }

    fn sound_speed_array(&self) -> ArrayView3<'_, f64> {
        self.sound_speed.view()
    }

    fn density_array_mut(&mut self) -> Option<ArrayViewMut3<'_, f64>> {
        Some(self.density.view_mut())
    }

    fn sound_speed_array_mut(&mut self) -> Option<ArrayViewMut3<'_, f64>> {
        Some(self.sound_speed.view_mut())
    }

    fn absorption_array(&self) -> ArrayView3<'_, f64> {
        self.absorption.view()
    }

    fn nonlinearity_array(&self) -> ArrayView3<'_, f64> {
        self.nonlinearity.view()
    }
}

// Acoustic properties
impl AcousticProperties for HeterogeneousMedium {
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        let absorption = PowerLawAbsorption {
            alpha_0: self.alpha0[[ix, iy, iz]],
            y: self.delta[[ix, iy, iz]],
            f_ref: self.reference_frequency,
            dispersion_correction: false,
        };
        absorption.absorption_at_frequency(frequency)
    }

    fn nonlinearity_parameter(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.b_a[[ix, iy, iz]]
    }

    fn acoustic_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        let thermal_diff = self.thermal_diffusivity[[ix, iy, iz]];
        let sound_speed = self.sound_speed[[ix, iy, iz]];
        thermal_diff / sound_speed
    }
}

// Elastic properties
impl ElasticProperties for HeterogeneousMedium {
    fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.lame_lambda[[ix, iy, iz]]
    }

    fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.lame_mu[[ix, iy, iz]]
    }
}

// Elastic array access
impl ElasticArrayAccess for HeterogeneousMedium {
    fn lame_lambda_array(&self) -> Array3<f64> {
        self.lame_lambda.clone()
    }

    fn lame_mu_array(&self) -> Array3<f64> {
        self.lame_mu.clone()
    }

    fn shear_sound_speed_array(&self) -> Array3<f64> {
        self.shear_sound_speed.clone()
    }

    fn shear_viscosity_coeff_array(&self) -> Array3<f64> {
        self.shear_viscosity_coeff.clone()
    }

    fn bulk_viscosity_coeff_array(&self) -> Array3<f64> {
        self.bulk_viscosity_coeff.clone()
    }
}

// Thermal properties
impl ThermalProperties for HeterogeneousMedium {
    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.specific_heat[[ix, iy, iz]].max(100.0)
    }

    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.thermal_conductivity[[ix, iy, iz]].max(0.01)
    }

    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.thermal_diffusivity[[ix, iy, iz]].max(1e-9)
    }

    fn thermal_expansion(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.thermal_expansion[[ix, iy, iz]]
    }
}

// Thermal field management
impl ThermalField for HeterogeneousMedium {
    fn update_thermal_field(&mut self, temperature: &Array3<f64>) {
        self.temperature = temperature.clone();
    }

    fn thermal_field(&self) -> &Array3<f64> {
        &self.temperature
    }
}

// Optical properties
impl OpticalProperties for HeterogeneousMedium {
    fn optical_absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.mu_a[[ix, iy, iz]].max(0.0)
    }

    fn optical_scattering_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.mu_s_prime[[ix, iy, iz]].max(0.0)
    }
}

// Viscous properties
impl ViscousProperties for HeterogeneousMedium {
    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.viscosity[[ix, iy, iz]].max(1e-6)
    }

    fn shear_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.shear_viscosity_coeff[[ix, iy, iz]].max(1e-6)
    }

    fn bulk_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.bulk_viscosity_coeff[[ix, iy, iz]].max(1e-6)
    }
}

// Bubble properties
impl BubbleProperties for HeterogeneousMedium {
    fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.surface_tension[[ix, iy, iz]].max(0.01)
    }

    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.ambient_pressure
    }

    fn vapor_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.vapor_pressure[[ix, iy, iz]].max(1.0)
    }

    fn polytropic_index(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.polytropic_index[[ix, iy, iz]]
    }

    fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.gas_diffusion_coeff[[ix, iy, iz]].max(1e-12)
    }
}

// Bubble state management
impl BubbleState for HeterogeneousMedium {
    fn bubble_radius(&self) -> &Array3<f64> {
        &self.bubble_radius
    }

    fn bubble_velocity(&self) -> &Array3<f64> {
        &self.bubble_velocity
    }

    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>) {
        self.bubble_radius = radius.clone();
        self.bubble_velocity = velocity.clone();
    }
}
