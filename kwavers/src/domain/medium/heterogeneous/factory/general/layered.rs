//! Layered heterogeneous medium factory method.

use crate::core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use crate::core::constants::thermodynamic::ROOM_TEMPERATURE_K;
use super::HeterogeneousFactory;
use crate::domain::grid::Grid;
use crate::domain::medium::heterogeneous::core::HeterogeneousMedium;
use ndarray::Array3;

impl HeterogeneousFactory {
    /// Create a heterogeneous medium representing a layered structure
    ///
    /// **k-Wave Equivalent**: Defining layers with different properties
    ///
    /// # Arguments
    /// * `grid` - Computational grid
    /// * `layers` - Vector of (z_start, z_end, sound_speed, density, absorption, nonlinearity)
    /// * `reference_frequency` - Reference frequency for absorption (Hz)
    ///
    /// # Example
    /// ```
    /// // Water layer from z=0 to z=0.01m, then tissue layer
    /// let layers = vec![
    ///     (0.0, 0.01, 1500.0, 1000.0, 0.0, 0.0),
    ///     (0.01, 0.02, 1540.0, 1060.0, 0.5, 6.0),
    /// ];
    /// ```
    #[must_use]
    pub fn from_layers(
        grid: &Grid,
        layers: &[(f64, f64, f64, f64, f64, f64)],
        reference_frequency: f64,
    ) -> HeterogeneousMedium {
        let (nx, ny, nz) = grid.dimensions();

        let mut sound_speed = Array3::zeros((nx, ny, nz));
        let mut density = Array3::zeros((nx, ny, nz));
        let mut absorption = Array3::zeros((nx, ny, nz));
        let mut nonlinearity = Array3::zeros((nx, ny, nz));

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let (_, _, z) = grid.indices_to_coordinates(i, j, k);

                    for &(z_start, z_end, c, rho, alpha, ba) in layers {
                        if z >= z_start && z < z_end {
                            sound_speed[[i, j, k]] = c;
                            density[[i, j, k]] = rho;
                            absorption[[i, j, k]] = alpha;
                            nonlinearity[[i, j, k]] = ba;
                            break;
                        }
                    }
                }
            }
        }

        HeterogeneousMedium {
            use_trilinear_interpolation: true,
            density,
            sound_speed,
            viscosity: Array3::zeros((nx, ny, nz)),
            surface_tension: Array3::zeros((nx, ny, nz)),
            ambient_pressure: ATMOSPHERIC_PRESSURE,
            vapor_pressure: Array3::zeros((nx, ny, nz)),
            polytropic_index: Array3::zeros((nx, ny, nz)),
            specific_heat: Array3::zeros((nx, ny, nz)),
            thermal_conductivity: Array3::zeros((nx, ny, nz)),
            thermal_expansion: Array3::zeros((nx, ny, nz)),
            gas_diffusion_coeff: Array3::zeros((nx, ny, nz)),
            thermal_diffusivity: Array3::zeros((nx, ny, nz)),
            mu_a: Array3::zeros((nx, ny, nz)),
            mu_s_prime: Array3::zeros((nx, ny, nz)),
            temperature: Array3::from_elem((nx, ny, nz), ROOM_TEMPERATURE_K),
            bubble_radius: Array3::zeros((nx, ny, nz)),
            bubble_velocity: Array3::zeros((nx, ny, nz)),
            alpha0: Array3::zeros((nx, ny, nz)),
            delta: Array3::zeros((nx, ny, nz)),
            b_a: nonlinearity.clone(),
            absorption,
            alpha_power: Array3::from_elem((nx, ny, nz), 1.0),
            nonlinearity,
            shear_sound_speed: Array3::zeros((nx, ny, nz)),
            shear_viscosity_coeff: Array3::zeros((nx, ny, nz)),
            bulk_viscosity_coeff: Array3::zeros((nx, ny, nz)),
            lame_lambda: Array3::zeros((nx, ny, nz)),
            lame_mu: Array3::zeros((nx, ny, nz)),
            reference_frequency,
        }
    }
}
