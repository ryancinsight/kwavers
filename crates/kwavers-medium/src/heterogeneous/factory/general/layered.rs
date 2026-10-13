//! Layered heterogeneous medium factory method.

use super::HeterogeneousFactory;
use crate::heterogeneous::core::HeterogeneousMedium;
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use kwavers_core::constants::thermodynamic::ROOM_TEMPERATURE_K;
use kwavers_grid::Grid;

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

        let mut sound_speed = grid.array_zeros();
        let mut density = grid.array_zeros();
        let mut absorption = grid.array_zeros();
        let mut nonlinearity = grid.array_zeros();

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
            viscosity: grid.array_zeros(),
            surface_tension: grid.array_zeros(),
            ambient_pressure: ATMOSPHERIC_PRESSURE,
            vapor_pressure: grid.array_zeros(),
            polytropic_index: grid.array_zeros(),
            specific_heat: grid.array_zeros(),
            thermal_conductivity: grid.array_zeros(),
            thermal_expansion: grid.array_zeros(),
            gas_diffusion_coeff: grid.array_zeros(),
            thermal_diffusivity: grid.array_zeros(),
            mu_a: grid.array_zeros(),
            mu_s_prime: grid.array_zeros(),
            temperature: grid.array_fill(ROOM_TEMPERATURE_K),
            bubble_radius: grid.array_zeros(),
            bubble_velocity: grid.array_zeros(),
            alpha0: grid.array_zeros(),
            delta: grid.array_zeros(),
            b_a: nonlinearity.clone(),
            absorption,
            alpha_power: grid.array_fill(1.0),
            nonlinearity,
            shear_sound_speed: grid.array_zeros(),
            shear_viscosity_coeff: grid.array_zeros(),
            bulk_viscosity_coeff: grid.array_zeros(),
            lame_lambda: grid.array_zeros(),
            lame_mu: grid.array_zeros(),
            reference_frequency,
        }
    }
}
