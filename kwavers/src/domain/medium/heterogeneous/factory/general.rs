//! General-purpose factory for heterogeneous media creation from arrays
//!
//! **Factory Pattern**: Enables creation from NumPy arrays for k-Wave compatibility
//! **k-Wave Parity**: Direct equivalent to k-Wave's medium struct with spatially varying properties

use crate::domain::grid::Grid;
use crate::domain::medium::heterogeneous::core::HeterogeneousMedium;
use ndarray::Array3;

/// Factory for creating heterogeneous media from arrays or functions
///
/// **Design Principle**: Single responsibility for array-based initialization
/// **k-Wave Compatibility**: Matches k-Wave's medium creation patterns
#[derive(Debug)]
pub struct HeterogeneousFactory;

/// Function type for spatially varying properties
pub type PropertyFunction = Box<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>;

impl HeterogeneousFactory {
    /// Create a heterogeneous medium from pre-existing 3D arrays
    ///
    /// **k-Wave Equivalent**: `medium.sound_speed = c_array; medium.density = rho_array;`
    ///
    /// # Arguments
    /// * `sound_speed` - 3D array of sound speeds [m/s]
    /// * `density` - 3D array of densities [kg/m³]
    /// * `absorption` - Optional 3D array of absorption coefficients [dB/(MHz·cm)]
    /// * `nonlinearity` - Optional 3D array of B/A parameters
    /// * `reference_frequency` - Reference frequency for absorption [Hz]
    ///
    /// # Returns
    /// * `HeterogeneousMedium` configured with the provided arrays
    ///
    /// # Errors
    /// Returns an error if array shapes don't match
    pub fn from_arrays(
        sound_speed: Array3<f64>,
        density: Array3<f64>,
        absorption: Option<Array3<f64>>,
        nonlinearity: Option<Array3<f64>>,
        reference_frequency: f64,
    ) -> Result<HeterogeneousMedium, String> {
        let shape = sound_speed.shape();
        if density.shape() != shape {
            return Err(format!(
                "Density shape {:?} doesn't match sound_speed shape {:?}",
                density.shape(),
                shape
            ));
        }

        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

        // Use provided arrays or create zeros
        let absorption = absorption.unwrap_or_else(|| Array3::zeros((nx, ny, nz)));
        let nonlinearity = nonlinearity.unwrap_or_else(|| Array3::zeros((nx, ny, nz)));

        // Validate shapes
        if absorption.shape() != shape {
            return Err(format!(
                "Absorption shape {:?} doesn't match sound_speed shape {:?}",
                absorption.shape(),
                shape
            ));
        }
        if nonlinearity.shape() != shape {
            return Err(format!(
                "Nonlinearity shape {:?} doesn't match sound_speed shape {:?}",
                nonlinearity.shape(),
                shape
            ));
        }

        Ok(HeterogeneousMedium {
            use_trilinear_interpolation: true,
            density,
            sound_speed,
            viscosity: Array3::zeros((nx, ny, nz)),
            surface_tension: Array3::zeros((nx, ny, nz)),
            ambient_pressure: 101325.0, // Standard atmospheric pressure
            vapor_pressure: Array3::zeros((nx, ny, nz)),
            polytropic_index: Array3::zeros((nx, ny, nz)),
            specific_heat: Array3::zeros((nx, ny, nz)),
            thermal_conductivity: Array3::zeros((nx, ny, nz)),
            thermal_expansion: Array3::zeros((nx, ny, nz)),
            gas_diffusion_coeff: Array3::zeros((nx, ny, nz)),
            thermal_diffusivity: Array3::zeros((nx, ny, nz)),
            mu_a: Array3::zeros((nx, ny, nz)),
            mu_s_prime: Array3::zeros((nx, ny, nz)),
            temperature: Array3::from_elem((nx, ny, nz), 293.15), // 20°C default
            bubble_radius: Array3::zeros((nx, ny, nz)),
            bubble_velocity: Array3::zeros((nx, ny, nz)),
            alpha0: Array3::zeros((nx, ny, nz)),
            delta: Array3::zeros((nx, ny, nz)),
            b_a: nonlinearity.clone(),
            absorption,
            nonlinearity,
            shear_sound_speed: Array3::zeros((nx, ny, nz)),
            shear_viscosity_coeff: Array3::zeros((nx, ny, nz)),
            bulk_viscosity_coeff: Array3::zeros((nx, ny, nz)),
            lame_lambda: Array3::zeros((nx, ny, nz)),
            lame_mu: Array3::zeros((nx, ny, nz)),
            reference_frequency,
        })
    }

    /// Create a heterogeneous medium from spatial functions
    ///
    /// **k-Wave Equivalent**: Defining medium properties as functions of position
    ///
    /// # Arguments
    /// * `grid` - Computational grid defining the domain
    /// * `sound_speed_fn` - Function f(x, y, z) -> sound speed [m/s]
    /// * `density_fn` - Function f(x, y, z) -> density [kg/m³]
    /// * `absorption_fn` - Optional function f(x, y, z) -> absorption [dB/(MHz·cm)]
    /// * `nonlinearity_fn` - Optional function f(x, y, z) -> B/A parameter
    /// * `reference_frequency` - Reference frequency for absorption [Hz]
    ///
    /// # Returns
    /// * `HeterogeneousMedium` with properties evaluated at grid points
    #[must_use]
    pub fn from_functions<F1, F2>(
        grid: &Grid,
        sound_speed_fn: F1,
        density_fn: F2,
        absorption_fn: Option<PropertyFunction>,
        nonlinearity_fn: Option<PropertyFunction>,
        reference_frequency: f64,
    ) -> HeterogeneousMedium
    where
        F1: Fn(f64, f64, f64) -> f64 + Send + Sync,
        F2: Fn(f64, f64, f64) -> f64 + Send + Sync,
    {
        let (nx, ny, nz) = grid.dimensions();

        // Evaluate functions at grid points
        let mut sound_speed = Array3::zeros((nx, ny, nz));
        let mut density = Array3::zeros((nx, ny, nz));
        let mut absorption = Array3::zeros((nx, ny, nz));
        let mut nonlinearity = Array3::zeros((nx, ny, nz));

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    sound_speed[[i, j, k]] = sound_speed_fn(x, y, z);
                    density[[i, j, k]] = density_fn(x, y, z);

                    if let Some(ref abs_fn) = absorption_fn {
                        absorption[[i, j, k]] = abs_fn(x, y, z);
                    }
                    if let Some(ref nl_fn) = nonlinearity_fn {
                        nonlinearity[[i, j, k]] = nl_fn(x, y, z);
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
            ambient_pressure: 101325.0,
            vapor_pressure: Array3::zeros((nx, ny, nz)),
            polytropic_index: Array3::zeros((nx, ny, nz)),
            specific_heat: Array3::zeros((nx, ny, nz)),
            thermal_conductivity: Array3::zeros((nx, ny, nz)),
            thermal_expansion: Array3::zeros((nx, ny, nz)),
            gas_diffusion_coeff: Array3::zeros((nx, ny, nz)),
            thermal_diffusivity: Array3::zeros((nx, ny, nz)),
            mu_a: Array3::zeros((nx, ny, nz)),
            mu_s_prime: Array3::zeros((nx, ny, nz)),
            temperature: Array3::from_elem((nx, ny, nz), 293.15),
            bubble_radius: Array3::zeros((nx, ny, nz)),
            bubble_velocity: Array3::zeros((nx, ny, nz)),
            alpha0: Array3::zeros((nx, ny, nz)),
            delta: Array3::zeros((nx, ny, nz)),
            b_a: nonlinearity.clone(),
            absorption,
            nonlinearity,
            shear_sound_speed: Array3::zeros((nx, ny, nz)),
            shear_viscosity_coeff: Array3::zeros((nx, ny, nz)),
            bulk_viscosity_coeff: Array3::zeros((nx, ny, nz)),
            lame_lambda: Array3::zeros((nx, ny, nz)),
            lame_mu: Array3::zeros((nx, ny, nz)),
            reference_frequency,
        }
    }

    /// Create a heterogeneous medium representing a layered structure
    ///
    /// **k-Wave Equivalent**: Defining layers with different properties
    ///
    /// # Arguments
    /// * `grid` - Computational grid
    /// * `layers` - Vector of (z_start, z_end, sound_speed, density, absorption, nonlinearity)
    /// * `reference_frequency` - Reference frequency for absorption [Hz]
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

                    // Find which layer this point belongs to
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
            ambient_pressure: 101325.0,
            vapor_pressure: Array3::zeros((nx, ny, nz)),
            polytropic_index: Array3::zeros((nx, ny, nz)),
            specific_heat: Array3::zeros((nx, ny, nz)),
            thermal_conductivity: Array3::zeros((nx, ny, nz)),
            thermal_expansion: Array3::zeros((nx, ny, nz)),
            gas_diffusion_coeff: Array3::zeros((nx, ny, nz)),
            thermal_diffusivity: Array3::zeros((nx, ny, nz)),
            mu_a: Array3::zeros((nx, ny, nz)),
            mu_s_prime: Array3::zeros((nx, ny, nz)),
            temperature: Array3::from_elem((nx, ny, nz), 293.15),
            bubble_radius: Array3::zeros((nx, ny, nz)),
            bubble_velocity: Array3::zeros((nx, ny, nz)),
            alpha0: Array3::zeros((nx, ny, nz)),
            delta: Array3::zeros((nx, ny, nz)),
            b_a: nonlinearity.clone(),
            absorption,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_arrays_basic() {
        let c = Array3::from_elem((10, 10, 10), 1500.0);
        let rho = Array3::from_elem((10, 10, 10), 1000.0);

        let medium = HeterogeneousFactory::from_arrays(c, rho, None, None, 1e6).unwrap();

        assert_eq!(medium.sound_speed[[0, 0, 0]], 1500.0);
        assert_eq!(medium.density[[0, 0, 0]], 1000.0);
        assert_eq!(medium.reference_frequency, 1e6);
    }

    #[test]
    fn test_from_arrays_with_optional() {
        let c = Array3::from_elem((10, 10, 10), 1500.0);
        let rho = Array3::from_elem((10, 10, 10), 1000.0);
        let alpha = Array3::from_elem((10, 10, 10), 0.5);
        let ba = Array3::from_elem((10, 10, 10), 6.0);

        let medium = HeterogeneousFactory::from_arrays(c, rho, Some(alpha), Some(ba), 1e6).unwrap();

        assert_eq!(medium.absorption[[0, 0, 0]], 0.5);
        assert_eq!(medium.nonlinearity[[0, 0, 0]], 6.0);
    }

    #[test]
    fn test_from_functions() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();

        let medium = HeterogeneousFactory::from_functions(
            &grid,
            |_x, _y, _z| 1500.0,
            |_x, _y, _z| 1000.0,
            Some(Box::new(|_x, _y, z| if z > 0.005 { 0.5 } else { 0.0 })),
            None,
            1e6,
        );

        assert_eq!(medium.sound_speed[[0, 0, 0]], 1500.0);
        assert_eq!(medium.density[[0, 0, 0]], 1000.0);
    }

    #[test]
    fn test_from_layers() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();

        let layers = vec![
            (0.0, 0.005, 1500.0, 1000.0, 0.0, 0.0),   // Water
            (0.005, 0.010, 1540.0, 1060.0, 0.5, 6.0), // Tissue
        ];

        let medium = HeterogeneousFactory::from_layers(&grid, &layers, 1e6);

        // Check water layer (first 5 z-slices)
        assert_eq!(medium.sound_speed[[0, 0, 0]], 1500.0);
        assert_eq!(medium.density[[0, 0, 0]], 1000.0);

        // Check tissue layer (last 5 z-slices)
        assert_eq!(medium.sound_speed[[0, 0, 9]], 1540.0);
        assert_eq!(medium.density[[0, 0, 9]], 1060.0);
    }
}
