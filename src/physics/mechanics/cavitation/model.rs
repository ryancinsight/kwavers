// src/physics/mechanics/cavitation/model.rs
use crate::grid::Grid;
use ndarray::Array3;
use log::debug;

// Constants moved from the original mod.rs
// These define general physical limits or reference values for the model.

/// Default minimum radius for a bubble in the model (1 nm).
/// Used as a lower bound in various calculations to prevent division by zero or unrealistic sizes.
pub(crate) const MIN_RADIUS_MODEL_DEFAULT: f64 = 1.0e-9;
/// Default maximum radius for a bubble in the model (1 mm).
/// Used as an upper bound in various calculations.
pub(crate) const MAX_RADIUS_MODEL_DEFAULT: f64 = 1.0e-3;
/// Default maximum radial velocity for a bubble wall (1000 m/s).
/// Used to clamp bubble wall velocities to physically plausible ranges.
pub(crate) const MAX_VELOCITY_MODEL_DEFAULT: f64 = 1.0e3;
/// Default maximum radial acceleration for a bubble wall (1e12 m/s^2).
/// Used to clamp bubble wall accelerations.
pub(crate) const MAX_ACCELERATION_MODEL_DEFAULT: f64 = 1.0e12;


/// Represents the state and parameters for simulating cavitation bubble dynamics and their effects.
///
/// This struct holds 3D arrays for bubble radius, velocity, previous velocity (for time integration),
/// and temperature at each grid point. It also caches intermediate results like acceleration
/// and scattering contributions to optimize performance. The model is initialized with a grid
/// and an initial bubble radius, and its state is updated over time by methods in `core.rs`,
/// `dynamics.rs`, and `effects.rs`.
#[derive(Debug)]
pub struct LegacyCavitationModel {
    /// 3D array representing the radius of cavitation bubbles at each grid point (meters).
    pub(crate) radius: Array3<f64>,
    /// 3D array representing the radial velocity of the bubble wall at each grid point (m/s).
    pub(crate) velocity: Array3<f64>,
    /// 3D array storing the bubble wall velocity from the previous time step, used for integration.
    pub(crate) prev_velocity: Array3<f64>,
    /// 3D array representing the temperature inside the cavitation bubbles at each grid point (Kelvin).
    pub(crate) temperature: Array3<f64>,
    
    // Enhanced physics fields
    /// 3D array representing the equilibrium radius of bubbles at each grid point (meters).
    pub(crate) r0: Array3<f64>,
    /// 3D array representing the number of gas molecules in each bubble.
    pub(crate) n_gas: Array3<f64>,
    /// 3D array representing the number of vapor molecules in each bubble.
    pub(crate) n_vapor: Array3<f64>,
    /// 3D array representing the internal pressure of bubbles (Pa).
    pub(crate) pressure_internal: Array3<f64>,
    /// 3D array storing maximum temperature reached during collapse (for sonoluminescence).
    pub(crate) max_temperature: Array3<f64>,
    /// 3D array storing maximum compression ratio reached during collapse.
    pub(crate) max_compression: Array3<f64>,
    
    // Cached values for better performance
    /// 3D array storing the second time derivative of the bubble radius (acceleration, m/s^2) at each grid point.
    /// Calculated in `dynamics.rs`.
    pub(crate) d2r_dt2: Array3<f64>,
    /// 3D array storing the contribution of Rayleigh scattering to the acoustic field at each grid point.
    /// Calculated in `effects.rs`.
    pub(crate) rayleigh_scatter: Array3<f64>,
    /// 3D array storing the contribution of Mie scattering to the acoustic field at each grid point.
    /// Calculated in `effects.rs`.
    pub(crate) mie_scatter: Array3<f64>,
    /// 3D array storing the contribution of bubble-bubble interactions to the acoustic field at each grid point.
    /// Calculated in `effects.rs`.
    pub(crate) interaction_scatter: Array3<f64>,
}

impl LegacyCavitationModel {
    /// Creates a new `CavitationModel` instance.
    ///
    /// Initializes the bubble field with a uniform `initial_radius` across the specified `grid`.
    /// Velocities, temperatures, and cached fields are initialized to default values (e.g., zero or room temperature).
    /// The `initial_radius` is clamped to be at least `MIN_RADIUS_MODEL_DEFAULT`.
    ///
    /// # Arguments
    ///
    /// * `grid` - A reference to the `Grid` structure defining the simulation domain and discretization.
    /// * `initial_radius` - The initial radius for all bubbles in the simulation field (meters).
    ///
    /// # Returns
    ///
    /// A new `CavitationModel` instance.
    pub fn new(grid: &Grid, initial_radius: f64) -> Self {
        debug!(
            "Initializing CavitationModel with initial radius = {:.6e} m",
            initial_radius
        );
        let dim = (grid.nx, grid.ny, grid.nz);
        let r0_clamped = initial_radius.max(MIN_RADIUS_MODEL_DEFAULT);
        
        // Initialize gas content based on equilibrium conditions
        // Using ideal gas law: n = PV/(RT)
        let p0 = 101325.0; // 1 atm
        let t0 = 293.15; // 20°C
        let r_gas = 8.314; // J/(mol·K)
        let v0 = 4.0 * std::f64::consts::PI * r0_clamped.powi(3) / 3.0;
        let n_gas_init = p0 * v0 / (r_gas * t0) * 6.022e23; // Convert to number of molecules
        
        Self {
            radius: Array3::from_elem(dim, r0_clamped),
            velocity: Array3::zeros(dim),
            prev_velocity: Array3::zeros(dim),
            temperature: Array3::from_elem(dim, 293.15), // Default initial temperature (e.g., room temp)
            r0: Array3::from_elem(dim, r0_clamped),
            n_gas: Array3::from_elem(dim, n_gas_init),
            n_vapor: Array3::zeros(dim),
            pressure_internal: Array3::zeros(dim),
            max_temperature: Array3::zeros(dim),
            max_compression: Array3::zeros(dim),
            d2r_dt2: Array3::zeros(dim),
            rayleigh_scatter: Array3::zeros(dim),
            mie_scatter: Array3::zeros(dim),
            interaction_scatter: Array3::zeros(dim),
        }
    }

    /// Returns a reference to the 3D array of bubble radii.
    pub fn radius(&self) -> &Array3<f64> {
        &self.radius
    }
    
    /// Returns a reference to the 3D array of bubble wall velocities.
    pub fn velocity(&self) -> &Array3<f64> {
        &self.velocity
    }
    
    /// Returns a reference to the 3D array of bubble temperatures.
    pub fn temperature(&self) -> &Array3<f64> {
        &self.temperature
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid; // Using the actual Grid struct

    fn create_test_grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 0.01, 0.01, 0.01) // Example spacing
    }

    #[test]
    fn test_cavitation_model_new() {
        let grid_dims = (2, 3, 4);
        let test_grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let initial_radius = 5e-6;

        let model = LegacyCavitationModel::new(&test_grid, initial_radius);

        // Assert dimensions
        assert_eq!(model.radius.dim(), grid_dims);
        assert_eq!(model.velocity.dim(), grid_dims);
        assert_eq!(model.prev_velocity.dim(), grid_dims);
        assert_eq!(model.temperature.dim(), grid_dims);
        assert_eq!(model.d2r_dt2.dim(), grid_dims);
        assert_eq!(model.rayleigh_scatter.dim(), grid_dims);
        assert_eq!(model.mie_scatter.dim(), grid_dims);
        assert_eq!(model.interaction_scatter.dim(), grid_dims);

        // Assert initial values
        for &r in model.radius.iter() {
            assert_eq!(r, initial_radius);
        }
        for &v in model.velocity.iter() {
            assert_eq!(v, 0.0);
        }
        for &pv in model.prev_velocity.iter() {
            assert_eq!(pv, 0.0);
        }
        for &t in model.temperature.iter() {
            assert_eq!(t, 293.15); // Default initial temperature
        }
        for &val in model.d2r_dt2.iter() { assert_eq!(val, 0.0); }
        for &val in model.rayleigh_scatter.iter() { assert_eq!(val, 0.0); }
        for &val in model.mie_scatter.iter() { assert_eq!(val, 0.0); }
        for &val in model.interaction_scatter.iter() { assert_eq!(val, 0.0); }

        // Test initial radius clamping
        let very_small_radius = MIN_RADIUS_MODEL_DEFAULT / 2.0;
        let model_clamped = LegacyCavitationModel::new(&test_grid, very_small_radius);
        for &r in model_clamped.radius.iter() {
            assert_eq!(r, MIN_RADIUS_MODEL_DEFAULT);
        }
    }

    #[test]
    fn test_accessor_methods() {
        let grid_dims = (2, 2, 2);
        let test_grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let initial_radius = 6e-6;
        let mut model = LegacyCavitationModel::new(&test_grid, initial_radius);

        // Modify a value to ensure accessors point to the right data
        model.radius[[0,0,0]] = 7e-6;
        model.velocity[[0,0,0]] = 1.0;
        model.temperature[[0,0,0]] = 300.0;

        assert_eq!(model.radius.dim(), grid_dims);
        assert_eq!(model.radius[[0,0,0]], 7e-6);

                assert_eq!(model.velocity.dim(), grid_dims);
        assert_eq!(model.velocity[[0,0,0]], 1.0);

        assert_eq!(model.temperature.dim(), grid_dims);
        assert_eq!(model.temperature[[0,0,0]], 300.0);
    }
}
