// src/physics/mechanics/acoustic_wave/nonlinear/config.rs
use crate::grid::Grid;
use ndarray::Array3;
use log::debug;

/// Represents an improved nonlinear wave model solver.
///
/// This struct encapsulates the parameters and state for simulating acoustic wave propagation
/// with nonlinear effects, focusing on optimized efficiency and physical accuracy.
/// It includes settings for performance monitoring, physical model characteristics,
/// precomputed values for faster calculations, and stability control mechanisms.
#[derive(Debug, Clone)]
pub struct NonlinearWave {
    // Performance metrics
    /// Time spent in the nonlinear term calculation part of the update, in seconds.
    pub(super) nonlinear_time: f64,
    /// Time spent in FFT operations during the update, in seconds.
    pub(super) fft_time: f64,
    /// Time spent applying the source term during the update, in seconds.
    pub(super) source_time: f64,
    /// Time spent combining linear and nonlinear field components, in seconds.
    pub(super) combination_time: f64,
    /// Number of times the `update_wave` method has been called.
    pub(super) call_count: usize,
    
    // Physical model settings
    /// Scaling factor for the nonlinearity term, allowing adjustment for strong nonlinear effects.
    pub(super) nonlinearity_scaling: f64,
    /// Flag to enable or disable adaptive time-stepping for potentially more stable simulations.
    pub(super) use_adaptive_timestep: bool,
    /// Order of k-space correction for dispersion handling (e.g., 1 for first-order, 2 for second-order).
    pub(super) k_space_correction_order: usize,
    
    // Precomputed arrays
    /// Precomputed k-squared values (square of wavenumber magnitudes) for the grid, used to speed up calculations.
    pub(super) k_squared: Option<Array3<f64>>,
    
    // Stability parameters
    /// Maximum absolute pressure value allowed in the simulation to prevent numerical instability.
    pub(super) max_pressure: f64,
    /// Threshold for the CFL (Courant-Friedrichs-Lewy) condition, used in stability checks.
    pub(super) stability_threshold: f64,
    /// Safety factor applied to the CFL condition for determining stable timestep.
    pub(super) cfl_safety_factor: f64,
    /// Flag to enable or disable clamping of pressure gradients to `max_gradient`.
    pub(super) clamp_gradients: bool,
}

impl NonlinearWave {
    /// Creates a new `NonlinearWave` solver instance.
    ///
    /// Initializes the solver with default parameters and precomputes necessary values
    /// based on the provided `grid`.
    ///
    /// # Arguments
    ///
    /// * `grid` - A reference to the `Grid` structure defining the simulation domain and discretization.
    ///
    /// # Returns
    ///
    /// A new `NonlinearWave` instance.
    pub fn new(grid: &Grid) -> Self {
        debug!("Initializing NonlinearWave solver");
        
        // Precompute k-squared values to avoid recomputation in every step
        let k_squared = Some(grid.k_squared());
        
        Self {
            nonlinear_time: 0.0,
            fft_time: 0.0,
            source_time: 0.0,
            combination_time: 0.0,
            call_count: 0,
            nonlinearity_scaling: 1.0,     // Default scaling factor for nonlinearity
            use_adaptive_timestep: false,  // Default to fixed timestep
            k_space_correction_order: 2,   // Default to second-order correction
            k_squared,
            max_pressure: 1e8,  // 100 MPa maximum pressure
            stability_threshold: 0.5, // CFL condition threshold
            cfl_safety_factor: 0.8,   // Safety factor for CFL condition
            clamp_gradients: true,    // Enable gradient clamping
        }
    }

    /// Sets the scaling factor for the nonlinearity term.
    ///
    /// This allows adjustment for strong nonlinear effects. The scaling factor must be positive.
    ///
    /// # Arguments
    ///
    /// * `scaling` - The nonlinearity scaling factor. Must be greater than 0.0.
    pub fn set_nonlinearity_scaling(&mut self, scaling: f64) {
        assert!(scaling > 0.0, "Nonlinearity scaling must be positive");
        self.nonlinearity_scaling = scaling;
    }

    /// Enables or disables adaptive time-stepping.
    ///
    /// Adaptive time-stepping can help maintain stability in simulations with varying conditions.
    ///
    /// # Arguments
    ///
    /// * `enable` - `true` to enable adaptive time-stepping, `false` to disable it.
    pub fn set_adaptive_timestep(&mut self, enable: bool) {
        self.use_adaptive_timestep = enable;
    }

    /// Sets the order of k-space correction for dispersion.
    ///
    /// Higher orders can provide more accurate handling of wave dispersion but may increase
    /// computational cost. The order must be between 1 and 4 (inclusive).
    ///
    /// # Arguments
    ///
    /// * `order` - The desired order of k-space correction (1-4).
    pub fn set_k_space_correction_order(&mut self, order: usize) {
        assert!(order > 0 && order <= 4, "Correction order must be between 1 and 4");
        self.k_space_correction_order = order;
    }

    /// Sets the maximum allowed absolute pressure value in the simulation.
    ///
    /// This is used to clamp pressure values and prevent numerical instability.
    ///
    /// # Arguments
    ///
    /// * `max_pressure` - The maximum absolute pressure value.
    pub fn set_max_pressure(&mut self, max_pressure: f64) {
        self.max_pressure = max_pressure;
    }

    /// Sets parameters related to simulation stability.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The CFL condition threshold.
    /// * `safety_factor` - The safety factor for the CFL condition.
    /// * `clamp_gradients` - `true` to enable clamping of pressure gradients, `false` otherwise.
    pub fn set_stability_params(&mut self, threshold: f64, safety_factor: f64, clamp_gradients: bool) {
        self.stability_threshold = threshold;
        self.cfl_safety_factor = safety_factor;
        self.clamp_gradients = clamp_gradients;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid; // Use the actual Grid struct
    // Removed unused ndarray::Array3 import from here, it's used by NonlinearWave itself.

    // create_test_grid now returns the actual Grid struct
    fn create_test_grid() -> Grid { 
        Grid::new(10, 10, 10, 0.1, 0.1, 0.1) // Use the actual Grid constructor
    }

    #[test]
    fn test_new_nonlinear_wave() {
        let test_grid = create_test_grid();
        let wave = NonlinearWave::new(&test_grid);

        assert_eq!(wave.nonlinearity_scaling, 1.0);
        assert_eq!(wave.use_adaptive_timestep, false);
        assert_eq!(wave.k_space_correction_order, 2);
        assert_eq!(wave.max_pressure, 1e8);
        assert_eq!(wave.stability_threshold, 0.5);
        assert_eq!(wave.cfl_safety_factor, 0.8);
        assert_eq!(wave.clamp_gradients, true);
        assert!(wave.k_squared.is_some());
        assert_eq!(wave.call_count, 0);
        assert_eq!(wave.nonlinear_time, 0.0);
        // ... and other performance metrics
    }

    #[test]
    fn test_set_nonlinearity_scaling() {
        let test_grid = create_test_grid();
        let mut wave = NonlinearWave::new(&test_grid);
        wave.set_nonlinearity_scaling(2.5);
        assert_eq!(wave.nonlinearity_scaling, 2.5);
    }

    #[test]
    #[should_panic(expected = "Nonlinearity scaling must be positive")]
    fn test_set_nonlinearity_scaling_panic_zero() {
        let test_grid = create_test_grid();
        let mut wave = NonlinearWave::new(&test_grid);
        wave.set_nonlinearity_scaling(0.0);
    }

    #[test]
    #[should_panic(expected = "Nonlinearity scaling must be positive")]
    fn test_set_nonlinearity_scaling_panic_negative() {
        let test_grid = create_test_grid();
        let mut wave = NonlinearWave::new(&test_grid);
        wave.set_nonlinearity_scaling(-1.0);
    }
    
    #[test]
    fn test_set_adaptive_timestep() {
        let test_grid = create_test_grid();
        let mut wave = NonlinearWave::new(&test_grid);
        wave.set_adaptive_timestep(true);
        assert_eq!(wave.use_adaptive_timestep, true);
        wave.set_adaptive_timestep(false);
        assert_eq!(wave.use_adaptive_timestep, false);
    }

    #[test]
    fn test_set_k_space_correction_order() {
        let test_grid = create_test_grid();
        let mut wave = NonlinearWave::new(&test_grid);
        wave.set_k_space_correction_order(3);
        assert_eq!(wave.k_space_correction_order, 3);
    }

    #[test]
    #[should_panic(expected = "Correction order must be between 1 and 4")]
    fn test_set_k_space_correction_order_panic_zero() {
        let test_grid = create_test_grid();
        let mut wave = NonlinearWave::new(&test_grid);
        wave.set_k_space_correction_order(0);
    }

    #[test]
    #[should_panic(expected = "Correction order must be between 1 and 4")]
    fn test_set_k_space_correction_order_panic_high() {
        let test_grid = create_test_grid();
        let mut wave = NonlinearWave::new(&test_grid);
        wave.set_k_space_correction_order(5);
    }

    #[test]
    fn test_set_max_pressure() {
        let test_grid = create_test_grid();
        let mut wave = NonlinearWave::new(&test_grid);
        wave.set_max_pressure(5e7);
        assert_eq!(wave.max_pressure, 5e7);
    }

    #[test]
    fn test_set_stability_params() {
        let test_grid = create_test_grid();
        let mut wave = NonlinearWave::new(&test_grid);
        wave.set_stability_params(0.4, 0.7, false);
        assert_eq!(wave.stability_threshold, 0.4);
        assert_eq!(wave.cfl_safety_factor, 0.7);
        assert_eq!(wave.clamp_gradients, false);
    }
}
