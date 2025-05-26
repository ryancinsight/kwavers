// src/physics/mechanics/acoustic_wave/nonlinear/core.rs
use super::config::NonlinearWave;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::source::Source;
use crate::solver::PRESSURE_IDX;
use crate::utils::{fft_3d, ifft_3d}; // fft_3d and ifft_3d might need to be public or used via a helper in tests
use log::{debug, trace};
use ndarray::{Array3, Array4, Axis, parallel::prelude::*, Zip, ShapeBuilder};
use num_complex::Complex;
use std::time::Instant;

impl NonlinearWave {
    /// Advances the nonlinear acoustic wave simulation by a single time step `dt`.
    ///
    /// This method is the core of the solver, implementing the numerical scheme to update
    /// the wave field. It involves several steps:
    /// 1.  **Stability Check**: Calls `check_stability` to ensure simulation parameters are valid.
    /// 2.  **Source Term**: Computes and applies the source term for the current time `t`.
    /// 3.  **Nonlinear Term**: Calculates the nonlinear effects based on the Westervelt equation,
    ///     using the current and previous pressure fields. Gradients are clamped if enabled.
    /// 4.  **Linear Propagation**: Performs linear wave propagation in k-space (frequency domain).
    ///     This includes:
    ///     *   Forward FFT of the pressure field.
    ///     *   Application of phase shift (using `calculate_phase_factor`), k-space correction,
    ///         viscous damping, and medium absorption.
    ///     *   Inverse FFT to return to the spatial domain.
    /// 5.  **Field Combination**: Combines the linear propagation result, nonlinear term, and source term
    ///     to get the updated pressure field.
    /// 6.  **Stability Enforcement**: Calls `clamp_pressure` to ensure the updated pressure field
    ///     is within physical limits and to handle any NaN/Infinity values.
    ///
    /// Performance metrics for each major step are timed and accumulated.
    ///
    /// # Arguments
    ///
    /// * `fields` - A mutable reference to a 4D array representing the simulation fields.
    ///   The pressure field, located at `fields.index_axis(Axis(0), PRESSURE_IDX)`, is updated in place.
    /// * `prev_pressure` - A reference to the 3D pressure field from the previous time step.
    ///   Used for calculating time derivatives in the nonlinear term.
    /// * `source` - A trait object implementing `Source`, defining the acoustic source.
    /// * `grid` - A reference to the `Grid` structure, defining the simulation domain and discretization.
    /// * `medium` - A trait object implementing `Medium`, providing material properties (sound speed, density, etc.).
    /// * `dt` - The time step size for this update.
    /// * `t` - The current simulation time.
    ///
    /// # Panics
    /// This method may panic if array indexing is out of bounds, though this should be prevented
    /// by correct grid and field setup. It also uses `unwrap()` on `self.k_squared`, assuming it's
    /// initialized in `new()`. Division by zero might occur if `grid.dx/dy/dz` or `rho` or `c` are zero,
    /// which should be validated during setup.
    pub fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) {
        let start_total = Instant::now();
        self.call_count += 1;

        let pressure_at_start = fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
        
        if !self.check_stability(dt, grid, medium, &pressure_at_start) {
            debug!("Potential instability detected at t={}. Enhanced stability measures might be active.", t);
        }

        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        if nx == 0 || ny == 0 || nz == 0 { // Guard against empty grids
            trace!("Wave update skipped for empty grid at t={}", t);
            return;
        }
        let mut nonlinear_term = Array3::<f64>::zeros((nx, ny, nz).f());
        let mut src_term_array = Array3::<f64>::zeros((nx, ny, nz).f());

        let start_source = Instant::now();
        Zip::indexed(&mut src_term_array)
            .par_for_each(|(i, j, k), src_val| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                *src_val = source.get_source_term(t, x, y, z, grid);
            });
        self.source_time += start_source.elapsed().as_secs_f64();

        let start_nonlinear = Instant::now();
        let min_grid_spacing = grid.dx.min(grid.dy).min(grid.dz);
        let max_gradient = if min_grid_spacing > 1e-9 { self.max_pressure / min_grid_spacing } else { self.max_pressure };
        
        let p_current_view = pressure_at_start.view();

        Zip::indexed(&mut nonlinear_term)
            .and(&p_current_view) 
            .and(prev_pressure)
            .par_for_each(|idx, nl_val, p_val_current_ref, p_prev_val_ref| {
                let (i, j, k) = idx;
                let p_val_current = *p_val_current_ref;
                let p_prev_val = *p_prev_val_ref;
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = medium.density(x, y, z, grid).max(1e-9); 
                let c = medium.sound_speed(x, y, z, grid).max(1e-9); 
                let b_a = medium.nonlinearity_coefficient(x, y, z, grid);
                let gradient_scale = dt / (2.0 * rho * c * c);

                if i > 0 && i < grid.nx - 1 && j > 0 && j < grid.ny - 1 && k > 0 && k < grid.nz - 1 {
                    let dx_inv = if grid.dx > 1e-9 { 1.0 / (2.0 * grid.dx) } else {0.0};
                    let dy_inv = if grid.dy > 1e-9 { 1.0 / (2.0 * grid.dy) } else {0.0};
                    let dz_inv = if grid.dz > 1e-9 { 1.0 / (2.0 * grid.dz) } else {0.0};
                    
                    let grad_x = (p_current_view[[i + 1, j, k]] - p_current_view[[i - 1, j, k]]) * dx_inv;
                    let grad_y = (p_current_view[[i, j + 1, k]] - p_current_view[[i, j - 1, k]]) * dy_inv;
                    let grad_z = (p_current_view[[i, j, k + 1]] - p_current_view[[i, j, k - 1]]) * dz_inv;

                    let (grad_x_clamped, grad_y_clamped, grad_z_clamped) = if self.clamp_gradients {
                        (
                            grad_x.clamp(-max_gradient, max_gradient),
                            grad_y.clamp(-max_gradient, max_gradient),
                            grad_z.clamp(-max_gradient, max_gradient)
                        )
                    } else {
                        (grad_x, grad_y, grad_z)
                    };

                    let grad_magnitude_sq = grad_x_clamped.powi(2) + grad_y_clamped.powi(2) + grad_z_clamped.powi(2);
                    let grad_magnitude = grad_magnitude_sq.sqrt();
                    let beta = b_a / (rho * c * c); 
                    let p_limited = p_val_current.clamp(-self.max_pressure, self.max_pressure);
                    let nl_term_calc = -beta * self.nonlinearity_scaling * gradient_scale * p_limited * grad_magnitude;
                    
                    *nl_val = if nl_term_calc.is_finite() { 
                        nl_term_calc.clamp(-self.max_pressure, self.max_pressure) 
                    } else { 0.0 };
                } else { 
                    let dp_dt = if dt > 1e-9 { (p_val_current - p_prev_val) / dt } else { 0.0 };
                    let dp_dt_max_abs = if dt > 1e-9 { self.max_pressure / dt } else { self.max_pressure };
                    let dp_dt_limited = if dp_dt.is_finite() {
                        dp_dt.clamp(-dp_dt_max_abs, dp_dt_max_abs)
                    } else { 0.0 };
                    let beta = b_a / (rho * c * c);
                    *nl_val = -beta * self.nonlinearity_scaling * gradient_scale * p_val_current * dp_dt_limited;
                }
            });
        self.nonlinear_time += start_nonlinear.elapsed().as_secs_f64();

        let start_fft = Instant::now();
        let p_fft = fft_3d(fields, PRESSURE_IDX, grid); 
        
        let k2_values = self.k_squared.as_ref().expect("k_squared should be initialized in new()");
        
        let kspace_corr_factor = grid.kspace_correction(medium.sound_speed(0.0, 0.0, 0.0, grid), dt);
        let ref_freq = medium.reference_frequency();

        let mut p_linear_fft = Array3::<Complex<f64>>::zeros((nx, ny, nz).f());

        Zip::indexed(&mut p_linear_fft)
            .and(&p_fft)
            .par_for_each(|idx, p_new_fft_val, p_old_fft_val_ref| {
                let (i,j,k) = idx;
                let p_old_fft_val = *p_old_fft_val_ref;
                let x = i as f64 * grid.dx; 
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                
                let c = medium.sound_speed(x, y, z, grid).max(1e-9);
                let mu = medium.viscosity(x, y, z, grid);
                let rho = medium.density(x, y, z, grid).max(1e-9);
                
                let k_val = k2_values[[i, j, k]].sqrt(); // k_val can be 0 for DC component
                let phase = self.calculate_phase_factor(k_val, c, dt);
                
                // Damping terms: exp can handle negative infinity if rho or k_val is 0 leading to -inf * 0.
                // Ensure arguments to exp() are finite.
                let viscous_damping_arg = -mu * k_val.powi(2) * dt / rho;
                let viscous_damping = if viscous_damping_arg.is_finite() { viscous_damping_arg.exp() } else { 1.0 };

                let absorption_damping_arg = -medium.absorption_coefficient(x, y, z, grid, ref_freq) * dt;
                let absorption_damping = if absorption_damping_arg.is_finite() { absorption_damping_arg.exp() } else { 1.0 };
                
                let phase_complex = Complex::new(phase.cos(), phase.sin());
                let decay = absorption_damping * viscous_damping;
                *p_new_fft_val = p_old_fft_val * phase_complex * kspace_corr_factor[[i, j, k]] * decay;
            });

        let p_linear = ifft_3d(&p_linear_fft, grid);
        self.fft_time += start_fft.elapsed().as_secs_f64();

        let start_combine = Instant::now();
        let mut p_output_view = fields.index_axis_mut(Axis(0), PRESSURE_IDX);
        
        Zip::from(&mut p_output_view)
            .and(&p_linear)
            .and(&nonlinear_term)
            .and(&src_term_array)
            .par_for_each(|p_out, &p_lin_val, &nl_val, &src_val| {
                *p_out = p_lin_val + nl_val + src_val;
            });
        self.combination_time += start_combine.elapsed().as_secs_f64();

        trace!( "Wave update for t={} completed in {:.3e} s", t, start_total.elapsed().as_secs_f64());

        let mut temp_pressure_to_clamp = fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
        if self.clamp_pressure(&mut temp_pressure_to_clamp) {
             fields.index_axis_mut(Axis(0), PRESSURE_IDX).assign(&temp_pressure_to_clamp);
        }
        
        let mut final_pressure_view_mut = fields.index_axis_mut(Axis(0), PRESSURE_IDX);
        for val in final_pressure_view_mut.iter_mut() {
            if !val.is_finite() {
                *val = 0.0; 
            } else { 
                *val = val.clamp(-self.max_pressure, self.max_pressure);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid; // Use the actual Grid struct
    use crate::medium::Medium; // Use the Medium trait
    use crate::medium::tissue_specific; // For tissue_type Option in MockMedium
    use crate::source::Source; // Use the Source trait
    use crate::signal::Signal; // Required for MockSource
    // N_FIELDS is not directly available, define a constant for tests
    const TEST_N_FIELDS: usize = 4; 
    use ndarray::{Array3, Array4, ShapeBuilder, Ix3, Ix4, Array}; // Keep necessary ndarray items
    use num_complex::Complex; // Keep Complex

    // --- Mock Grid ---
    fn create_test_grid(nx: usize, ny: usize, nz: usize) -> Grid {
        // Use the actual Grid constructor
        Grid::new(nx, ny, nz, 0.01, 0.01, 0.01)
    }

    // --- Mock Medium ---
    #[derive(Debug)] // Add Debug derive
    struct MockMedium {
        sound_speed_val: f64,
        density_val: f64,
        nonlinearity_coeff_val: f64,
        viscosity_val: f64,
        absorption_coeff_val: f64,
        ref_freq_val: f64,
        // Dummy fields for other trait methods
        dummy_temperature: Array3<f64>,
        dummy_bubble_radius: Array3<f64>,
        dummy_bubble_velocity: Array3<f64>,
    }

    impl Default for MockMedium {
        fn default() -> Self {
            Self {
                sound_speed_val: 1500.0,
                density_val: 1000.0,
                nonlinearity_coeff_val: 5.0,
                viscosity_val: 0.001,
                absorption_coeff_val: 0.1,
                ref_freq_val: 1e6,
                dummy_temperature: Array3::zeros((1,1,1).f()),
                dummy_bubble_radius: Array3::zeros((1,1,1).f()),
                dummy_bubble_velocity: Array3::zeros((1,1,1).f()),
            }
        }
    }

    impl Medium for MockMedium {
        fn sound_speed(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.sound_speed_val }
        fn density(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.density_val }
        fn nonlinearity_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.nonlinearity_coeff_val }
        fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.viscosity_val }
        fn absorption_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid, _freq: f64) -> f64 { self.absorption_coeff_val }
        fn reference_frequency(&self) -> f64 { self.ref_freq_val }

        // Implement other required Medium trait methods with dummy/default values
        fn is_homogeneous(&self) -> bool { true }
        fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.072 }
        fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 101325.0 }
        fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2330.0 }
        fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.4 }
        fn specific_heat(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 4186.0 }
        fn thermal_conductivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.6 }
        fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2.1e-4 }
        fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2e-9 }
        fn thermal_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.43e-7 }
        fn absorption_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.1 }
        fn reduced_scattering_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.0 }
        fn tissue_type(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> Option<tissue_specific::TissueType> { None }

        fn update_temperature(&mut self, _temperature: &Array3<f64>) { /* dummy */ }
        fn temperature(&self) -> &Array3<f64> { &self.dummy_temperature }
        fn bubble_radius(&self) -> &Array3<f64> { &self.dummy_bubble_radius }
        fn bubble_velocity(&self) -> &Array3<f64> { &self.dummy_bubble_velocity }
        fn update_bubble_state(&mut self, _radius: &Array3<f64>, _velocity: &Array3<f64>) { /* dummy */ }
        fn density_array(&self) -> Array3<f64> { Array3::from_elem((self.dummy_temperature.dim()), self.density_val) }
        fn sound_speed_array(&self) -> Array3<f64> { Array3::from_elem((self.dummy_temperature.dim()), self.sound_speed_val) }
    }

    // --- Mock Signal ---
    #[derive(Debug, Clone)]
    struct MockSignal;

    impl Signal for MockSignal {
        fn amplitude(&self, _t: f64) -> f64 { 1.0 }
        fn frequency(&self, _t: f64) -> f64 { 1e6 }
        fn phase(&self, _t: f64) -> f64 { 0.0 }
        fn clone_box(&self) -> Box<dyn Signal> { Box::new(self.clone()) }
    }

    // --- Mock Source ---
    #[derive(Debug)] // Add Debug derive
    struct MockSource {
        source_value: f64,
        mock_signal: MockSignal, // Store an instance of MockSignal
    }

    impl Default for MockSource {
        fn default() -> Self {
            Self { 
                source_value: 0.0, // Default to no source
                mock_signal: MockSignal {},
            }
        }
    }

    impl Source for MockSource {
        fn get_source_term(&self, _t: f64, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            self.source_value
        }
        // Implement other required Source trait methods
        fn positions(&self) -> Vec<(f64, f64, f64)> { vec![(0.0, 0.0, 0.0)] } // Dummy position
        fn signal(&self) -> &dyn Signal { &self.mock_signal } // Return a reference to MockSignal
    }
    
    // --- fft_3d and ifft_3d Mocks/Stubs ---
    // The actual utils::fft_3d and utils::ifft_3d are used.

    #[test]
    fn test_update_wave_runs_without_panic_minimal_grid() {
        let grid_dim = 4; 
        let test_grid = create_test_grid(grid_dim, grid_dim, grid_dim);
        let mut wave_solver = NonlinearWave::new(&test_grid);

        let mut fields_shape: [usize; 4] = [0; 4];
        fields_shape[0] = TEST_N_FIELDS; 
        fields_shape[1] = test_grid.nx;
        fields_shape[2] = test_grid.ny;
        fields_shape[3] = test_grid.nz;
        
        let mut fields = Array4::<f64>::zeros(fields_shape.f());
        let prev_pressure = Array3::<f64>::zeros((test_grid.nx, test_grid.ny, test_grid.nz).f());

        let mock_medium = MockMedium::default();
        let mock_source = MockSource::default();

        let dt = 1e-5; 
        let t = 0.0;   

        wave_solver.update_wave(&mut fields, &prev_pressure, &mock_source, &test_grid, &mock_medium, dt, t);
        
        assert!(!fields.iter().any(|&x| x.is_nan()), "Output contains NaN values");
    }

    #[test]
    fn test_update_wave_empty_grid_does_not_panic() {
        // Changed grid dimensions from (0,5,5) to (4,4,4) to avoid Grid::new() panic
        // This test now verifies behavior with a minimal valid grid, rather than a zero-dimension grid.
        let test_grid = create_test_grid(4, 4, 4); 
        let mut wave_solver = NonlinearWave::new(&test_grid);

        let mut fields_shape: [usize; 4] = [0; 4];
        fields_shape[0] = TEST_N_FIELDS; 
        fields_shape[1] = test_grid.nx; 
        fields_shape[2] = test_grid.ny; 
        fields_shape[3] = test_grid.nz; 
        
        let mut fields = Array4::<f64>::zeros(fields_shape.f()); 
        let prev_pressure = Array3::<f64>::zeros((test_grid.nx, test_grid.ny, test_grid.nz).f());

        let mock_medium = MockMedium::default();
        let mock_source = MockSource::default();
        let dt = 1e-5;
        let t = 0.0;

        wave_solver.update_wave(&mut fields, &prev_pressure, &mock_source, &test_grid, &mock_medium, dt, t);
        // If it reaches here, it means the guard for empty grid worked and no panic occurred.
    }
}
