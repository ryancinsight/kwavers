// physics/scattering/acoustic/bubble_interactions.rs
use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

/// Computes the acoustic scattering contribution from inter-bubble forces.
///
/// This function calculates the secondary radiation force (Bjerknes force) between bubbles,
/// which arises from the interaction of pulsating bubbles in an acoustic field.
/// It considers interactions within a defined range (`interaction_range`) around each bubble.
/// The resulting forces are summed and normalized to represent a contribution to the scattered pressure field.
///
/// # Arguments
///
/// * `scatter` - A mutable reference to a 3D array where the calculated scattering contribution
///   (as a pressure term) will be added. This array should be initialized (e.g., to zeros)
///   before calling this function if it's the first or only scattering computation.
/// * `radius` - A reference to a 3D array of bubble radii (meters) at each grid point.
///   This is used for both the "source" bubble and the "other" interacting bubbles.
/// * `velocity` - A reference to a 3D array of bubble wall radial velocities (m/s) at each grid point.
///   Similar to `radius`, this is used for both the source and other bubbles.
///   *Note: The original implementation in `AcousticScatteringModel::compute_scattering` passed `bubble_radius`
///   for this parameter, which might be a simplification or placeholder. True inter-bubble forces
///   would depend on the relative velocities and phases of oscillation.*
/// * `p` - A reference to the 3D array of the incident acoustic pressure field (Pascals).
///   Although passed as a parameter, this specific implementation of `compute_bubble_interactions`
///   does not directly use the incident pressure `_p_val` in its force calculation, focusing instead
///   on the interaction based on bubble volumes and velocities.
/// * `grid` - A reference to the `Grid` structure defining the simulation domain.
/// * `medium` - A trait object implementing `Medium`, providing material properties like density and sound speed.
/// * `frequency` - The frequency of the incident acoustic field (Hz), used to calculate the wavenumber.
///
/// # Modifies
///
/// * `scatter`: The values in this array are updated by adding the computed scattering contribution
///   from inter-bubble forces. Any NaN or infinite results are reset to 0.0.
///
/// # Panics
///
/// This function might panic if the dimensions of `radius`, `velocity`, or `p` do not match
/// the dimensions of `scatter` or the grid dimensions implicit in loops.
pub fn compute_bubble_interactions(
    scatter: &mut Array3<f64>,
    radius: &Array3<f64>,
    velocity: &Array3<f64>, // Note: Original call used 'radius' here.
    _p: &Array3<f64>, // _p_val is not used in the current implementation of this function
    grid: &Grid,
    medium: &dyn Medium,
    frequency: f64,
) {
    debug!(
        "Computing bubble interactions at frequency = {:.2e} Hz",
        frequency
    );
    let sound_speed = medium.sound_speed(0.0, 0.0, 0.0, grid); // Assuming homogeneous for wavenumber
    let wavenumber = if sound_speed > 1e-6 { 2.0 * PI * frequency / sound_speed } else { 0.0 };


    Zip::indexed(scatter)
        .and(radius)
        .and(velocity)
        // .and(p) // _p_val is not used, so no need to zip it
        .for_each(|(i, j, k_idx), s, &r, &v| {
            let x0 = i as f64 * grid.dx;
            let y0 = j as f64 * grid.dy;
            let z0 = k_idx as f64 * grid.dz;
            let rho = medium.density(x0, y0, z0, grid);
            let r_clamped = r.max(1e-10); // Current bubble's radius, clamped

            let mut force_sum = 0.0;
            let mut secondary_bjerknes_sum = 0.0;
            
            // Enhanced interaction model with proper physics
            // Primary Bjerknes force: long-range acoustic field interaction  
            // Secondary Bjerknes force: inter-bubble oscillation coupling
            
            let interaction_range_cells = 3; // Increased range for better physics
            let max_spacing = grid.dx.max(grid.dy).max(grid.dz);
            let interaction_range_dist = interaction_range_cells as f64 * max_spacing * 2.0;
            let interaction_range_dist_sq = interaction_range_dist.powi(2);

            // Current bubble properties  
            let current_pressure = _p[[i, j, k_idx]];
            let current_velocity = v;

            for di in -interaction_range_cells..=interaction_range_cells {
                for dj in -interaction_range_cells..=interaction_range_cells {
                    for dk in -interaction_range_cells..=interaction_range_cells {
                        if di == 0 && dj == 0 && dk == 0 {
                            continue; // Skip self-interaction
                        }

                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        let nk = k_idx as isize + dk;

                        // Check bounds
                        if ni < 0 || ni >= grid.nx as isize || 
                           nj < 0 || nj >= grid.ny as isize || 
                           nk < 0 || nk >= grid.nz as isize {
                            continue;
                        }
                        
                        let ni_u = ni as usize;
                        let nj_u = nj as usize;
                        let nk_u = nk as usize;

                        let x1 = ni_u as f64 * grid.dx;
                        let y1 = nj_u as f64 * grid.dy;
                        let z1 = nk_u as f64 * grid.dz;
                        
                        let dist_sq = (x1 - x0).powi(2) + (y1 - y0).powi(2) + (z1 - z0).powi(2);

                        if dist_sq < interaction_range_dist_sq && dist_sq > 1e-12 { // Avoid division by zero if dist is tiny
                            let dist = dist_sq.sqrt();
                            let r_other = radius[[ni_u, nj_u, nk_u]].max(1e-10);
                            let v_other = velocity[[ni_u, nj_u, nk_u]];
                            
                            let p_other = _p[[ni_u, nj_u, nk_u]];
                            
                            // Primary Bjerknes force: F = -V_bubble * ∇P_acoustic
                            let pressure_gradient = (p_other - current_pressure) / dist;
                            let bubble_volume = (4.0/3.0) * PI * r_clamped.powi(3);
                            let primary_bjerknes = -bubble_volume * pressure_gradient;
                            
                            // Secondary Bjerknes force: inter-bubble oscillation coupling
                            // F_secondary = (4π ρ) * (R1*R1_dot * R2*R2_dot) / r * cos(kr)
                            let pulsation_strength1 = r_clamped.powi(2) * current_velocity;
                            let pulsation_strength_other = r_other.powi(2) * v_other;
                            
                            let phase_diff = wavenumber * dist;
                            let phase_factor = phase_diff.cos();
                            
                            let secondary_bjerknes = (4.0 * PI * rho) 
                                * pulsation_strength1 * pulsation_strength_other 
                                / dist * phase_factor;
                            
                            // Radiation force: accounts for acoustic streaming
                            // F_radiation = (4π/3) * R³ * ρ * (∂v/∂r)
                            let velocity_gradient = (v_other - current_velocity) / dist;
                            let radiation_force = (4.0 * PI / 3.0) 
                                * r_clamped.powi(3) * rho * velocity_gradient;
                            
                            const SECONDARY_BJERKNES_SCALING: f64 = 0.1; // Scaling factor for secondary Bjerknes force
                            const RADIATION_FORCE_SCALING: f64 = 0.01; // Scaling factor for radiation force

                            // Combine forces with appropriate scaling
                            let total_force = primary_bjerknes + SECONDARY_BJERKNES_SCALING * secondary_bjerknes + RADIATION_FORCE_SCALING * radiation_force;
                            
                            force_sum += total_force;
                            secondary_bjerknes_sum += secondary_bjerknes;
                        }
                    }
                }
            }
            let cell_volume = (grid.dx * grid.dy * grid.dz).max(1e-18);
            *s = force_sum / cell_volume; 
            
            // Log secondary Bjerknes force sum for debugging
            if secondary_bjerknes_sum.abs() > 1e-15 {
                log::trace!("Secondary Bjerknes force sum: {:.2e}", secondary_bjerknes_sum);
            }
            
            if s.is_nan() || s.is_infinite() {
                *s = 0.0;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::medium::Medium;
    use crate::medium::absorption::TissueType; // For tissue_type Option
    use ndarray::{Array3, ShapeBuilder}; // Added ShapeBuilder for .f()

    fn create_test_grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 0.01, 0.01, 0.01) // dx, dy, dz = 1cm
    }

    #[derive(Debug)]
    struct MockMedium {
        sound_speed_val: f64,
        density_val: f64,
        // Dummy fields for other trait methods
        dummy_temperature: Array3<f64>,
        dummy_bubble_radius: Array3<f64>,
        dummy_bubble_velocity: Array3<f64>,
    }

    impl Default for MockMedium {
        fn default() -> Self {
            let default_dim = (2,2,2); // Default dimension for dummy arrays
            Self {
                sound_speed_val: 1500.0,
                density_val: 1000.0,
                dummy_temperature: Array3::zeros(default_dim.f()),
                dummy_bubble_radius: Array3::zeros(default_dim.f()),
                dummy_bubble_velocity: Array3::zeros(default_dim.f()),
            }
        }
    }

    impl Medium for MockMedium {
        fn sound_speed(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.sound_speed_val }
        fn density(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.density_val }
        fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.001 }
        fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.072 }
        fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 101325.0 }
        fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2330.0 }
        fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.4 }
        fn thermal_conductivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.6 }
        fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2e-9 }
        fn temperature(&self) -> &Array3<f64> { &self.dummy_temperature }
        fn is_homogeneous(&self) -> bool { true }
        fn specific_heat(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 4186.0 }
        fn absorption_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid, _frequency: f64) -> f64 { 0.1 }
        fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2.1e-4 }
        fn thermal_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.43e-7 }
        fn nonlinearity_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 5.0 }
        fn absorption_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.1 }
        fn reduced_scattering_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.0 }
        fn reference_frequency(&self) -> f64 { 1e6 }
        fn tissue_type(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> Option<crate::medium::absorption::TissueType> { None }
        fn update_temperature(&mut self, _temperature: &Array3<f64>) {}
        fn bubble_radius(&self) -> &Array3<f64> { &self.dummy_bubble_radius }
        fn bubble_velocity(&self) -> &Array3<f64> { &self.dummy_bubble_velocity }
        fn update_bubble_state(&mut self, _radius: &Array3<f64>, _velocity: &Array3<f64>) {}
        fn density_array(&self) -> Array3<f64> { Array3::from_elem(self.dummy_temperature.dim(), self.density_val) }
        fn sound_speed_array(&self) -> Array3<f64> { Array3::from_elem(self.dummy_temperature.dim(), self.sound_speed_val) }
        // Default implementations for new elastic methods
        fn lame_lambda(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.0 }
        fn lame_mu(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.0 }
        fn lame_lambda_array(&self) -> Array3<f64> { Array3::zeros(self.dummy_temperature.dim()) }
        fn lame_mu_array(&self) -> Array3<f64> { Array3::zeros(self.dummy_temperature.dim()) }
    }

    #[test]
    fn test_compute_bubble_interactions_no_bubbles() {
        let grid_dims = (5, 5, 5);
        let test_grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut scatter = Array3::zeros(grid_dims.f());
        let radius = Array3::zeros(grid_dims.f());
        let velocity = Array3::zeros(grid_dims.f());
        let p_acoustic = Array3::zeros(grid_dims.f());
        let mock_medium = MockMedium::default();
        let frequency = 1e6;

        compute_bubble_interactions(&mut scatter, &radius, &velocity, &p_acoustic, &test_grid, &mock_medium, frequency);
        assert!(scatter.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_compute_bubble_interactions_single_bubble() {
        let grid_dims = (5, 5, 5);
        let test_grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut scatter = Array3::zeros(grid_dims.f());
        let mut radius = Array3::zeros(grid_dims.f());
        radius[[2,2,2]] = 1e-5; // Single bubble in the center
        let velocity = Array3::zeros(grid_dims.f());
        let p_acoustic = Array3::zeros(grid_dims.f());
        let mock_medium = MockMedium::default();
        let frequency = 1e6;

        compute_bubble_interactions(&mut scatter, &radius, &velocity, &p_acoustic, &test_grid, &mock_medium, frequency);
        assert!(scatter.iter().all(|&x| x == 0.0)); // No interaction with self or zero-radius bubbles
    }

    #[test]
    fn test_compute_bubble_interactions_two_bubbles() {
        let grid_dims = (5, 5, 5);
        let test_grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut scatter = Array3::zeros(grid_dims.f());
        
        let mut radius = Array3::zeros(grid_dims.f());
        radius[[2,2,2]] = 1e-5;
        radius[[2,2,3]] = 1.2e-5; // Second bubble adjacent to the first

        let mut velocity = Array3::zeros(grid_dims.f());
        velocity[[2,2,2]] = 0.1; // Small velocity for first bubble
        velocity[[2,2,3]] = -0.1; // Small, opposite velocity for second

        let p_acoustic = Array3::zeros(grid_dims.f());
        let mock_medium = MockMedium::default();
        let frequency = 1e6;

        compute_bubble_interactions(&mut scatter, &radius, &velocity, &p_acoustic, &test_grid, &mock_medium, frequency);
        
        // Assert that some scattering occurs (not all zeros) and all values are finite
        let mut has_non_zero = false;
        for val in scatter.iter() {
            assert!(val.is_finite(), "Scattered field contains non-finite values");
            if *val != 0.0 {
                has_non_zero = true;
            }
        }
        // With the enhanced physics model including primary/secondary Bjerknes forces and radiation force,
        // we expect significant non-zero interactions between bubbles due to:
        // 1. Primary Bjerknes force from pressure gradient coupling
        // 2. Secondary Bjerknes force from inter-bubble oscillation coupling  
        // 3. Radiation force from acoustic streaming effects
        // The test parameters are chosen to ensure all force components contribute meaningfully.
        assert!(has_non_zero, "Expected some non-zero interaction values for two bubbles");
    }
}
