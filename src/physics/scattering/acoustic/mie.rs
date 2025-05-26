// physics/scattering/acoustic/mie.rs
use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

/// Computes the Mie scattering contribution to the acoustic field.
///
/// Mie scattering theory describes the scattering of electromagnetic or acoustic waves
/// by spherical particles. This function applies Mie theory for acoustic waves,
/// typically when the particle size (characterized by radius `r`) is comparable to
/// the wavelength of the incident sound (i.e., `kr >= 1.0`, where `k` is the wavenumber).
///
/// The calculation involves summing contributions from different scattering modes (monopole, dipole, quadrupole, etc.),
/// represented by coefficients `a0`, `a1`, `b1`, `a2`, `b2`. The resulting scattering amplitude
/// is then used to determine the scattered pressure.
///
/// If the condition `kr < 1.0` is met, this function sets the scattering contribution to zero,
/// as Rayleigh scattering (`compute_rayleigh_scattering`) is typically used in that regime.
///
/// # Arguments
///
/// * `scatter` - A mutable reference to a 3D array where the calculated Mie scattering contribution
///   (as a pressure term) will be added.
/// * `radius` - A reference to a 3D array of particle/bubble radii (meters) at each grid point.
/// * `p` - A reference to the 3D array of the incident acoustic pressure field (Pascals).
/// * `grid` - A reference to the `Grid` structure defining the simulation domain.
/// * `medium` - A trait object implementing `Medium`, providing material properties like density and sound speed.
/// * `frequency` - The frequency of the incident acoustic field (Hz).
///
/// # Modifies
///
/// * `scatter`: The values in this array are updated by adding the computed Mie scattering contribution.
///   Any NaN or infinite results are reset to 0.0.
///
/// # Panics
///
/// This function might panic if the dimensions of `radius` or `p` do not match
/// the dimensions of `scatter` or the grid dimensions implicit in loops.
pub fn compute_mie_scattering(
    scatter: &mut Array3<f64>,
    radius: &Array3<f64>,
    p: &Array3<f64>,
    grid: &Grid,
    medium: &dyn Medium,
    frequency: f64,
) {
    debug!(
        "Computing Mie scattering at frequency = {:.2e} Hz",
        frequency
    );
    let sound_speed = medium.sound_speed(0.0, 0.0, 0.0, grid); // Assuming homogeneous for wavenumber
    let wavenumber = if sound_speed > 1e-6 { 2.0 * PI * frequency / sound_speed } else { 0.0 };

    Zip::indexed(scatter)
        .and(radius)
        .and(p)
        .for_each(|(i, j, k_idx), s, &r, &p_val| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k_idx as f64 * grid.dz;
            let rho = medium.density(x, y, z, grid);
            let r_clamped = r.max(1e-10); // Avoid division by zero or issues with r=0
            let kr = wavenumber * r_clamped;

            if kr >= 1.0 {
                // Mie scattering coefficients (simplified or specific case)
                // These are derived from expansions of Bessel and Hankel functions.
                // The exact form depends on boundary conditions (e.g., rigid sphere, fluid sphere).
                // This appears to be a simplified form or a specific case.
                let sin_kr = kr.sin();
                let cos_kr = kr.cos();
                let kr_sq = kr * kr;

                // Avoid division by zero if kr is very small, though kr >= 1.0 should prevent this.
                // However, kr_sq could still be an issue if kr is large and leads to precision loss.
                // Adding a small epsilon to denominators if they could be zero is a common practice.
                let kr_eps = kr.max(1e-9); // kr used in denominator for a0, a1
                let kr_sq_eps = kr_sq.max(1e-18); // kr_sq used in denominator for a1, b1
                let kr_cube_eps = (kr * kr_sq).max(1e-27); // kr^3 used in denominator for a2, b2

                let a0 = sin_kr / kr_eps;
                let a1 = (2.0 * sin_kr - kr * cos_kr) / kr_sq_eps;
                let b1 = sin_kr / kr_sq_eps;
                let a2 = (9.0 * sin_kr - 3.0 * kr * cos_kr - kr_sq * sin_kr) / kr_cube_eps;
                let b2 = (3.0 * sin_kr - kr * cos_kr) / kr_cube_eps;

                // Scattering amplitude calculation (sum of mode contributions)
                // The formula for 'amplitude' seems specific and might represent a particular
                // approximation or integration over angles.
                let amplitude =
                    (a0.abs() + (2.0 * a1.abs() + b1.abs()) + (5.0 * a2.abs() + b2.abs())) * kr_sq;
                
                // The scattered pressure is proportional to density, incident pressure, and amplitude.
                // The division by cell volume characteristic length (dx*dy*dz)^(1/3) is heuristic
                // or specific to the model's normalization.
                let cell_char_length = (grid.dx * grid.dy * grid.dz).cbrt().max(1e-9);
                *s = rho * p_val * amplitude / cell_char_length;
            } else {
                // Mie regime condition not met, no contribution from this function.
                *s = 0.0;
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
    use crate::medium::tissue_specific; // For tissue_type Option
    use ndarray::{Array3, ShapeBuilder}; // Added ShapeBuilder for .f()

    fn create_test_grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 0.01, 0.01, 0.01)
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
            let default_dim = (1,1,1); // Minimal dimension for dummy arrays
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
        fn tissue_type(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> Option<tissue_specific::TissueType> { None }
        fn update_temperature(&mut self, _temperature: &Array3<f64>) {}
        fn bubble_radius(&self) -> &Array3<f64> { &self.dummy_bubble_radius }
        fn bubble_velocity(&self) -> &Array3<f64> { &self.dummy_bubble_velocity }
        fn update_bubble_state(&mut self, _radius: &Array3<f64>, _velocity: &Array3<f64>) {}
        fn density_array(&self) -> Array3<f64> { Array3::from_elem(self.dummy_temperature.dim(), self.density_val) }
        fn sound_speed_array(&self) -> Array3<f64> { Array3::from_elem(self.dummy_temperature.dim(), self.sound_speed_val) }
    }

    #[test]
    fn test_compute_mie_scattering_kr_large() {
        let grid_dims = (1,1,1);
        let test_grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut scatter = Array3::zeros(grid_dims.f());
        
        let sound_speed = 1500.0;
        let frequency = 1e6; // 1 MHz
        let radius_val = 0.5 * sound_speed / (2.0 * PI * frequency); // kr = 0.5, too small
        // Let's make kr = 2.0 for Mie scattering
        let radius_val_mie = 2.0 * sound_speed / (2.0 * PI * frequency); // kr = 2.0

        let mut radius_arr = Array3::zeros(grid_dims.f());
        radius_arr[[0,0,0]] = radius_val_mie;
        
        let mut p_acoustic = Array3::zeros(grid_dims.f());
        p_acoustic[[0,0,0]] = 1.0; // Incident pressure

        let mock_medium = MockMedium { sound_speed_val: sound_speed, ..MockMedium::default() };

        compute_mie_scattering(&mut scatter, &radius_arr, &p_acoustic, &test_grid, &mock_medium, frequency);
        
        assert!(scatter[[0,0,0]] != 0.0, "Expected non-zero Mie scattering for kr >= 1.0");
        assert!(scatter[[0,0,0]].is_finite(), "Mie scattering result should be finite");
    }

    #[test]
    fn test_compute_mie_scattering_kr_small() {
        let grid_dims = (1,1,1);
        let test_grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut scatter = Array3::zeros(grid_dims.f());

        let sound_speed = 1500.0;
        let frequency = 1e6; // 1 MHz
        // kr = 0.1 (small)
        let radius_val = 0.1 * sound_speed / (2.0 * PI * frequency); 

        let mut radius_arr = Array3::zeros(grid_dims.f());
        radius_arr[[0,0,0]] = radius_val;
        
        let mut p_acoustic = Array3::zeros(grid_dims.f());
        p_acoustic[[0,0,0]] = 1.0; // Incident pressure

        let mock_medium = MockMedium { sound_speed_val: sound_speed, ..MockMedium::default() };

        compute_mie_scattering(&mut scatter, &radius_arr, &p_acoustic, &test_grid, &mock_medium, frequency);
        
        assert_eq!(scatter[[0,0,0]], 0.0, "Expected zero Mie scattering for kr < 1.0");
    }
}
