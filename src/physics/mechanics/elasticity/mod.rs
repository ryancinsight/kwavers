use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::{Array3, Array4, Axis, Zip};
use num_complex::Complex;
use std::sync::Arc;

/// Elastic wave propagation model supporting both longitudinal and shear waves
pub struct ElasticWave {
    // Model parameters
    lambda: Array3<f64>,     // First Lamé parameter
    mu: Array3<f64>,         // Second Lamé parameter (shear modulus)
    rho: Array3<f64>,        // Density
    
    // Wave field components
    displacement: Array4<f64>,  // [u_x, u_y, u_z] displacement vector
    velocity: Array4<f64>,      // [v_x, v_y, v_z] velocity vector
    stress: Array4<f64>,        // [σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_xz] stress tensor
    
    // Performance optimization
    k_squared: Option<Array3<f64>>,
}

impl ElasticWave {
    pub fn new(grid: &Grid) -> Self {
        let nx = grid.nx();
        let ny = grid.ny();
        let nz = grid.nz();
        
        ElasticWave {
            lambda: Array3::zeros((nx, ny, nz)),
            mu: Array3::zeros((nx, ny, nz)),
            rho: Array3::zeros((nx, ny, nz)),
            displacement: Array4::zeros((3, nx, ny, nz)),
            velocity: Array4::zeros((3, nx, ny, nz)),
            stress: Array4::zeros((6, nx, ny, nz)),
            k_squared: Some(grid.k_squared()),
        }
    }
    
    /// Initialize elastic parameters from medium properties
    pub fn initialize_from_medium(&mut self, medium: &dyn Medium, grid: &Grid) {
        Zip::indexed(&mut self.lambda)
            .par_for_each(|(i, j, k), lambda| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                
                let c_p = medium.sound_speed(x, y, z, grid);
                let rho = medium.density(x, y, z, grid);
                let mu = self.mu[[i, j, k]];
                
                // Calculate lambda from P-wave speed: c_p² = (λ + 2μ)/ρ
                *lambda = rho * c_p * c_p - 2.0 * mu;
            });
            
        Zip::indexed(&mut self.rho)
            .par_for_each(|(i, j, k), rho| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                *rho = medium.density(x, y, z, grid);
            });
    }
    
    /// Set shear modulus distribution
    pub fn set_shear_modulus(&mut self, mu: Array3<f64>) {
        self.mu = mu;
    }
    
    /// Update stress tensor using Hooke's law
    fn update_stress(&mut self, dt: f64, grid: &Grid) {
        let dx = grid.dx;
        let dy = grid.dy;
        let dz = grid.dz;
        
        Zip::indexed(&mut self.stress)
            .and(&self.displacement)
            .par_for_each(|idx, stress, disp| {
                let (i, j, k) = (idx.0, idx.1, idx.2);
                
                // Calculate strain components using central differences
                let e_xx = (disp[[0, i+1, j, k]] - disp[[0, i-1, j, k]]) / (2.0 * dx);
                let e_yy = (disp[[1, i, j+1, k]] - disp[[1, i, j-1, k]]) / (2.0 * dy);
                let e_zz = (disp[[2, i, j, k+1]] - disp[[2, i, j, k-1]]) / (2.0 * dz);
                
                let e_xy = 0.5 * ((disp[[0, i, j+1, k]] - disp[[0, i, j-1, k]]) / (2.0 * dy) +
                                 (disp[[1, i+1, j, k]] - disp[[1, i-1, j, k]]) / (2.0 * dx));
                                 
                let e_yz = 0.5 * ((disp[[1, i, j, k+1]] - disp[[1, i, j, k-1]]) / (2.0 * dz) +
                                 (disp[[2, i, j+1, k]] - disp[[2, i, j-1, k]]) / (2.0 * dy));
                                 
                let e_xz = 0.5 * ((disp[[0, i, j, k+1]] - disp[[0, i, j, k-1]]) / (2.0 * dz) +
                                 (disp[[2, i+1, j, k]] - disp[[2, i-1, j, k]]) / (2.0 * dx));
                
                let lambda = self.lambda[[i, j, k]];
                let mu = self.mu[[i, j, k]];
                
                // Normal stresses
                stress[[0]] = lambda * (e_xx + e_yy + e_zz) + 2.0 * mu * e_xx;  // σ_xx
                stress[[1]] = lambda * (e_xx + e_yy + e_zz) + 2.0 * mu * e_yy;  // σ_yy
                stress[[2]] = lambda * (e_xx + e_yy + e_zz) + 2.0 * mu * e_zz;  // σ_zz
                
                // Shear stresses
                stress[[3]] = 2.0 * mu * e_xy;  // σ_xy
                stress[[4]] = 2.0 * mu * e_yz;  // σ_yz
                stress[[5]] = 2.0 * mu * e_xz;  // σ_xz
            });
    }
    
    /// Update velocity using Newton's second law
    fn update_velocity(&mut self, dt: f64, grid: &Grid) {
        let dx = grid.dx;
        let dy = grid.dy;
        let dz = grid.dz;
        
        Zip::indexed(&mut self.velocity)
            .and(&self.stress)
            .par_for_each(|idx, vel, stress| {
                let (i, j, k) = (idx.0, idx.1, idx.2);
                let rho = self.rho[[i, j, k]];
                
                // Calculate force components from stress gradients
                let f_x = (stress[[0, i+1, j, k]] - stress[[0, i-1, j, k]]) / (2.0 * dx) +
                         (stress[[3, i, j+1, k]] - stress[[3, i, j-1, k]]) / (2.0 * dy) +
                         (stress[[5, i, j, k+1]] - stress[[5, i, j, k-1]]) / (2.0 * dz);
                         
                let f_y = (stress[[3, i+1, j, k]] - stress[[3, i-1, j, k]]) / (2.0 * dx) +
                         (stress[[1, i, j+1, k]] - stress[[1, i, j-1, k]]) / (2.0 * dy) +
                         (stress[[4, i, j, k+1]] - stress[[4, i, j, k-1]]) / (2.0 * dz);
                         
                let f_z = (stress[[5, i+1, j, k]] - stress[[5, i-1, j, k]]) / (2.0 * dx) +
                         (stress[[4, i, j+1, k]] - stress[[4, i, j-1, k]]) / (2.0 * dy) +
                         (stress[[2, i, j, k+1]] - stress[[2, i, j, k-1]]) / (2.0 * dz);
                
                // Update velocities
                vel[[0]] += dt * f_x / rho;  // v_x
                vel[[1]] += dt * f_y / rho;  // v_y
                vel[[2]] += dt * f_z / rho;  // v_z
            });
    }
    
    /// Update displacement using velocity
    fn update_displacement(&mut self, dt: f64) {
        self.displacement += &(&self.velocity * dt);
    }
    
    /// Single time step of elastic wave propagation
    pub fn step(&mut self, dt: f64, grid: &Grid) {
        // Update stress based on current displacement
        self.update_stress(dt, grid);
        
        // Update velocity using stress gradients
        self.update_velocity(dt, grid);
        
        // Update displacement using velocity
        self.update_displacement(dt);
    }
    
    /// Get displacement field component
    pub fn get_displacement(&self, component: usize) -> Array3<f64> {
        self.displacement.index_axis(Axis(0), component).to_owned()
    }
    
    /// Get velocity field component
    pub fn get_velocity(&self, component: usize) -> Array3<f64> {
        self.velocity.index_axis(Axis(0), component).to_owned()
    }
    
    /// Get stress tensor component
    pub fn get_stress(&self, component: usize) -> Array3<f64> {
        self.stress.index_axis(Axis(0), component).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_elastic_wave_initialization() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
        let wave = ElasticWave::new(&grid);
        
        assert_eq!(wave.displacement.shape(), &[3, 10, 10, 10]);
        assert_eq!(wave.velocity.shape(), &[3, 10, 10, 10]);
        assert_eq!(wave.stress.shape(), &[6, 10, 10, 10]);
    }
    
    #[test]
    fn test_stress_strain_relationship() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
        let mut wave = ElasticWave::new(&grid);
        
        // Set uniform elastic parameters
        let lambda = Array3::ones((10, 10, 10)) * 2.0e9;  // 2 GPa
        let mu = Array3::ones((10, 10, 10)) * 1.0e9;      // 1 GPa
        wave.lambda = lambda;
        wave.mu = mu;
        
        // Apply uniform strain
        wave.displacement.slice_mut(s![0, .., .., ..]).fill(0.01);  // 1% strain in x
        
        // Update stress
        wave.update_stress(1e-6, &grid);
        
        // Check stress components
        let sigma_xx = wave.get_stress(0);
        let expected_stress = 2.0e9 * 0.01;  // λε for uniform strain
        
        assert_relative_eq!(
            sigma_xx[[5, 5, 5]],
            expected_stress,
            max_relative = 0.01
        );
    }
}
