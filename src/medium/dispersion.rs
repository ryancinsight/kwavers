use super::Medium;
use crate::grid::Grid;
use ndarray::Array3;
use num_complex::Complex64;

/// Trait for media with frequency-dependent properties (dispersion)
pub trait DispersiveMedium: Medium {
    /// Get complex wave number at a specific frequency
    fn get_wave_number(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> Complex64 {
        let omega = 2.0 * std::f64::consts::PI * frequency;
        let c0 = self.sound_speed(x, y, z, grid);
        let rho0 = self.density(x, y, z, grid);
        let alpha = self.absorption_coefficient(x, y, z, grid, frequency);
        
        // Convert from dB/MHz/cm to Np/m
        let alpha_np = alpha * 0.115_f64;
        
        // Complex wave number including absorption
        let k_real = omega / c0;
        let k_imag = alpha_np;
        
        Complex64::new(k_real, k_imag)
    }
    
    /// Get complex density at a specific frequency
    fn get_complex_density(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> Complex64 {
        // Default implementation assumes frequency-independent density
        Complex64::new(self.density(x, y, z, grid), 0.0)
    }
    
    /// Get complex compressibility at a specific frequency
    fn get_complex_compressibility(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> Complex64 {
        let c0 = self.sound_speed(x, y, z, grid);
        let rho0 = self.density(x, y, z, grid);
        let kappa0 = 1.0 / (rho0 * c0 * c0);
        
        // Default implementation assumes frequency-independent compressibility
        Complex64::new(kappa0, 0.0)
    }
    
    /// Get attenuation coefficient at a specific frequency
    fn get_attenuation(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        self.absorption_coefficient(x, y, z, grid, frequency)
    }
    
    /// Get phase velocity at a specific frequency
    fn get_phase_velocity(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        let k = self.get_wave_number(x, y, z, grid, frequency);
        let omega = 2.0 * std::f64::consts::PI * frequency;
        omega / k.re
    }
    
    /// Get group velocity at a specific frequency
    fn get_group_velocity(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        // Numerical approximation of group velocity using central difference
        let df = frequency * 0.001;  // Small frequency step
        let k1 = self.get_wave_number(x, y, z, grid, frequency - df);
        let k2 = self.get_wave_number(x, y, z, grid, frequency + df);
        
        let domega = 4.0 * std::f64::consts::PI * df;
        let dk = k2.re - k1.re;
        
        domega / dk
    }
}

/// Implementation for power law absorption with dispersion
pub struct PowerLawDispersiveMedium {
    density: Array3<f64>,
    sound_speed: Array3<f64>,
    alpha0: f64,      // Power law absorption coefficient
    y: f64,           // Power law exponent
}

impl PowerLawDispersiveMedium {
    pub fn new(
        density: Array3<f64>,
        sound_speed: Array3<f64>,
        alpha0: f64,
        y: f64,
    ) -> Self {
        Self {
            density,
            sound_speed,
            alpha0,
            y,
        }
    }
}

impl Medium for PowerLawDispersiveMedium {
    fn density(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let i = grid.x_idx(x);
        let j = grid.y_idx(y);
        let k = grid.z_idx(z);
        self.density[[i, j, k]]
    }
    
    fn sound_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let i = grid.x_idx(x);
        let j = grid.y_idx(y);
        let k = grid.z_idx(z);
        self.sound_speed[[i, j, k]]
    }
    
    fn absorption_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid, frequency: f64) -> f64 {
        // Power law absorption: α = α₀ * f^y
        self.alpha0 * frequency.powf(self.y)
    }
    
    // Implement other required Medium trait methods with default values
    fn is_homogeneous(&self) -> bool { false }
    fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.0 }
    fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.0 }
    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 101325.0 }
    fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2300.0 }
    fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.4 }
    fn specific_heat(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 4186.0 }
    fn thermal_conductivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.6 }
    fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2.1e-4 }
    fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2.0e-5 }
    fn thermal_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.4e-7 }
    fn nonlinearity_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 3.5 }
    fn absorption_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.1 }
    fn reduced_scattering_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 10.0 }
    fn reference_frequency(&self) -> f64 { 1.0e6 }
    
    fn update_temperature(&mut self, _temperature: &Array3<f64>) {}
    fn temperature(&self) -> &Array3<f64> { unimplemented!() }
    fn bubble_radius(&self) -> &Array3<f64> { unimplemented!() }
    fn bubble_velocity(&self) -> &Array3<f64> { unimplemented!() }
    fn update_bubble_state(&mut self, _radius: &Array3<f64>, _velocity: &Array3<f64>) {}
    fn density_array(&self) -> Array3<f64> { self.density.clone() }
    fn sound_speed_array(&self) -> Array3<f64> { self.sound_speed.clone() }
}

impl DispersiveMedium for PowerLawDispersiveMedium {
    // Use default implementations from trait
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_power_law_dispersion() {
        let nx = 10;
        let ny = 10;
        let nz = 10;
        let grid = Grid::new(nx, ny, nz, 0.001, 0.001, 0.001);
        
        let density = Array3::<f64>::ones((nx, ny, nz)) * 1000.0;
        let sound_speed = Array3::<f64>::ones((nx, ny, nz)) * 1500.0;
        
        let medium = PowerLawDispersiveMedium::new(
            density,
            sound_speed,
            0.5,    // α₀
            1.1,    // y
        );
        
        let f1 = 1.0e6;  // 1 MHz
        let f2 = 2.0e6;  // 2 MHz
        
        // Test power law absorption scaling
        let alpha1 = medium.get_attenuation(0.0, 0.0, 0.0, &grid, f1);
        let alpha2 = medium.get_attenuation(0.0, 0.0, 0.0, &grid, f2);
        
        // α₂/α₁ should equal (f₂/f₁)^y
        let expected_ratio = (f2/f1).powf(1.1);
        assert_relative_eq!(alpha2/alpha1, expected_ratio, max_relative = 1e-10);
        
        // Test wave number calculation
        let k = medium.get_wave_number(0.0, 0.0, 0.0, &grid, f1);
        assert!(k.im > 0.0);  // Should have positive attenuation
        
        // Test phase velocity
        let vp = medium.get_phase_velocity(0.0, 0.0, 0.0, &grid, f1);
        assert_relative_eq!(vp, 1500.0, max_relative = 1e-3);  // Should be close to c₀ at reference frequency
    }
}
