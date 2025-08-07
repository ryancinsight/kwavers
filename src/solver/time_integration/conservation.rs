//! Conservation properties for multi-rate time integration
//! 
//! This module ensures conservation of mass, momentum, and energy
//! during multi-rate time integration by implementing conservative
//! coupling strategies and monitoring conservation errors.
//!
//! References:
//! - Constantinescu, E. M., & Sandu, A. (2007). "Multirate timestepping
//!   methods for hyperbolic conservation laws" Journal of Scientific
//!   Computing, 33(3), 239-278.

use crate::{KwaversResult, KwaversError, ValidationError};
use crate::Grid;
use ndarray::{Array3, Zip};
use std::collections::HashMap;

/// Conservation quantities for monitoring
#[derive(Debug, Clone)]
pub struct ConservedQuantities {
    /// Total mass
    pub mass: f64,
    /// Total momentum (x, y, z components)
    pub momentum: (f64, f64, f64),
    /// Total energy
    pub energy: f64,
    /// Total angular momentum
    pub angular_momentum: (f64, f64, f64),
}

/// Conservation monitor for multi-rate integration
pub struct ConservationMonitor {
    /// Initial conserved quantities
    initial_quantities: Option<ConservedQuantities>,
    /// History of conservation errors
    error_history: Vec<ConservationError>,
    /// Tolerance for conservation violations
    tolerance: f64,
    /// Adiabatic index (gamma) for the medium
    gamma: f64,
}

/// Conservation error at a time step
#[derive(Debug, Clone)]
pub struct ConservationError {
    /// Time at which error was measured
    pub time: f64,
    /// Relative mass error
    pub mass_error: f64,
    /// Relative momentum error
    pub momentum_error: f64,
    /// Relative energy error
    pub energy_error: f64,
    /// Relative angular momentum error
    pub angular_momentum_error: f64,
}

impl ConservationMonitor {
    /// Create a new conservation monitor
    pub fn new(tolerance: f64) -> Self {
        Self::with_gamma(tolerance, 1.4) // Default to air
    }
    
    /// Create a new conservation monitor with specified gamma
    pub fn with_gamma(tolerance: f64, gamma: f64) -> Self {
        Self {
            initial_quantities: None,
            error_history: Vec::new(),
            tolerance,
            gamma,
        }
    }
    
    /// Initialize conservation monitoring
    pub fn initialize(
        &mut self,
        fields: &HashMap<String, Array3<f64>>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let quantities = self.compute_conserved_quantities(fields, grid)?;
        self.initial_quantities = Some(quantities);
        Ok(())
    }
    
    /// Check conservation and return any violations
    pub fn check_conservation(
        &mut self,
        fields: &HashMap<String, Array3<f64>>,
        grid: &Grid,
        time: f64,
    ) -> KwaversResult<Option<ConservationError>> {
        let initial = self.initial_quantities.as_ref()
            .ok_or_else(|| KwaversError::Validation(ValidationError::FieldValidation {
                field: "initial_quantities".to_string(),
                value: "None".to_string(),
                constraint: "Must initialize conservation monitor first".to_string(),
            }))?;
        
        let current = self.compute_conserved_quantities(fields, grid)?;
        
        // Compute relative errors
        let mass_error = (current.mass - initial.mass).abs() / initial.mass.max(1e-10);
        
        let momentum_mag_initial = (initial.momentum.0.powi(2) + 
                                   initial.momentum.1.powi(2) + 
                                   initial.momentum.2.powi(2)).sqrt();
        let momentum_mag_current = (current.momentum.0.powi(2) + 
                                   current.momentum.1.powi(2) + 
                                   current.momentum.2.powi(2)).sqrt();
        let momentum_error = (momentum_mag_current - momentum_mag_initial).abs() 
                           / momentum_mag_initial.max(1e-10);
        
        let energy_error = (current.energy - initial.energy).abs() / initial.energy.max(1e-10);
        
        let ang_mom_mag_initial = (initial.angular_momentum.0.powi(2) + 
                                  initial.angular_momentum.1.powi(2) + 
                                  initial.angular_momentum.2.powi(2)).sqrt();
        let ang_mom_mag_current = (current.angular_momentum.0.powi(2) + 
                                  current.angular_momentum.1.powi(2) + 
                                  current.angular_momentum.2.powi(2)).sqrt();
        let angular_momentum_error = (ang_mom_mag_current - ang_mom_mag_initial).abs() 
                                   / ang_mom_mag_initial.max(1e-10);
        
        let error = ConservationError {
            time,
            mass_error,
            momentum_error,
            energy_error,
            angular_momentum_error,
        };
        
        self.error_history.push(error.clone());
        
        // Check if any conservation law is violated
        if mass_error > self.tolerance || 
           momentum_error > self.tolerance ||
           energy_error > self.tolerance ||
           angular_momentum_error > self.tolerance {
            Ok(Some(error))
        } else {
            Ok(None)
        }
    }
    
    /// Compute conserved quantities from fields
    fn compute_conserved_quantities(
        &self,
        fields: &HashMap<String, Array3<f64>>,
        grid: &Grid,
    ) -> KwaversResult<ConservedQuantities> {
        let dv = grid.dx * grid.dy * grid.dz;
        
        // Get density field (assuming it exists)
        let density = fields.get("density")
            .or_else(|| fields.get("rho"))
            .ok_or_else(|| KwaversError::Validation(ValidationError::FieldValidation {
                field: "density".to_string(),
                value: "missing".to_string(),
                constraint: "Density field required for conservation monitoring".to_string(),
            }))?;
        
        // Compute total mass
        let mass = density.iter().sum::<f64>() * dv;
        
        // Get velocity fields if available
        let (momentum, angular_momentum) = if let (Some(vx), Some(vy), Some(vz)) = 
            (fields.get("velocity_x"), fields.get("velocity_y"), fields.get("velocity_z")) {
            
            let mut px = 0.0;
            let mut py = 0.0;
            let mut pz = 0.0;
            let mut lx = 0.0;
            let mut ly = 0.0;
            let mut lz = 0.0;
            
            let (nx, ny, nz) = density.dim();
            
            // Compute momentum and angular momentum using iterators
            Zip::indexed(&density)
                .and(&vx)
                .and(&vy)
                .and(&vz)
                .for_each(|(i, j, k), &rho, &vx_val, &vy_val, &vz_val| {
                    // Linear momentum
                    px += rho * vx_val * dv;
                    py += rho * vy_val * dv;
                    pz += rho * vz_val * dv;

                    // Position relative to grid center
                    let x = (i as f64 - nx as f64 / 2.0) * grid.dx;
                    let y = (j as f64 - ny as f64 / 2.0) * grid.dy;
                    let z = (k as f64 - nz as f64 / 2.0) * grid.dz;

                    // Angular momentum L = r × p
                    lx += rho * (y * vz_val - z * vy_val) * dv;
                    ly += rho * (z * vx_val - x * vz_val) * dv;
                    lz += rho * (x * vy_val - y * vx_val) * dv;
                });
            
            ((px, py, pz), (lx, ly, lz))
        } else {
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        };
        
        // Compute total energy
        let energy = if let Some(pressure) = fields.get("pressure") {
            // E = kinetic + internal energy
            let mut total_energy = 0.0;
            
            if let (Some(vx), Some(vy), Some(vz)) = 
                (fields.get("velocity_x"), fields.get("velocity_y"), fields.get("velocity_z")) {
                
                let gamma_minus_one = self.gamma - 1.0;
                
                Zip::from(density)
                    .and(vx)
                    .and(vy)
                    .and(vz)
                    .and(pressure)
                    .for_each(|&rho, &vx, &vy, &vz, &p| {
                        // Kinetic energy: 0.5 * rho * v²
                        let kinetic = 0.5 * rho * (vx*vx + vy*vy + vz*vz);
                        // Internal energy: p / (gamma - 1) for ideal gas
                        // For liquids, a different equation of state may be needed
                        let internal = p / gamma_minus_one;
                        total_energy += (kinetic + internal) * dv;
                    });
            } else {
                // Just internal energy
                let gamma_minus_one = self.gamma - 1.0;
                total_energy = pressure.iter().sum::<f64>() * dv / gamma_minus_one;
            }
            
            total_energy
        } else {
            0.0
        };
        
        Ok(ConservedQuantities {
            mass,
            momentum,
            energy,
            angular_momentum,
        })
    }
    
    /// Get conservation error history
    pub fn error_history(&self) -> &[ConservationError] {
        &self.error_history
    }
    
    /// Get maximum conservation error
    pub fn max_error(&self) -> Option<ConservationError> {
        self.error_history.iter()
            .max_by(|a, b| {
                let a_max = a.mass_error.max(a.momentum_error)
                    .max(a.energy_error).max(a.angular_momentum_error);
                let b_max = b.mass_error.max(b.momentum_error)
                    .max(b.energy_error).max(b.angular_momentum_error);
                a_max.partial_cmp(&b_max).unwrap()
            })
            .cloned()
    }
    
    /// Set the adiabatic index for the medium
    pub fn set_gamma(&mut self, gamma: f64) {
        self.gamma = gamma;
    }
    
    /// Get gamma value for common media
    pub fn gamma_for_medium(medium: &str) -> f64 {
        match medium.to_lowercase().as_str() {
            "air" => 1.4,
            "water" | "liquid" => 7.15, // Tait equation parameter for water
            "tissue" => 4.0, // Approximate for soft tissue
            "helium" => 1.66,
            "argon" => 1.67,
            _ => 1.4, // Default to air
        }
    }
}

/// Conservative coupling interface for multi-rate integration
pub trait ConservativeCoupling {
    /// Apply conservative coupling between fast and slow components
    fn apply_conservative_coupling(
        &self,
        fast_fields: &mut HashMap<String, Array3<f64>>,
        slow_fields: &mut HashMap<String, Array3<f64>>,
        dt: f64,
        grid: &Grid,
    ) -> KwaversResult<()>;
    
    /// Compute flux corrections for conservation
    fn compute_flux_corrections(
        &self,
        fields: &HashMap<String, Array3<f64>>,
        grid: &Grid,
    ) -> KwaversResult<HashMap<String, Array3<f64>>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Grid;
    
    #[test]
    fn test_gamma_for_medium() {
        assert!((ConservationMonitor::gamma_for_medium("air") - 1.4).abs() < 1e-10);
        assert!((ConservationMonitor::gamma_for_medium("water") - 7.15).abs() < 1e-10);
        assert!((ConservationMonitor::gamma_for_medium("tissue") - 4.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_conservation_with_different_gamma() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        
        // Create fields
        let mut fields = HashMap::new();
        let density = Array3::from_elem((10, 10, 10), 1000.0); // kg/m³
        let pressure = Array3::from_elem((10, 10, 10), 1e5); // Pa
        fields.insert("density".to_string(), density);
        fields.insert("pressure".to_string(), pressure);
        
        // Test with air
        let mut monitor_air = ConservationMonitor::with_gamma(1e-10, 1.4);
        monitor_air.initialize(&fields, &grid).unwrap();
        let quantities_air = monitor_air.compute_conserved_quantities(&fields, &grid).unwrap();
        
        // Test with water
        let mut monitor_water = ConservationMonitor::with_gamma(1e-10, 7.15);
        monitor_water.initialize(&fields, &grid).unwrap();
        let quantities_water = monitor_water.compute_conserved_quantities(&fields, &grid).unwrap();
        
        // Energy should be different due to different gamma
        assert!((quantities_air.energy - quantities_water.energy).abs() > 1e-6);
        
        // But mass should be the same
        assert!((quantities_air.mass - quantities_water.mass).abs() < 1e-10);
    }
}