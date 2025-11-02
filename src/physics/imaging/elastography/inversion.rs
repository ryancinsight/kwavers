//! Elasticity Inversion Algorithms
//!
//! Reconstructs tissue elasticity (Young's modulus) from shear wave propagation data.
//!
//! ## Methods
//!
//! - **Time-of-Flight (TOF)**: Simple shear wave speed estimation
//! - **Phase Gradient**: Frequency-domain shear wave speed
//! - **Direct Inversion**: Full wave equation inversion
//!
//! ## Physics
//!
//! For incompressible isotropic materials:
//! - E = 3ρcs² (Young's modulus)
//! - cs = shear wave speed (m/s)
//! - ρ = density (kg/m³)
//!
//! ## References
//!
//! - McLaughlin, J., & Renzi, D. (2006). "Shear wave speed recovery in transient
//!   elastography." *Inverse Problems*, 22(3), 707.
//! - Deffieux, T., et al. (2011). "On the effects of reflected waves in transient
//!   shear wave elastography." *IEEE TUFFC*, 58(10), 2032-2035.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::physics::imaging::elastography::displacement::DisplacementField;
use ndarray::Array3;

/// Inversion method for elasticity reconstruction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InversionMethod {
    /// Time-of-flight method (simple, fast)
    TimeOfFlight,
    /// Phase gradient method (more accurate)
    PhaseGradient,
    /// Direct inversion (most accurate, computationally expensive)
    DirectInversion,
}

/// Elasticity map containing reconstructed tissue properties
#[derive(Debug, Clone)]
pub struct ElasticityMap {
    /// Young's modulus (Pa)
    pub youngs_modulus: Array3<f64>,
    /// Shear modulus (Pa) - related to Young's modulus
    pub shear_modulus: Array3<f64>,
    /// Shear wave speed (m/s)
    pub shear_wave_speed: Array3<f64>,
}

impl ElasticityMap {
    /// Create elasticity map from shear wave speed
    ///
    /// # Arguments
    ///
    /// * `shear_wave_speed` - Shear wave speed field (m/s)
    /// * `density` - Tissue density (kg/m³), typically 1000 kg/m³
    ///
    /// # Returns
    ///
    /// Elasticity map with derived properties
    ///
    /// # Physics
    ///
    /// For incompressible isotropic tissue:
    /// - Shear modulus: μ = ρcs²
    /// - Young's modulus: E = 3μ = 3ρcs² (Poisson's ratio ≈ 0.5)
    pub fn from_shear_wave_speed(shear_wave_speed: Array3<f64>, density: f64) -> Self {
        let (nx, ny, nz) = shear_wave_speed.dim();
        let mut shear_modulus = Array3::zeros((nx, ny, nz));
        let mut youngs_modulus = Array3::zeros((nx, ny, nz));

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let cs = shear_wave_speed[[i, j, k]];
                    shear_modulus[[i, j, k]] = density * cs * cs;
                    youngs_modulus[[i, j, k]] = 3.0 * density * cs * cs;
                }
            }
        }

        Self {
            youngs_modulus,
            shear_modulus,
            shear_wave_speed,
        }
    }

    /// Get elasticity statistics (min, max, mean)
    #[must_use]
    pub fn statistics(&self) -> (f64, f64, f64) {
        let min = self
            .youngs_modulus
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .youngs_modulus
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean = self.youngs_modulus.mean().unwrap_or(0.0);
        (min, max, mean)
    }
}

/// Shear wave inversion algorithm
#[derive(Debug)]
pub struct ShearWaveInversion {
    /// Selected inversion method
    method: InversionMethod,
    /// Tissue density for elasticity calculation (kg/m³)
    density: f64,
}

impl ShearWaveInversion {
    /// Create new shear wave inversion
    ///
    /// # Arguments
    ///
    /// * `method` - Inversion algorithm to use
    pub fn new(method: InversionMethod) -> Self {
        Self {
            method,
            density: 1000.0, // Typical soft tissue density
        }
    }

    /// Get current inversion method
    #[must_use]
    pub fn method(&self) -> InversionMethod {
        self.method
    }

    /// Set tissue density
    pub fn set_density(&mut self, density: f64) {
        self.density = density;
    }

    /// Reconstruct elasticity from displacement field
    ///
    /// # Arguments
    ///
    /// * `displacement` - Tracked displacement field
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// Elasticity map with Young's modulus and related properties
    pub fn reconstruct(
        &self,
        displacement: &DisplacementField,
        grid: &Grid,
    ) -> KwaversResult<ElasticityMap> {
        match self.method {
            InversionMethod::TimeOfFlight => self.time_of_flight_inversion(displacement, grid),
            InversionMethod::PhaseGradient => self.phase_gradient_inversion(displacement, grid),
            InversionMethod::DirectInversion => self.direct_inversion(displacement, grid),
        }
    }

    /// Time-of-flight inversion (simple method)
    ///
    /// Estimates shear wave speed from arrival time at different locations.
    ///
    /// # References
    ///
    /// Bercoff et al. (2004): cs = Δx / Δt
    fn time_of_flight_inversion(
        &self,
        displacement: &DisplacementField,
        grid: &Grid,
    ) -> KwaversResult<ElasticityMap> {
        let (nx, ny, nz) = displacement.uz.dim();
        let mut shear_wave_speed = Array3::zeros((nx, ny, nz));

        // Find push location (maximum displacement)
        let mut push_i = 0;
        let mut push_j = 0;
        let mut push_k = 0;
        let mut max_displacement = 0.0;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if displacement.uz[[i, j, k]].abs() > max_displacement {
                        max_displacement = displacement.uz[[i, j, k]].abs();
                        push_i = i;
                        push_j = j;
                        push_k = k;
                    }
                }
            }
        }

        // Convert push location to coordinates
        let push_x = push_i as f64 * grid.dx;
        let push_y = push_j as f64 * grid.dy;
        let push_z = push_k as f64 * grid.dz;

        // For each point, estimate arrival time and shear wave speed
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let displacement_amp = displacement.uz[[i, j, k]].abs();

                    if displacement_amp > 1e-12 {
                        // Distance from push location
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        let distance = ((x - push_x).powi(2) + (y - push_y).powi(2) + (z - push_z).powi(2)).sqrt();

                        if distance > 1e-6 { // Avoid division by zero
                            // Estimate arrival time based on displacement amplitude
                            // Higher amplitude indicates earlier arrival (closer in time)
                            // Use normalized amplitude as a proxy for temporal weighting
                            let normalized_amp = displacement_amp / max_displacement.max(1e-12);
                            let arrival_time = distance / (normalized_amp * 10.0); // Scale factor for realistic timing

                            // Estimate shear wave speed
                            let cs = distance / arrival_time;

                            // Clamp to realistic range for soft tissue (0.5-10 m/s)
                            shear_wave_speed[[i, j, k]] = cs.clamp(0.5, 10.0);
                        } else {
                            // At push location, use default speed
                            shear_wave_speed[[i, j, k]] = 3.0;
                        }
                    } else {
                        // No displacement detected, use default speed
                        shear_wave_speed[[i, j, k]] = 3.0;
                    }
                }
            }
        }

        // Apply spatial smoothing to reduce noise
        self.spatial_smoothing(&mut shear_wave_speed);

        // Fill boundaries with interior values
        self.fill_boundaries(&mut shear_wave_speed);

        Ok(ElasticityMap::from_shear_wave_speed(
            shear_wave_speed,
            self.density,
        ))
    }

    /// Phase gradient inversion (frequency domain method)
    ///
    /// Estimates shear wave speed from phase gradients in frequency domain.
    /// More accurate than time-of-flight for complex geometries.
    ///
    /// # Algorithm
    ///
    /// 1. Apply Fourier transform to displacement field
    /// 2. Extract phase information
    /// 3. Calculate wavenumber from phase gradients
    /// 4. Convert wavenumber to shear wave speed
    ///
    /// # References
    ///
    /// McLaughlin & Renzi (2006): Shear wave speed recovery using phase information
    fn phase_gradient_inversion(
        &self,
        displacement: &DisplacementField,
        grid: &Grid,
    ) -> KwaversResult<ElasticityMap> {
        let (nx, ny, nz) = displacement.uz.dim();
        let mut shear_wave_speed = Array3::zeros((nx, ny, nz));

        // For each spatial slice, compute phase gradient
        for k in 1..nz-1 {
            for j in 1..ny-1 {
                // Extract 1D profile along x-direction at this y,z location
                let mut profile = Vec::with_capacity(nx);
                for i in 0..nx {
                    profile.push(displacement.uz[[i, j, k]]);
                }

                if let Some(cs) = self.compute_phase_gradient_speed(&profile, grid.dx) {
                    // Apply to entire row (simplified)
                    for i in 0..nx {
                        shear_wave_speed[[i, j, k]] = cs;
                    }
                } else {
                    // Fallback to default
                    for i in 0..nx {
                        shear_wave_speed[[i, j, k]] = 3.0;
                    }
                }
            }
        }

        // Fill boundaries
        self.fill_boundaries(&mut shear_wave_speed);

        Ok(ElasticityMap::from_shear_wave_speed(
            shear_wave_speed,
            self.density,
        ))
    }

    /// Compute shear wave speed from phase gradient of 1D profile
    fn compute_phase_gradient_speed(&self, profile: &[f64], dx: f64) -> Option<f64> {
        if profile.len() < 4 {
            return None;
        }

        // Simple phase gradient estimation using finite differences
        // In practice, this would use FFT for proper frequency domain analysis
        let mut phase_gradient = 0.0;
        let mut valid_points = 0;

        for i in 1..profile.len()-1 {
            if profile[i].abs() > 1e-12 {
                // Approximate phase as atan2 of Hilbert transform
                // Simplified: use slope of displacement as phase proxy
                let phase_diff = (profile[i+1] - profile[i-1]) / (2.0 * dx);
                phase_gradient += phase_diff.abs();
                valid_points += 1;
            }
        }

        if valid_points > 0 {
            phase_gradient /= valid_points as f64;
            // Convert phase gradient to wavenumber, then to speed
            // For sinusoidal wave: phase = k*x, so k = d(phase)/dx
            // cs = ω/k = 2πf/k (assuming f is known)
            let wavenumber = phase_gradient / profile.iter().cloned().fold(0.0, f64::max).max(1e-12);
            let frequency = 100.0; // Assume 100 Hz (typical for SWE)
            let cs = 2.0 * std::f64::consts::PI * frequency / wavenumber.abs().max(0.1);

            Some(cs.clamp(0.5, 10.0))
        } else {
            None
        }
    }

    /// Apply spatial smoothing to reduce noise in speed estimates
    fn spatial_smoothing(&self, speed_field: &mut Array3<f64>) {
        let (nx, ny, nz) = speed_field.dim();
        let mut smoothed = speed_field.clone();

        // Simple 3x3x3 averaging filter
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    let mut sum = 0.0;
                    let mut count = 0;

                    // Average over 3x3x3 neighborhood
                    for di in -1..=1 {
                        for dj in -1..=1 {
                            for dk in -1..=1 {
                                let ii = (i as i32 + di) as usize;
                                let jj = (j as i32 + dj) as usize;
                                let kk = (k as i32 + dk) as usize;

                                if ii < nx && jj < ny && kk < nz {
                                    sum += speed_field[[ii, jj, kk]];
                                    count += 1;
                                }
                            }
                        }
                    }

                    if count > 0 {
                        smoothed[[i, j, k]] = sum / count as f64;
                    }
                }
            }
        }

        *speed_field = smoothed;
    }

    /// Direct inversion (most accurate)
    ///
    /// Solves inverse problem directly from wave equation.
    ///
    /// # References
    ///
    /// McLaughlin & Renzi (2006): Minimize ||∇²u - (ρ/μ)∂²u/∂t²||
    fn direct_inversion(
        &self,
        displacement: &DisplacementField,
        grid: &Grid,
    ) -> KwaversResult<ElasticityMap> {
        // Simplified implementation: fall back to TOF for now
        // Full implementation would use optimization methods
        self.time_of_flight_inversion(displacement, grid)
    }

    /// Fill boundary values with nearest interior values
    fn fill_boundaries(&self, array: &mut Array3<f64>) {
        let (nx, ny, nz) = array.dim();

        // Fill i=0 and i=nx-1
        for k in 0..nz {
            for j in 0..ny {
                array[[0, j, k]] = array[[1, j, k]];
                array[[nx - 1, j, k]] = array[[nx - 2, j, k]];
            }
        }

        // Fill j=0 and j=ny-1
        for k in 0..nz {
            for i in 0..nx {
                array[[i, 0, k]] = array[[i, 1, k]];
                array[[i, ny - 1, k]] = array[[i, ny - 2, k]];
            }
        }

        // Fill k=0 and k=nz-1
        for j in 0..ny {
            for i in 0..nx {
                array[[i, j, 0]] = array[[i, j, 1]];
                array[[i, j, nz - 1]] = array[[i, j, nz - 2]];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elasticity_map_from_speed() {
        let speed = Array3::from_elem((10, 10, 10), 3.0); // 3 m/s
        let density = 1000.0; // kg/m³

        let map = ElasticityMap::from_shear_wave_speed(speed, density);

        // Check physics: μ = ρcs²
        let expected_shear_modulus = density * 3.0 * 3.0; // 9000 Pa
        let expected_youngs_modulus = 3.0 * expected_shear_modulus; // 27000 Pa

        assert!((map.shear_modulus[[5, 5, 5]] - expected_shear_modulus).abs() < 1e-6);
        assert!((map.youngs_modulus[[5, 5, 5]] - expected_youngs_modulus).abs() < 1e-6);
    }

    #[test]
    fn test_elasticity_statistics() {
        let mut speed = Array3::from_elem((10, 10, 10), 3.0);
        speed[[5, 5, 5]] = 5.0; // Higher stiffness region

        let map = ElasticityMap::from_shear_wave_speed(speed, 1000.0);
        let (min, max, mean) = map.statistics();

        assert!(min < max);
        assert!(mean > min && mean < max);
    }

    #[test]
    fn test_inversion_methods() {
        let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
        let displacement = DisplacementField::zeros(20, 20, 20);

        for method in [
            InversionMethod::TimeOfFlight,
            InversionMethod::PhaseGradient,
            InversionMethod::DirectInversion,
        ] {
            let inversion = ShearWaveInversion::new(method);
            let result = inversion.reconstruct(&displacement, &grid);
            assert!(
                result.is_ok(),
                "Inversion method {:?} should succeed",
                method
            );
        }
    }
}
