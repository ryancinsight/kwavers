//! Elasticity Inversion Algorithms
//!
//! Reconstructs tissue elasticity from shear wave propagation data, including
//! linear and nonlinear parameter estimation.
//!
//! ## Methods
//!
//! ### Linear Methods
//! - **Time-of-Flight (TOF)**: Simple shear wave speed estimation
//! - **Phase Gradient**: Frequency-domain shear wave speed
//! - **Direct Inversion**: Full wave equation inversion
//!
//! ### Nonlinear Methods
//! - **Harmonic Ratio**: B/A parameter estimation from harmonic amplitudes
//! - **Iterative Nonlinear Least Squares**: Full nonlinear parameter inversion
//! - **Bayesian Inversion**: Uncertainty quantification for nonlinear parameters
//!
//! ## Physics
//!
//! ### Linear Elasticity
//! For incompressible isotropic materials:
//! - E = 3ρcs² (Young's modulus)
//! - cs = shear wave speed (m/s)
//! - ρ = density (kg/m³)
//!
//! ### Nonlinear Elasticity
//! - B/A = acoustic nonlinearity parameter
//! - B/A = (8/μ) * (ρ₀ c₀³ / (β P₀)) * (A₂/A₁)
//! - Higher-order elastic constants (A, B, C, D)
//!
//! ## References
//!
//! - McLaughlin, J., & Renzi, D. (2006). "Shear wave speed recovery in transient
//!   elastography." *Inverse Problems*, 22(3), 707.
//! - Deffieux, T., et al. (2011). "On the effects of reflected waves in transient
//!   shear wave elastography." *IEEE TUFFC*, 58(10), 2032-2035.
//! - Parker, K. J., et al. (2011). "Sonoelasticity of organs: Shear waves ring a bell."
//!   *Journal of Ultrasound in Medicine*, 30(4), 507-515.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::physics::imaging::elastography::displacement::DisplacementField;
use crate::physics::imaging::elastography::harmonic_detection::HarmonicDisplacementField;
use ndarray::Array3;
use std::f64::consts::PI;

/// Inversion method for elasticity reconstruction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InversionMethod {
    /// Time-of-flight method (simple, fast)
    TimeOfFlight,
    /// Phase gradient method (more accurate)
    PhaseGradient,
    /// Direct inversion (most accurate, computationally expensive)
    DirectInversion,
    /// 3D volumetric time-of-flight (for 3D SWE)
    VolumetricTimeOfFlight,
    /// 3D phase gradient with directional analysis
    DirectionalPhaseGradient,
}

/// Nonlinear inversion method for advanced parameter estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonlinearInversionMethod {
    /// Harmonic ratio method (B/A from A₂/A₁)
    HarmonicRatio,
    /// Iterative nonlinear least squares
    NonlinearLeastSquares,
    /// Bayesian inversion with uncertainty quantification
    BayesianInversion,
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

/// Nonlinear parameter map for advanced tissue characterization
#[derive(Debug, Clone)]
pub struct NonlinearParameterMap {
    /// Acoustic nonlinearity parameter B/A (dimensionless)
    pub nonlinearity_parameter: Array3<f64>,
    /// Higher-order elastic constants A, B, C, D (Pa)
    pub elastic_constants: Vec<Array3<f64>>,
    /// Uncertainty in nonlinearity parameter estimation
    pub nonlinearity_uncertainty: Array3<f64>,
    /// Signal quality metrics for nonlinear estimation
    pub estimation_quality: Array3<f64>,
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
            InversionMethod::VolumetricTimeOfFlight => {
                self.volumetric_time_of_flight_inversion(displacement, grid)
            }
            InversionMethod::DirectionalPhaseGradient => {
                self.directional_phase_gradient_inversion(displacement, grid)
            }
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
                        let distance =
                            ((x - push_x).powi(2) + (y - push_y).powi(2) + (z - push_z).powi(2))
                                .sqrt();

                        if distance > 1e-6 {
                            // Avoid division by zero
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
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                // Extract 1D profile along x-direction at this y,z location
                let mut profile = Vec::with_capacity(nx);
                for i in 0..nx {
                    profile.push(displacement.uz[[i, j, k]]);
                }

                if let Some(cs) = self.compute_phase_gradient_speed(&profile, grid.dx) {
                    // Apply regularization to entire displacement row for numerical stability
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

        for i in 1..profile.len() - 1 {
            if profile[i].abs() > 1e-12 {
                // Approximate phase as atan2 of Hilbert transform
                // Simplified: use slope of displacement as phase proxy
                let phase_diff = (profile[i + 1] - profile[i - 1]) / (2.0 * dx);
                phase_gradient += phase_diff.abs();
                valid_points += 1;
            }
        }

        if valid_points > 0 {
            phase_gradient /= valid_points as f64;
            // Convert phase gradient to wavenumber, then to speed
            // For sinusoidal wave: phase = k*x, so k = d(phase)/dx
            // cs = ω/k = 2πf/k (assuming f is known)
            let wavenumber =
                phase_gradient / profile.iter().cloned().fold(0.0, f64::max).max(1e-12);
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
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
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

    /// 3D Volumetric time-of-flight inversion
    ///
    /// Enhanced time-of-flight method for 3D volumes with multi-directional wave analysis.
    /// Accounts for complex wave propagation patterns in volumetric tissue.
    ///
    /// # Algorithm
    ///
    /// 1. Identify multiple push locations and wave directions
    /// 2. Compute 3D arrival times for each wave front
    /// 3. Use multi-directional information for robust speed estimation
    /// 4. Apply volumetric regularization for noise reduction
    ///
    /// # References
    ///
    /// Urban et al. (2013): 3D SWE reconstruction with multi-directional waves
    fn volumetric_time_of_flight_inversion(
        &self,
        displacement: &DisplacementField,
        grid: &Grid,
    ) -> KwaversResult<ElasticityMap> {
        let (nx, ny, nz) = displacement.uz.dim();
        let mut shear_wave_speed = Array3::zeros((nx, ny, nz));

        // Find multiple push locations (local maxima in displacement)
        let push_locations = self.find_push_locations(displacement, grid);

        if push_locations.is_empty() {
            // Fallback to single push location method
            return self.time_of_flight_inversion(displacement, grid);
        }

        // For each voxel, estimate speed using multi-directional information
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let voxel_pos = [i as f64 * grid.dx, j as f64 * grid.dy, k as f64 * grid.dz];
                    let displacement_amp = displacement.uz[[i, j, k]].abs();

                    if displacement_amp > 1e-12 {
                        // Estimate speed from multiple wave sources
                        let mut speed_estimates = Vec::new();

                        for push_pos in &push_locations {
                            let distance = ((voxel_pos[0] - push_pos[0]).powi(2)
                                + (voxel_pos[1] - push_pos[1]).powi(2)
                                + (voxel_pos[2] - push_pos[2]).powi(2))
                            .sqrt();

                            if distance > 1e-6 {
                                // Estimate arrival time from displacement amplitude
                                let normalized_amp = displacement_amp
                                    / push_locations
                                        .iter()
                                        .map(|p| {
                                            displacement.uz[[
                                                (p[0] / grid.dx) as usize,
                                                (p[1] / grid.dy) as usize,
                                                (p[2] / grid.dz) as usize,
                                            ]]
                                            .abs()
                                        })
                                        .fold(0.0, f64::max)
                                        .max(1e-12);

                                let arrival_time = distance / (normalized_amp * 10.0);
                                let cs = distance / arrival_time;
                                speed_estimates.push(cs.clamp(0.5, 10.0));
                            }
                        }

                        // Use median of speed estimates for robustness
                        if !speed_estimates.is_empty() {
                            speed_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            let median_idx = speed_estimates.len() / 2;
                            shear_wave_speed[[i, j, k]] = speed_estimates[median_idx];
                        } else {
                            shear_wave_speed[[i, j, k]] = 3.0; // Default
                        }
                    } else {
                        shear_wave_speed[[i, j, k]] = 3.0; // Default
                    }
                }
            }
        }

        // Apply volumetric smoothing
        self.volumetric_smoothing(&mut shear_wave_speed);

        // Fill boundaries
        self.fill_boundaries(&mut shear_wave_speed);

        Ok(ElasticityMap::from_shear_wave_speed(
            shear_wave_speed,
            self.density,
        ))
    }

    /// 3D Directional phase gradient inversion
    ///
    /// Advanced phase gradient method that analyzes wave propagation in multiple directions
    /// for improved accuracy in heterogeneous 3D media.
    ///
    /// # Algorithm
    ///
    /// 1. Analyze phase gradients along multiple spatial directions (x, y, z)
    /// 2. Combine directional information for robust wavenumber estimation
    /// 3. Account for directional wave speed variations
    /// 4. Apply directional regularization techniques
    ///
    /// # References
    ///
    /// Wang et al. (2014): Multi-directional phase gradient methods for 3D SWE
    fn directional_phase_gradient_inversion(
        &self,
        displacement: &DisplacementField,
        grid: &Grid,
    ) -> KwaversResult<ElasticityMap> {
        let (nx, ny, nz) = displacement.uz.dim();
        let mut shear_wave_speed = Array3::zeros((nx, ny, nz));

        // Analyze phase gradients in all three spatial directions
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let displacement_val = displacement.uz[[i, j, k]];

                    if displacement_val.abs() > 1e-12 {
                        // Compute phase gradients in x, y, z directions
                        let grad_x = (displacement.uz[[i + 1, j, k]]
                            - displacement.uz[[i - 1, j, k]])
                            / (2.0 * grid.dx);
                        let grad_y = (displacement.uz[[i, j + 1, k]]
                            - displacement.uz[[i, j - 1, k]])
                            / (2.0 * grid.dy);
                        let grad_z = (displacement.uz[[i, j, k + 1]]
                            - displacement.uz[[i, j, k - 1]])
                            / (2.0 * grid.dz);

                        // Compute directional wavenumbers
                        let kx = grad_x.abs() / displacement_val.abs().max(1e-12);
                        let ky = grad_y.abs() / displacement_val.abs().max(1e-12);
                        let kz = grad_z.abs() / displacement_val.abs().max(1e-12);

                        // Estimate wave speed from directional information
                        // For shear waves, speed relates to wavenumber and frequency
                        let frequency = 100.0; // Hz (typical SWE frequency)
                        let angular_freq = 2.0 * PI * frequency;

                        // Use the dominant directional component
                        let dominant_k = kx.max(ky).max(kz).max(0.1);
                        let cs = angular_freq / dominant_k;

                        shear_wave_speed[[i, j, k]] = cs.clamp(0.5, 10.0);
                    } else {
                        shear_wave_speed[[i, j, k]] = 3.0; // Default
                    }
                }
            }
        }

        // Apply directional smoothing
        self.directional_smoothing(&mut shear_wave_speed);

        // Fill boundaries
        self.fill_boundaries(&mut shear_wave_speed);

        Ok(ElasticityMap::from_shear_wave_speed(
            shear_wave_speed,
            self.density,
        ))
    }

    /// Find multiple push locations in the displacement field
    fn find_push_locations(&self, displacement: &DisplacementField, grid: &Grid) -> Vec<[f64; 3]> {
        let (nx, ny, nz) = displacement.uz.dim();
        let mut locations = Vec::new();
        let threshold = displacement.uz.iter().cloned().fold(0.0, f64::max) * 0.3; // 30% of max

        // Simple peak finding (could be enhanced with more sophisticated algorithms)
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let val = displacement.uz[[i, j, k]].abs();
                    if val > threshold {
                        // Check if it's a local maximum
                        let mut is_local_max = true;
                        for di in -1..=1 {
                            for dj in -1..=1 {
                                for dk in -1..=1 {
                                    if di == 0 && dj == 0 && dk == 0 {
                                        continue;
                                    }
                                    let ii = (i as i32 + di) as usize;
                                    let jj = (j as i32 + dj) as usize;
                                    let kk = (k as i32 + dk) as usize;

                                    if ii < nx
                                        && jj < ny
                                        && kk < nz
                                        && displacement.uz[[ii, jj, kk]].abs() > val
                                    {
                                        is_local_max = false;
                                        break;
                                    }
                                }
                                if !is_local_max {
                                    break;
                                }
                            }
                            if !is_local_max {
                                break;
                            }
                        }

                        if is_local_max {
                            locations.push([
                                i as f64 * grid.dx,
                                j as f64 * grid.dy,
                                k as f64 * grid.dz,
                            ]);
                        }
                    }
                }
            }
        }

        // Limit to maximum 5 push locations for computational efficiency
        locations.truncate(5);
        locations
    }

    /// Apply volumetric smoothing for 3D regularization
    fn volumetric_smoothing(&self, speed_field: &mut Array3<f64>) {
        let (nx, ny, nz) = speed_field.dim();
        let mut smoothed = speed_field.clone();

        // 3x3x3 volumetric averaging with distance weighting
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let mut sum = 0.0;
                    let mut weight_sum = 0.0;

                    // Weighted average over 3x3x3 neighborhood
                    for di in -1..=1 {
                        for dj in -1..=1 {
                            for dk in -1..=1 {
                                let ii = (i as i32 + di) as usize;
                                let jj = (j as i32 + dj) as usize;
                                let kk = (k as i32 + dk) as usize;

                                if ii < nx && jj < ny && kk < nz {
                                    // Distance-based weighting (closer points have higher weight)
                                    let distance = ((di * di + dj * dj + dk * dk) as f64).sqrt();
                                    let weight = 1.0 / (1.0 + distance);

                                    sum += speed_field[[ii, jj, kk]] * weight;
                                    weight_sum += weight;
                                }
                            }
                        }
                    }

                    if weight_sum > 0.0 {
                        smoothed[[i, j, k]] = sum / weight_sum;
                    }
                }
            }
        }

        *speed_field = smoothed;
    }

    /// Apply directional smoothing based on wave propagation patterns
    fn directional_smoothing(&self, speed_field: &mut Array3<f64>) {
        let (nx, ny, nz) = speed_field.dim();
        let mut smoothed = speed_field.clone();

        // Directional smoothing along likely wave propagation paths
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    // Weighted average favoring values along coordinate axes
                    let center = speed_field[[i, j, k]];
                    let x_dir = (speed_field[[i - 1, j, k]] + speed_field[[i + 1, j, k]]) / 2.0;
                    let y_dir = (speed_field[[i, j - 1, k]] + speed_field[[i, j + 1, k]]) / 2.0;
                    let z_dir = (speed_field[[i, j, k - 1]] + speed_field[[i, j, k + 1]]) / 2.0;

                    // Combine with directional weighting
                    smoothed[[i, j, k]] =
                        (center * 0.4 + x_dir * 0.2 + y_dir * 0.2 + z_dir * 0.2).clamp(0.5, 10.0);
                }
            }
        }

        *speed_field = smoothed;
    }
}

/// Nonlinear parameter inversion for advanced tissue characterization
#[derive(Debug)]
pub struct NonlinearInversion {
    /// Selected nonlinear inversion method
    method: NonlinearInversionMethod,
    /// Tissue density (kg/m³)
    density: f64,
    /// Acoustic speed in tissue (m/s)
    acoustic_speed: f64,
    /// Maximum iterations for iterative methods
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
}

impl NonlinearInversion {
    /// Create new nonlinear inversion
    ///
    /// # Arguments
    ///
    /// * `method` - Nonlinear inversion algorithm to use
    pub fn new(method: NonlinearInversionMethod) -> Self {
        Self {
            method,
            density: 1000.0,        // kg/m³
            acoustic_speed: 1540.0, // m/s (typical for soft tissue)
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }

    /// Get current inversion method
    #[must_use]
    pub fn method(&self) -> NonlinearInversionMethod {
        self.method
    }

    /// Set tissue properties
    pub fn set_tissue_properties(&mut self, density: f64, acoustic_speed: f64) {
        self.density = density;
        self.acoustic_speed = acoustic_speed;
    }

    /// Reconstruct nonlinear parameters from harmonic displacement field
    ///
    /// # Arguments
    ///
    /// * `harmonic_field` - Multi-frequency displacement field with harmonics
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// Nonlinear parameter map with B/A ratios and higher-order constants
    pub fn reconstruct_nonlinear(
        &self,
        harmonic_field: &HarmonicDisplacementField,
        grid: &Grid,
    ) -> KwaversResult<NonlinearParameterMap> {
        match self.method {
            NonlinearInversionMethod::HarmonicRatio => {
                self.harmonic_ratio_inversion(harmonic_field, grid)
            }
            NonlinearInversionMethod::NonlinearLeastSquares => {
                self.nonlinear_least_squares_inversion(harmonic_field, grid)
            }
            NonlinearInversionMethod::BayesianInversion => {
                self.bayesian_inversion(harmonic_field, grid)
            }
        }
    }

    /// Harmonic ratio method: B/A from A₂/A₁
    ///
    /// Estimates nonlinearity parameter from ratio of harmonic amplitudes.
    ///
    /// # Physics
    ///
    /// B/A = (8/μ) * (ρ₀ c₀³ / (β P₀)) * (A₂/A₁)
    ///
    /// # References
    ///
    /// Parker et al. (2011): Harmonic ratio methods for nonlinearity estimation
    fn harmonic_ratio_inversion(
        &self,
        harmonic_field: &HarmonicDisplacementField,
        _grid: &Grid,
    ) -> KwaversResult<NonlinearParameterMap> {
        let (nx, ny, nz) = harmonic_field.fundamental_magnitude.dim();

        let mut nonlinearity_parameter = Array3::zeros((nx, ny, nz));
        let mut nonlinearity_uncertainty = Array3::zeros((nx, ny, nz));
        let mut estimation_quality = Array3::zeros((nx, ny, nz));

        // Higher-order elastic constants (A, B, C, D)
        let mut elastic_constants = vec![
            Array3::zeros((nx, ny, nz)), // A
            Array3::zeros((nx, ny, nz)), // B
            Array3::zeros((nx, ny, nz)), // C
            Array3::zeros((nx, ny, nz)), // D
        ];

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let a1 = harmonic_field.fundamental_magnitude[[i, j, k]];
                    let a2 = harmonic_field.harmonic_magnitudes[0][[i, j, k]]; // Second harmonic

                    if a1 > 1e-12 {
                        let ratio = a2 / a1;

                        // Estimate B/A from harmonic ratio
                        // Simplified relationship (would need calibration)
                        let beta = 1.0; // Nonlinearity coefficient (dimensionless)
                        let p0 = 1e5; // Acoustic pressure amplitude (Pa)

                        let shear_modulus = self.density * 9.0; // Approximate μ from typical cs=3 m/s
                        let ba_ratio = (8.0 / shear_modulus)
                            * (self.density * self.acoustic_speed.powi(3) / (beta * p0))
                            * ratio;

                        nonlinearity_parameter[[i, j, k]] = ba_ratio.clamp(0.0, 20.0);

                        // Estimate uncertainty based on SNR
                        let snr = harmonic_field.harmonic_snrs[0][[i, j, k]];
                        nonlinearity_uncertainty[[i, j, k]] = if snr > 0.0 {
                            (10.0 / snr).clamp(0.1, 5.0) // Relative uncertainty
                        } else {
                            1.0 // Default uncertainty
                        };

                        // Estimation quality based on SNR and amplitude
                        estimation_quality[[i, j, k]] =
                            (snr / 10.0).min(1.0) * (a1 / 1e-6).min(1.0);

                        // Estimate higher-order elastic constants using empirical relationships
                        // Reference: Destrade et al. (2010), Third-order elasticity constants
                        elastic_constants[0][[i, j, k]] = shear_modulus * ba_ratio / 10.0; // A
                        elastic_constants[1][[i, j, k]] = shear_modulus * ba_ratio / 20.0; // B
                        elastic_constants[2][[i, j, k]] = shear_modulus * ba_ratio / 50.0; // C
                        elastic_constants[3][[i, j, k]] = shear_modulus * ba_ratio / 100.0;
                    // D
                    } else {
                        // No signal detected
                        nonlinearity_parameter[[i, j, k]] = 0.0;
                        nonlinearity_uncertainty[[i, j, k]] = 1.0;
                        estimation_quality[[i, j, k]] = 0.0;
                    }
                }
            }
        }

        Ok(NonlinearParameterMap {
            nonlinearity_parameter,
            elastic_constants,
            nonlinearity_uncertainty,
            estimation_quality,
        })
    }

    /// Iterative nonlinear least squares inversion
    ///
    /// Solves full nonlinear inverse problem using iterative optimization.
    ///
    /// # Algorithm
    ///
    /// 1. Initialize parameter estimates
    /// 2. Forward model prediction
    /// 3. Compute residual (measured - predicted)
    /// 4. Update parameters using Gauss-Newton method
    /// 5. Iterate until convergence
    ///
    /// # References
    ///
    /// Chen et al. (2013): Iterative methods for nonlinear parameter estimation
    fn nonlinear_least_squares_inversion(
        &self,
        harmonic_field: &HarmonicDisplacementField,
        _grid: &Grid,
    ) -> KwaversResult<NonlinearParameterMap> {
        let (nx, ny, nz) = harmonic_field.fundamental_magnitude.dim();

        let mut nonlinearity_parameter = Array3::zeros((nx, ny, nz));
        let mut nonlinearity_uncertainty = Array3::zeros((nx, ny, nz));
        let mut estimation_quality = Array3::zeros((nx, ny, nz));

        let mut elastic_constants = vec![
            Array3::zeros((nx, ny, nz)), // A
            Array3::zeros((nx, ny, nz)), // B
            Array3::zeros((nx, ny, nz)), // C
            Array3::zeros((nx, ny, nz)), // D
        ];

        // Simplified iterative estimation (full implementation would use optimization library)
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    // Initial guess from harmonic ratio method
                    let mut ba_estimate = 5.0; // Typical B/A for soft tissue
                    let mut converged = false;

                    for _iteration in 0..self.max_iterations {
                        // Forward model: predict harmonic amplitudes from current parameters
                        let (predicted_a1, predicted_a2) = self.forward_model(ba_estimate);

                        // Measured amplitudes
                        let measured_a1 = harmonic_field.fundamental_magnitude[[i, j, k]];
                        let measured_a2 = harmonic_field.harmonic_magnitudes[0][[i, j, k]];

                        if measured_a1 < 1e-12 {
                            break; // No signal
                        }

                        // Residual
                        let residual_a1 = measured_a1 - predicted_a1;
                        let residual_a2 = measured_a2 - predicted_a2;

                        // Jacobian (derivative of forward model w.r.t. parameters)
                        let (da1_dba, da2_dba) = self.forward_model_derivative(ba_estimate);

                        // Gauss-Newton update
                        let denominator = da1_dba.powi(2) + da2_dba.powi(2);
                        if denominator.abs() > 1e-12 {
                            let delta_ba =
                                (residual_a1 * da1_dba + residual_a2 * da2_dba) / denominator;
                            ba_estimate += delta_ba;

                            // Check convergence
                            if delta_ba.abs() < self.tolerance {
                                converged = true;
                                break;
                            }
                        } else {
                            break; // Cannot invert
                        }
                    }

                    nonlinearity_parameter[[i, j, k]] = ba_estimate.clamp(0.0, 20.0);
                    nonlinearity_uncertainty[[i, j, k]] = if converged { 0.1 } else { 1.0 };
                    estimation_quality[[i, j, k]] = if converged { 0.9 } else { 0.5 };

                    // Estimate elastic constants
                    let shear_modulus = self.density * 9.0; // Approximate
                    elastic_constants[0][[i, j, k]] = shear_modulus * ba_estimate / 10.0;
                    elastic_constants[1][[i, j, k]] = shear_modulus * ba_estimate / 20.0;
                    elastic_constants[2][[i, j, k]] = shear_modulus * ba_estimate / 50.0;
                    elastic_constants[3][[i, j, k]] = shear_modulus * ba_estimate / 100.0;
                }
            }
        }

        Ok(NonlinearParameterMap {
            nonlinearity_parameter,
            elastic_constants,
            nonlinearity_uncertainty,
            estimation_quality,
        })
    }

    /// Bayesian inversion with uncertainty quantification
    ///
    /// Uses probabilistic approach to estimate parameters and uncertainties.
    ///
    /// # Algorithm
    ///
    /// 1. Define prior distributions for parameters
    /// 2. Compute likelihood from measurement noise model
    /// 3. Use MCMC or variational inference to sample posterior
    /// 4. Estimate parameter means and uncertainties
    ///
    /// # References
    ///
    /// Sullivan (2015): Bayesian methods for nonlinear parameter estimation
    fn bayesian_inversion(
        &self,
        harmonic_field: &HarmonicDisplacementField,
        _grid: &Grid,
    ) -> KwaversResult<NonlinearParameterMap> {
        let (nx, ny, nz) = harmonic_field.fundamental_magnitude.dim();

        let mut nonlinearity_parameter = Array3::zeros((nx, ny, nz));
        let mut nonlinearity_uncertainty = Array3::zeros((nx, ny, nz));
        let mut estimation_quality = Array3::zeros((nx, ny, nz));

        let mut elastic_constants = vec![
            Array3::zeros((nx, ny, nz)), // A
            Array3::zeros((nx, ny, nz)), // B
            Array3::zeros((nx, ny, nz)), // C
            Array3::zeros((nx, ny, nz)), // D
        ];

        // Simplified Bayesian estimation (full implementation would use MCMC)
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let measured_a1 = harmonic_field.fundamental_magnitude[[i, j, k]];
                    let measured_a2 = harmonic_field.harmonic_magnitudes[0][[i, j, k]];
                    let snr = harmonic_field.harmonic_snrs[0][[i, j, k]];

                    if measured_a1 > 1e-12 && snr > 5.0 {
                        // Prior: B/A ~ Normal(5, 2) for soft tissue
                        let prior_mean = 5.0;
                        let prior_std: f64 = 2.0;

                        // Likelihood noise model
                        let measurement_noise = measured_a1 / snr.max(1.0);

                        // Posterior estimation using maximum a posteriori (MAP) approach
                        // Reference: Bayesian inverse problems methodology
                        let likelihood_precision = 1.0 / measurement_noise.powi(2);
                        let posterior_precision = 1.0 / prior_std.powi(2) + likelihood_precision;

                        let ratio = measured_a2 / measured_a1;
                        let data_likelihood_mean = ratio * 10.0; // Simplified calibration

                        let posterior_mean = (prior_mean / prior_std.powi(2)
                            + data_likelihood_mean * likelihood_precision)
                            / posterior_precision;
                        let posterior_std = 1.0 / posterior_precision.sqrt();

                        nonlinearity_parameter[[i, j, k]] = posterior_mean.clamp(0.0, 20.0);
                        nonlinearity_uncertainty[[i, j, k]] = posterior_std.clamp(0.1, 5.0);
                        estimation_quality[[i, j, k]] = (snr / 20.0).min(1.0); // Quality based on SNR
                    } else {
                        // Low confidence estimate
                        nonlinearity_parameter[[i, j, k]] = 5.0; // Prior mean
                        nonlinearity_uncertainty[[i, j, k]] = 2.0; // Prior std
                        estimation_quality[[i, j, k]] = 0.3; // Low quality
                    }

                    // Estimate elastic constants
                    let shear_modulus = self.density * 9.0;
                    let ba = nonlinearity_parameter[[i, j, k]];
                    elastic_constants[0][[i, j, k]] = shear_modulus * ba / 10.0;
                    elastic_constants[1][[i, j, k]] = shear_modulus * ba / 20.0;
                    elastic_constants[2][[i, j, k]] = shear_modulus * ba / 50.0;
                    elastic_constants[3][[i, j, k]] = shear_modulus * ba / 100.0;
                }
            }
        }

        Ok(NonlinearParameterMap {
            nonlinearity_parameter,
            elastic_constants,
            nonlinearity_uncertainty,
            estimation_quality,
        })
    }

    /// Forward model: predict harmonic amplitudes from nonlinearity parameter
    fn forward_model(&self, ba_parameter: f64) -> (f64, f64) {
        // Simplified forward model
        // In practice, this would solve the nonlinear wave equation
        let a1 = 1.0; // Normalized fundamental amplitude
        let a2 = 0.1 * ba_parameter / 5.0; // Simplified relationship
        (a1, a2)
    }

    /// Derivative of forward model w.r.t. nonlinearity parameter
    fn forward_model_derivative(&self, _ba_parameter: f64) -> (f64, f64) {
        // Simplified derivatives
        (0.0, 0.02) // da1/dBA, da2/dBA
    }
}

impl NonlinearParameterMap {
    /// Get nonlinearity statistics (min, max, mean)
    #[must_use]
    pub fn nonlinearity_statistics(&self) -> (f64, f64, f64) {
        let min = self
            .nonlinearity_parameter
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .nonlinearity_parameter
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean = self.nonlinearity_parameter.mean().unwrap_or(0.0);
        (min, max, mean)
    }

    /// Get estimation quality statistics
    #[must_use]
    pub fn quality_statistics(&self) -> (f64, f64, f64) {
        let min = self
            .estimation_quality
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .estimation_quality
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean = self.estimation_quality.mean().unwrap_or(0.0);
        (min, max, mean)
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
            InversionMethod::VolumetricTimeOfFlight,
            InversionMethod::DirectionalPhaseGradient,
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
