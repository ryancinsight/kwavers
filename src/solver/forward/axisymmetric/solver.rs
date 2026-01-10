//! Axisymmetric k-space pseudospectral solver
//!
//! Main solver implementation for axisymmetric wave propagation.

#![allow(deprecated)]

use super::config::AxisymmetricConfig;
use super::config::AxisymmetricMedium;
use super::coordinates::CylindricalGrid;
use super::transforms::DiscreteHankelTransform;
use crate::core::error::KwaversResult;
use crate::domain::medium::adapters::CylindricalMediumProjection;
use crate::domain::medium::Medium;
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Axisymmetric k-space pseudospectral solver
///
/// Implements `kspaceFirstOrderAS` equivalent functionality for efficient
/// simulation of axially symmetric acoustic wave propagation.
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::solver::forward::axisymmetric::{
///     AxisymmetricSolver, AxisymmetricConfig, AxisymmetricMedium
/// };
///
/// let config = AxisymmetricConfig::default();
/// let medium = AxisymmetricMedium::homogeneous(128, 64, 1500.0, 1000.0);
/// let mut solver = AxisymmetricSolver::new(config, medium)?;
///
/// // Set initial pressure source
/// solver.set_initial_pressure(&initial_pressure);
///
/// // Run simulation
/// let sensor_data = solver.run()?;
/// ```
#[derive(Debug)]
pub struct AxisymmetricSolver {
    /// Configuration
    config: AxisymmetricConfig,
    /// Medium properties
    medium: AxisymmetricMedium,
    /// Cylindrical grid
    grid: CylindricalGrid,
    /// Discrete Hankel transform for radial direction
    dht: DiscreteHankelTransform,
    /// Pressure field (nz x nr)
    pressure: Array2<f64>,
    /// Axial velocity field (nz x nr)
    velocity_z: Array2<f64>,
    /// Radial velocity field (nz x nr)
    velocity_r: Array2<f64>,
    /// Density at staggered grid points
    rho0_sgz: Array2<f64>,
    rho0_sgr: Array2<f64>,
    /// PML absorption profiles
    pml_z: Array1<f64>,
    pml_r: Array1<f64>,
    /// k-space correction operator
    kappa: Array2<f64>,
    /// Current time step
    current_step: usize,
    /// Sensor mask (if any)
    sensor_mask: Option<Array2<bool>>,
    /// Recorded sensor data
    sensor_data: Vec<Array2<f64>>,
}

impl AxisymmetricSolver {
    /// Create a new axisymmetric solver from cylindrical medium projection
    ///
    /// This is the recommended constructor for new code. It accepts a domain-level
    /// medium projected to cylindrical coordinates.
    ///
    /// # Arguments
    ///
    /// * `config` - Solver configuration
    /// * `projection` - Cylindrical projection of a 3D medium
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use kwavers::solver::forward::axisymmetric::{AxisymmetricSolver, AxisymmetricConfig};
    /// use kwavers::domain::medium::{HomogeneousMedium, adapters::CylindricalMediumProjection};
    /// use kwavers::domain::grid::{Grid, CylindricalTopology};
    ///
    /// let grid = Grid::new(128, 128, 128, 1e-4, 1e-4, 1e-4)?;
    /// let medium = HomogeneousMedium::water(&grid);
    /// let topology = CylindricalTopology::new(128, 64, 1e-4, 1e-4)?;
    /// let projection = CylindricalMediumProjection::new(&medium, &grid, &topology)?;
    ///
    /// let config = AxisymmetricConfig::default();
    /// let solver = AxisymmetricSolver::new_with_projection(config, &projection)?;
    /// # Ok::<(), kwavers::core::error::KwaversError>(())
    /// ```
    pub fn new_with_projection<M: Medium>(
        config: AxisymmetricConfig,
        projection: &CylindricalMediumProjection<M>,
    ) -> KwaversResult<Self> {
        config.validate().map_err(|e| {
            crate::core::error::KwaversError::Config(
                crate::core::error::ConfigError::InvalidValue {
                    parameter: "config".to_string(),
                    value: "invalid".to_string(),
                    constraint: e,
                },
            )
        })?;

        // Verify projection dimensions match config
        let (nz_proj, nr_proj) = projection.sound_speed_field().dim();
        if nz_proj != config.nz || nr_proj != config.nr {
            return Err(crate::core::error::KwaversError::Config(
                crate::core::error::ConfigError::InvalidValue {
                    parameter: "projection dimensions".to_string(),
                    value: format!("({}, {})", nz_proj, nr_proj),
                    constraint: format!(
                        "Must match config dimensions ({}, {})",
                        config.nz, config.nr
                    ),
                },
            ));
        }

        // Build AxisymmetricMedium from projection for internal use
        let medium = AxisymmetricMedium {
            sound_speed: projection.sound_speed_field().to_owned(),
            density: projection.density_field().to_owned(),
            alpha_coeff: projection.absorption_field().to_owned(),
            alpha_power: 2.0, // Default, could be extended in future
            b_over_a: projection.nonlinearity_field().map(|arr| arr.to_owned()),
        };

        // Delegate to existing constructor
        #[allow(deprecated)]
        Self::new(config, medium)
    }

    /// Create a new axisymmetric solver (legacy constructor)
    ///
    /// # Deprecation
    ///
    /// This constructor is deprecated. Use [`AxisymmetricSolver::new_with_projection`]
    /// instead, which accepts domain-level media via `CylindricalMediumProjection`.
    ///
    /// # Migration Guide
    ///
    /// **Old code:**
    /// ```rust,ignore
    /// use kwavers::solver::forward::axisymmetric::{AxisymmetricConfig, AxisymmetricMedium, AxisymmetricSolver};
    ///
    /// let config = AxisymmetricConfig::default();
    /// let medium = AxisymmetricMedium::homogeneous(128, 64, 1500.0, 1000.0);
    /// let solver = AxisymmetricSolver::new(config, medium)?;
    /// ```
    ///
    /// **New code:**
    /// ```rust,ignore
    /// use kwavers::solver::forward::axisymmetric::{AxisymmetricConfig, AxisymmetricSolver};
    /// use kwavers::domain::medium::{HomogeneousMedium, adapters::CylindricalMediumProjection};
    /// use kwavers::domain::grid::{Grid, CylindricalTopology};
    ///
    /// // Create 3D grid and medium
    /// let grid = Grid::new(128, 128, 128, 1e-4, 1e-4, 1e-4)?;
    /// let medium = HomogeneousMedium::new(&grid, 1500.0, 1000.0, 0.0);
    ///
    /// // Create cylindrical topology and projection
    /// let topology = CylindricalTopology::new(128, 64, 1e-4, 1e-4)?;
    /// let projection = CylindricalMediumProjection::new(&medium, &grid, &topology)?;
    ///
    /// // Use new constructor
    /// let config = AxisymmetricConfig::default();
    /// let solver = AxisymmetricSolver::new_with_projection(config, &projection)?;
    /// ```
    #[deprecated(
        since = "2.16.0",
        note = "Use `new_with_projection` with `CylindricalMediumProjection` instead. \
                See migration guide in documentation."
    )]
    pub fn new(config: AxisymmetricConfig, medium: AxisymmetricMedium) -> KwaversResult<Self> {
        config.validate().map_err(|e| {
            crate::core::error::KwaversError::Config(
                crate::core::error::ConfigError::InvalidValue {
                    parameter: "config".to_string(),
                    value: "invalid".to_string(),
                    constraint: e,
                },
            )
        })?;

        // Check CFL stability
        let c_max = medium.max_sound_speed();
        if !config.is_stable(c_max) {
            tracing::warn!(
                "CFL condition may not be satisfied. CFL = {:.3}, consider reducing dt",
                config.cfl_number(c_max)
            );
        }

        // Create grid
        let grid = CylindricalGrid::new(config.nz, config.nr, config.dz, config.dr)?;

        // Create Hankel transform
        let dht = DiscreteHankelTransform::new(config.nr, grid.r_max());

        // Initialize fields
        let pressure = Array2::zeros((config.nz, config.nr));
        let velocity_z = Array2::zeros((config.nz, config.nr));
        let velocity_r = Array2::zeros((config.nz, config.nr));

        // Staggered grid densities
        let rho0_sgz = Self::compute_staggered_density(&medium.density, 0);
        let rho0_sgr = Self::compute_staggered_density(&medium.density, 1);

        // PML profiles
        let pml_z = Self::compute_pml_profile(config.nz, config.pml_size, config.pml_alpha);
        let pml_r = Self::compute_pml_profile(config.nr, config.pml_size, config.pml_alpha);

        // k-space correction operator
        let kappa = Self::compute_kspace_correction(&grid, &medium, &config);

        Ok(Self {
            config,
            medium,
            grid,
            dht,
            pressure,
            velocity_z,
            velocity_r,
            rho0_sgz,
            rho0_sgr,
            pml_z,
            pml_r,
            kappa,
            current_step: 0,
            sensor_mask: None,
            sensor_data: Vec::new(),
        })
    }

    /// Compute density on staggered grid
    fn compute_staggered_density(density: &Array2<f64>, axis: usize) -> Array2<f64> {
        let (nz, nr) = density.dim();
        let mut staggered = Array2::zeros((nz, nr));

        match axis {
            0 => {
                // Stagger in z direction
                for i in 0..nz - 1 {
                    for j in 0..nr {
                        staggered[[i, j]] = 0.5 * (density[[i, j]] + density[[i + 1, j]]);
                    }
                }
                // Boundary
                for j in 0..nr {
                    staggered[[nz - 1, j]] = density[[nz - 1, j]];
                }
            }
            1 => {
                // Stagger in r direction
                for i in 0..nz {
                    for j in 0..nr - 1 {
                        staggered[[i, j]] = 0.5 * (density[[i, j]] + density[[i, j + 1]]);
                    }
                    staggered[[i, nr - 1]] = density[[i, nr - 1]];
                }
            }
            _ => panic!("Invalid axis"),
        }

        staggered
    }

    /// Compute PML absorption profile
    fn compute_pml_profile(n: usize, pml_size: usize, alpha: f64) -> Array1<f64> {
        let mut profile = Array1::ones(n);

        for i in 0..pml_size {
            let depth = (pml_size - i) as f64 / pml_size as f64;
            let absorption = (depth * alpha).exp();
            profile[i] *= absorption;
            profile[n - 1 - i] *= absorption;
        }

        profile
    }

    /// Compute k-space correction operator
    fn compute_kspace_correction(
        grid: &CylindricalGrid,
        _medium: &AxisymmetricMedium,
        config: &AxisymmetricConfig,
    ) -> Array2<f64> {
        let (nz, nr) = (config.nz, config.nr);
        let mut kappa = Array2::ones((nz, nr));

        if config.use_kspace_correction {
            let c_ref = config.c_ref;
            let dt = config.dt;

            for i in 0..nz {
                let kz = grid.kz[i];
                for j in 0..nr {
                    let kr = grid.kr[j];
                    let k = (kz * kz + kr * kr).sqrt();

                    if k > 0.0 {
                        // k-space correction: sinc(c_ref * k * dt / 2)
                        let arg = c_ref * k * dt / 2.0;
                        kappa[[i, j]] = if arg.abs() < 1e-10 {
                            1.0
                        } else {
                            arg.sin() / arg
                        };
                    }
                }
            }
        }

        kappa
    }

    /// Set initial pressure distribution
    pub fn set_initial_pressure(&mut self, p0: &Array2<f64>) {
        assert_eq!(
            p0.dim(),
            self.pressure.dim(),
            "Pressure field dimension mismatch"
        );
        self.pressure.assign(p0);
    }

    /// Set sensor mask for recording
    pub fn set_sensor_mask(&mut self, mask: Array2<bool>) {
        assert_eq!(
            mask.dim(),
            self.pressure.dim(),
            "Sensor mask dimension mismatch"
        );
        self.sensor_mask = Some(mask);
    }

    /// Run the simulation
    pub fn run(&mut self) -> KwaversResult<Vec<Array2<f64>>> {
        self.sensor_data.clear();
        self.current_step = 0;

        for step in 0..self.config.nt {
            self.time_step()?;
            self.current_step = step + 1;

            // Record sensor data
            if let Some(ref mask) = self.sensor_mask {
                let mut record = Array2::zeros(self.pressure.dim());
                for ((i, j), &m) in mask.indexed_iter() {
                    if m {
                        record[[i, j]] = self.pressure[[i, j]];
                    }
                }
                self.sensor_data.push(record);
            }

            // Progress reporting
            if step % 100 == 0 {
                tracing::debug!(
                    "Axisymmetric solver step {}/{}, max pressure: {:.3e}",
                    step,
                    self.config.nt,
                    self.pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()))
                );
            }
        }

        Ok(self.sensor_data.clone())
    }

    /// Execute one time step
    fn time_step(&mut self) -> KwaversResult<()> {
        let dt = self.config.dt;

        // 1. Update velocity from pressure gradient
        self.update_velocity(dt)?;

        // 2. Update pressure from velocity divergence
        self.update_pressure(dt)?;

        // 3. Apply PML absorption
        self.apply_pml();

        Ok(())
    }

    /// Update velocity field from pressure gradient
    fn update_velocity(&mut self, dt: f64) -> KwaversResult<()> {
        let (nz, nr) = (self.config.nz, self.config.nr);

        // Compute pressure gradient using spectral methods
        // dp/dz using FFT in z-direction
        let dp_dz = self.compute_z_derivative(&self.pressure);

        // dp/dr using Hankel transform
        let dp_dr = self.compute_r_derivative(&self.pressure);

        // Update velocities: dv/dt = -1/rho * grad(p)
        for i in 0..nz {
            for j in 0..nr {
                self.velocity_z[[i, j]] -= dt * dp_dz[[i, j]] / self.rho0_sgz[[i, j]];
                self.velocity_r[[i, j]] -= dt * dp_dr[[i, j]] / self.rho0_sgr[[i, j]];
            }
        }

        Ok(())
    }

    /// Update pressure field from velocity divergence
    fn update_pressure(&mut self, dt: f64) -> KwaversResult<()> {
        let (nz, nr) = (self.config.nz, self.config.nr);

        // Compute velocity divergence: div(v) = dv_z/dz + (1/r) d(r*v_r)/dr
        let dvz_dz = self.compute_z_derivative(&self.velocity_z);

        // For radial term: (1/r) d(r*v_r)/dr
        let mut rv_r = Array2::zeros((nz, nr));
        for i in 0..nz {
            for j in 0..nr {
                let r = self.grid.r_at(j);
                rv_r[[i, j]] = r * self.velocity_r[[i, j]];
            }
        }
        let d_rvr_dr = self.compute_r_derivative(&rv_r);

        // Update pressure: dp/dt = -rho * c^2 * div(v)
        for i in 0..nz {
            for j in 0..nr {
                let r = self.grid.r_at(j).max(self.config.dr * 0.5); // Avoid r=0 singularity
                let rho = self.medium.density[[i, j]];
                let c = self.medium.sound_speed[[i, j]];
                let div_v = dvz_dz[[i, j]] + d_rvr_dr[[i, j]] / r;

                self.pressure[[i, j]] -= dt * rho * c * c * div_v;

                // Apply k-space correction
                self.pressure[[i, j]] *= self.kappa[[i, j]];
            }
        }

        Ok(())
    }

    /// Compute z-derivative using FFT
    fn compute_z_derivative(&self, field: &Array2<f64>) -> Array2<f64> {
        let (nz, nr) = field.dim();
        let mut result = Array2::zeros((nz, nr));

        for j in 0..nr {
            // Extract column
            let col: Vec<Complex64> = (0..nz)
                .map(|i| Complex64::new(field[[i, j]], 0.0))
                .collect();

            // FFT
            let mut spectrum = Self::fft_1d(&col);

            // Multiply by i*kz
            for (i, val) in spectrum.iter_mut().enumerate().take(nz) {
                *val *= Complex64::new(0.0, self.grid.kz[i]);
            }

            // Inverse FFT
            let deriv = Self::ifft_1d(&spectrum);

            // Store result
            for (i, v) in deriv.iter().enumerate().take(nz) {
                result[[i, j]] = v.re;
            }
        }

        result
    }

    /// Compute r-derivative using Hankel transform
    fn compute_r_derivative(&self, field: &Array2<f64>) -> Array2<f64> {
        let (nz, nr) = field.dim();
        let mut result = Array2::zeros((nz, nr));

        for i in 0..nz {
            // Extract row
            let row = field.index_axis(Axis(0), i).to_owned();

            // Forward Hankel transform
            let f_k = self.dht.forward(&row);

            // Apply radial derivative in k-space: multiply by -k for gradient
            let mut df_k = Array1::zeros(nr);
            for j in 0..nr {
                df_k[j] = -self.dht.k()[j] * f_k[j];
            }

            // Inverse Hankel transform
            let deriv = self.dht.inverse(&df_k);

            // Store result
            for j in 0..nr {
                result[[i, j]] = deriv[j];
            }
        }

        result
    }

    /// Apply PML absorption
    fn apply_pml(&mut self) {
        let (nz, nr) = (self.config.nz, self.config.nr);

        for i in 0..nz {
            for j in 0..nr {
                let pml_factor = self.pml_z[i] * self.pml_r[j];
                self.pressure[[i, j]] *= pml_factor;
                self.velocity_z[[i, j]] *= pml_factor;
                self.velocity_r[[i, j]] *= pml_factor;
            }
        }
    }

    /// Simple 1D FFT (for demonstration - should use rustfft in production)
    fn fft_1d(input: &[Complex64]) -> Vec<Complex64> {
        let n = input.len();
        if n <= 1 {
            return input.to_vec();
        }

        // Cooley-Tukey FFT
        let mut output = vec![Complex64::new(0.0, 0.0); n];

        if n.is_power_of_two() {
            // Radix-2 FFT
            Self::fft_radix2(input, &mut output, n, 1, false);
        } else {
            // Fallback to DFT for non-power-of-2
            Self::dft(input, &mut output, false);
        }

        output
    }

    /// Simple 1D inverse FFT
    fn ifft_1d(input: &[Complex64]) -> Vec<Complex64> {
        let n = input.len();
        let mut output = vec![Complex64::new(0.0, 0.0); n];

        if n.is_power_of_two() {
            Self::fft_radix2(input, &mut output, n, 1, true);
        } else {
            Self::dft(input, &mut output, true);
        }

        // Normalize
        let scale = 1.0 / n as f64;
        for x in &mut output {
            *x *= scale;
        }

        output
    }

    fn fft_radix2(
        input: &[Complex64],
        output: &mut [Complex64],
        n: usize,
        stride: usize,
        inverse: bool,
    ) {
        if n == 1 {
            output[0] = input[0];
            return;
        }

        let half = n / 2;
        let sign = if inverse { 1.0 } else { -1.0 };

        // Recursively compute even and odd parts
        let even: Vec<Complex64> = (0..half).map(|i| input[i * 2 * stride]).collect();
        let odd: Vec<Complex64> = (0..half).map(|i| input[(i * 2 + 1) * stride]).collect();

        let mut even_fft = vec![Complex64::new(0.0, 0.0); half];
        let mut odd_fft = vec![Complex64::new(0.0, 0.0); half];

        Self::fft_radix2(&even, &mut even_fft, half, 1, inverse);
        Self::fft_radix2(&odd, &mut odd_fft, half, 1, inverse);

        // Combine
        for k in 0..half {
            let angle = sign * 2.0 * PI * k as f64 / n as f64;
            let twiddle = Complex64::new(angle.cos(), angle.sin());
            let t = twiddle * odd_fft[k];
            output[k] = even_fft[k] + t;
            output[k + half] = even_fft[k] - t;
        }
    }

    fn dft(input: &[Complex64], output: &mut [Complex64], inverse: bool) {
        let n = input.len();
        let sign = if inverse { 1.0 } else { -1.0 };

        for (k, out_k) in output.iter_mut().enumerate() {
            let mut acc = Complex64::new(0.0, 0.0);
            for (j, &in_j) in input.iter().enumerate() {
                let angle = sign * 2.0 * PI * (k * j) as f64 / n as f64;
                acc += in_j * Complex64::new(angle.cos(), angle.sin());
            }
            *out_k = acc;
        }
    }

    /// Get current pressure field
    pub fn pressure(&self) -> &Array2<f64> {
        &self.pressure
    }

    /// Get the grid
    pub fn grid(&self) -> &CylindricalGrid {
        &self.grid
    }

    /// Get configuration
    pub fn config(&self) -> &AxisymmetricConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::CylindricalTopology;

    #[test]
    fn test_solver_creation_legacy() {
        let config = AxisymmetricConfig::default();
        #[allow(deprecated)]
        let medium = AxisymmetricMedium::homogeneous(config.nz, config.nr, 1500.0, 1000.0);
        #[allow(deprecated)]
        let solver = AxisymmetricSolver::new(config, medium);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_solver_creation_with_projection() {
        use crate::domain::grid::Grid;
        use crate::domain::medium::{adapters::CylindricalMediumProjection, HomogeneousMedium};

        let config = AxisymmetricConfig::default();

        // Create 3D grid and medium
        let grid = Grid::new(128, 128, 128, 1e-4, 1e-4, 1e-4).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);

        // Create cylindrical topology
        let topology =
            CylindricalTopology::new(config.nz, config.nr, config.dz, config.dr).unwrap();

        // Create projection
        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        // Create solver
        let solver = AxisymmetricSolver::new_with_projection(config, &projection);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_pml_profile() {
        let profile = AxisymmetricSolver::compute_pml_profile(64, 10, 2.0);
        assert_eq!(profile.len(), 64);

        // PML should absorb at boundaries
        assert!(profile[0] > 1.0); // Absorption factor
        assert!((profile[32] - 1.0).abs() < 1e-10); // No absorption in center
    }

    #[test]
    fn test_initial_pressure() {
        let config = AxisymmetricConfig {
            nz: 32,
            nr: 16,
            pml_size: 4,
            ..Default::default()
        };
        #[allow(deprecated)]
        let medium = AxisymmetricMedium::homogeneous(32, 16, 1500.0, 1000.0);
        #[allow(deprecated)]
        let mut solver = AxisymmetricSolver::new(config, medium).unwrap();

        let p0 = Array2::from_elem((32, 16), 1.0);
        solver.set_initial_pressure(&p0);

        assert_eq!(solver.pressure()[[16, 8]], 1.0);
    }
}
