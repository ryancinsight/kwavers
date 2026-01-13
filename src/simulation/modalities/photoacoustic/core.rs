//! Core Photoacoustic Simulator Module
//!
//! This module provides the main [`PhotoacousticSimulator`] struct that orchestrates
//! the complete photoacoustic imaging simulation pipeline:
//!
//! 1. Optical fluence computation (diffusion approximation)
//! 2. Initial pressure generation (photoacoustic effect)
//! 3. Acoustic wave propagation (FDTD time-stepping)
//! 4. Image reconstruction (universal back-projection)
//!
//! ## Architecture
//!
//! The [`PhotoacousticSimulator`] follows the **Facade Pattern**, providing a unified
//! interface to the complex subsystems implemented in the optics, acoustics, and
//! reconstruction modules.
//!
//! ## Design Philosophy
//!
//! - **Separation of Concerns**: Each subsystem (optics, acoustics, reconstruction) is isolated
//! - **Single Responsibility**: Core orchestrates workflow; subsystems handle domain logic
//! - **Dependency Inversion**: Simulator depends on abstractions (Grid, Medium traits)
//! - **Testability**: Each component can be tested independently

use crate::domain::imaging::photoacoustic::{InitialPressure, PhotoacousticParameters, PhotoacousticResult};
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::properties::OpticalPropertyData;
use crate::domain::medium::Medium;
use crate::domain::source::GridSource;
use crate::solver::forward::fdtd::{FdtdConfig, FdtdSolver};
use crate::solver::inverse::reconstruction::photoacoustic::{
    PhotoacousticAlgorithm, PhotoacousticConfig, PhotoacousticReconstructor,
};
use crate::solver::reconstruction::Reconstructor;
use ndarray::{Array2, Array3};

use super::acoustics;
use super::optics;
use super::reconstruction;

/// Photoacoustic Imaging Simulator
///
/// Main orchestrator for photoacoustic imaging simulations. Integrates optical diffusion,
/// acoustic wave propagation, and image reconstruction into a unified pipeline.
///
/// # Architecture
///
/// - **Domain Layer**: Uses `Grid`, `Medium`, and domain types
/// - **Application Layer**: Orchestrates workflow through public methods
/// - **Infrastructure Layer**: Delegates to optics, acoustics, reconstruction modules
///
/// # Example
///
/// ```rust,no_run
/// use kwavers::simulation::modalities::photoacoustic::PhotoacousticSimulator;
/// use kwavers::domain::grid::Grid;
/// use kwavers::domain::medium::homogeneous::HomogeneousMedium;
/// use kwavers::clinical::imaging::photoacoustic::PhotoacousticParameters;
///
/// # fn main() -> kwavers::core::error::KwaversResult<()> {
/// let grid = Grid::new(64, 64, 32, 0.001, 0.001, 0.001)?;
/// let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
/// let parameters = PhotoacousticParameters::default();
///
/// let mut simulator = PhotoacousticSimulator::new(grid, parameters, &medium)?;
///
/// // Run complete simulation
/// let fluence = simulator.compute_fluence()?;
/// let initial_pressure = simulator.compute_initial_pressure(&fluence)?;
/// let result = simulator.simulate(&initial_pressure)?;
///
/// println!("SNR: {:.2} dB", result.snr);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PhotoacousticSimulator {
    /// Computational grid
    grid: Grid,
    /// Simulation parameters
    parameters: PhotoacousticParameters,
    /// Optical properties field
    optical_properties: Array3<OpticalPropertyData>,
    /// FDTD acoustic wave solver
    fdtd_solver: FdtdSolver,
}

impl PhotoacousticSimulator {
    /// Create new photoacoustic simulator
    ///
    /// # Arguments
    ///
    /// - `grid`: Computational grid defining spatial domain
    /// - `parameters`: Photoacoustic simulation parameters (wavelengths, laser fluence, etc.)
    /// - `medium`: Acoustic medium for wave propagation
    ///
    /// # Returns
    ///
    /// Configured simulator ready for simulation
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Grid dimensions are invalid
    /// - FDTD solver configuration fails
    /// - Optical property initialization fails
    pub fn new(
        grid: Grid,
        parameters: PhotoacousticParameters,
        medium: &dyn Medium,
    ) -> KwaversResult<Self> {
        let optical_properties = optics::initialize_optical_properties(&grid, medium)?;

        // Configure FDTD solver for photoacoustic wave propagation
        let dt = 1e-8;
        let nt = 100;

        let fdtd_config = FdtdConfig {
            spatial_order: 2,
            staggered_grid: true,
            cfl_factor: 0.3,
            subgridding: false,
            subgrid_factor: 2,
            enable_gpu_acceleration: false,
            nt,
            dt,
            sensor_mask: None,
        };

        let fdtd_solver = FdtdSolver::new(fdtd_config, &grid, medium, GridSource::default())?;

        Ok(Self {
            grid,
            parameters,
            optical_properties,
            fdtd_solver,
        })
    }

    /// Compute optical fluence distribution using diffusion approximation
    ///
    /// Solves the steady-state diffusion equation: ∇·(D∇Φ) - μₐΦ = -S
    /// using the finite-difference diffusion solver from physics layer.
    ///
    /// # Returns
    ///
    /// Optical fluence field Φ(r) in W/m² (or J/m² for pulsed illumination)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kwavers::simulation::modalities::photoacoustic::PhotoacousticSimulator;
    /// # use kwavers::domain::grid::Grid;
    /// # use kwavers::domain::medium::homogeneous::HomogeneousMedium;
    /// # use kwavers::clinical::imaging::photoacoustic::PhotoacousticParameters;
    /// # fn main() -> kwavers::core::error::KwaversResult<()> {
    /// # let grid = Grid::new(32, 32, 16, 0.001, 0.001, 0.001)?;
    /// # let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    /// # let parameters = PhotoacousticParameters::default();
    /// # let simulator = PhotoacousticSimulator::new(grid, parameters, &medium)?;
    /// let fluence = simulator.compute_fluence()?;
    /// println!("Fluence computed: {:?}", fluence.dim());
    /// # Ok(())
    /// # }
    /// ```
    pub fn compute_fluence(&self) -> KwaversResult<Array3<f64>> {
        self.compute_fluence_at_wavelength(
            self.parameters
                .wavelengths
                .first()
                .copied()
                .unwrap_or(750.0),
        )
    }

    /// Compute optical fluence for a specific wavelength
    ///
    /// # Arguments
    ///
    /// - `wavelength_nm`: Optical wavelength in nanometers
    ///
    /// # Returns
    ///
    /// Optical fluence field at the specified wavelength
    pub fn compute_fluence_at_wavelength(&self, wavelength_nm: f64) -> KwaversResult<Array3<f64>> {
        optics::compute_fluence_at_wavelength(
            &self.grid,
            &self.optical_properties,
            self.parameters.laser_fluence,
            wavelength_nm,
        )
    }

    /// Compute fluence for all wavelengths in parallel
    ///
    /// # Returns
    ///
    /// Vector of fluence fields, one per wavelength
    pub fn compute_multi_wavelength_fluence(&self) -> KwaversResult<Vec<Array3<f64>>> {
        optics::compute_multi_wavelength_fluence(
            &self.grid,
            &self.optical_properties,
            self.parameters.laser_fluence,
            &self.parameters.wavelengths,
        )
    }

    /// Compute initial pressure distribution from optical absorption at single wavelength
    ///
    /// Uses photoacoustic equation: p₀(r) = Γ · μₐ(r) · Φ(r)
    ///
    /// # Arguments
    ///
    /// - `fluence`: Optical fluence field Φ(r) in W/m² or J/m²
    ///
    /// # Returns
    ///
    /// Initial pressure distribution with metadata
    pub fn compute_initial_pressure(
        &self,
        fluence: &Array3<f64>,
    ) -> KwaversResult<InitialPressure> {
        acoustics::compute_initial_pressure(
            &self.grid,
            &self.optical_properties,
            fluence,
            &self.parameters.gruneisen_parameters,
            &self.parameters.wavelengths,
        )
    }

    /// Compute multi-wavelength initial pressure distributions
    ///
    /// # Arguments
    ///
    /// - `fluence_fields`: Vector of fluence fields, one per wavelength
    ///
    /// # Returns
    ///
    /// Vector of initial pressure distributions
    pub fn compute_multi_wavelength_pressure(
        &self,
        fluence_fields: &[Array3<f64>],
    ) -> KwaversResult<Vec<InitialPressure>> {
        acoustics::compute_multi_wavelength_pressure(
            &self.grid,
            &self.optical_properties,
            fluence_fields,
            &self.parameters.gruneisen_parameters,
            &self.parameters.wavelengths,
        )
    }

    /// Run multi-wavelength photoacoustic simulation (parallel)
    ///
    /// Computes fluence and initial pressure for all wavelengths in parallel.
    ///
    /// # Returns
    ///
    /// Vector of (fluence, initial_pressure) tuples for each wavelength
    pub fn simulate_multi_wavelength(&self) -> KwaversResult<Vec<(Array3<f64>, InitialPressure)>> {
        // Compute fluence for all wavelengths in parallel
        let fluence_fields = self.compute_multi_wavelength_fluence()?;

        // Compute initial pressure for each wavelength
        let results: Result<Vec<_>, _> = fluence_fields
            .iter()
            .map(|fluence| {
                self.compute_initial_pressure(fluence)
                    .map(|pressure| (fluence.clone(), pressure))
            })
            .collect();

        results
    }

    /// Run photoacoustic simulation with numerically stable acoustic propagation
    ///
    /// # Arguments
    ///
    /// - `initial_pressure`: Initial pressure distribution from optical absorption
    ///
    /// # Returns
    ///
    /// Complete simulation result including:
    /// - Time-resolved pressure fields
    /// - Time points
    /// - Reconstructed image
    /// - Signal-to-noise ratio
    ///
    /// # Algorithm
    ///
    /// 1. Propagate acoustic wave using FDTD time-stepping
    /// 2. Record pressure snapshots at detector positions
    /// 3. Reconstruct initial pressure using universal back-projection
    /// 4. Compute SNR from reconstructed image
    pub fn simulate(
        &mut self,
        initial_pressure: &InitialPressure,
    ) -> KwaversResult<PhotoacousticResult> {
        // Configure simulation parameters (CFL-respecting)
        let num_time_steps = 400; // sufficient temporal samples for reconstruction
        let snapshot_interval = 10; // store every 10th step

        // Propagate acoustic wave
        let (pressure_fields, time_points) = acoustics::propagate_acoustic_wave(
            &self.grid,
            initial_pressure,
            self.parameters.speed_of_sound,
            self.fdtd_solver.config.cfl_factor,
            num_time_steps,
            snapshot_interval,
        )?;

        // Reconstruct image from recorded fields using PhotoacousticReconstructor
        let reconstructed_image = self.reconstruct_with_solver(&pressure_fields, &time_points)?;

        // Signal-to-noise ratio (energy-based)
        let signal_power = reconstructed_image.iter().map(|&x| x * x).sum::<f64>()
            / reconstructed_image.len() as f64;
        let noise_power = 1e-12; // noise floor
        let snr = 10.0 * (signal_power / noise_power).log10();

        Ok(PhotoacousticResult {
            pressure_fields,
            time: time_points,
            reconstructed_image,
            snr,
        })
    }

    /// Reconstruct using the dedicated PhotoacousticReconstructor from the solver module
    ///
    /// This method provides integration with the solver layer's reconstruction framework.
    ///
    /// # Arguments
    ///
    /// - `pressure_fields`: Time-resolved pressure snapshots
    /// - `time_points`: Corresponding time values
    ///
    /// # Returns
    ///
    /// Reconstructed initial pressure distribution
    fn reconstruct_with_solver(
        &self,
        pressure_fields: &[Array3<f64>],
        time_points: &[f64],
    ) -> KwaversResult<Array3<f64>> {
        let n_time = time_points.len();
        let detectors = reconstruction::compute_detector_positions(&self.grid, 72);
        let mut sensor_data = Array2::zeros((n_time, detectors.len()));

        // Extract sensor data for the reconstructor
        for (d_idx, &(dx, dy, dz)) in detectors.iter().enumerate() {
            for (t_idx, field) in pressure_fields.iter().enumerate() {
                sensor_data[[t_idx, d_idx]] =
                    reconstruction::interpolate_detector_signal(&self.grid, field, dx, dy, dz);
            }
        }

        let detector_positions: Vec<[f64; 3]> = detectors
            .iter()
            .map(|&(x, y, z)| [x * self.grid.dx, y * self.grid.dy, z * self.grid.dz])
            .collect();

        let config = PhotoacousticConfig {
            algorithm: PhotoacousticAlgorithm::UniversalBackProjection,
            sensor_positions: detector_positions.clone(),
            grid_size: [self.grid.nx, self.grid.ny, self.grid.nz],
            sound_speed: self.parameters.speed_of_sound,
            sampling_frequency: 1.0 / (time_points[1] - time_points[0]),
            envelope_detection: false,
            bandpass_filter: None,
            regularization_parameter: 0.0,
        };

        let reconstructor = PhotoacousticReconstructor::new(config);
        let recon_config = crate::solver::reconstruction::ReconstructionConfig::default();

        reconstructor.reconstruct(&sensor_data, &detector_positions, &self.grid, &recon_config)
    }

    /// Time Reversal Reconstruction (Universal Back-Projection)
    ///
    /// Reconstructs the initial pressure distribution using a back-projection algorithm.
    /// This implementation simulates a Universal Back-Projection (UBP) approach by
    /// extracting signals at virtual detector positions and back-projecting them
    /// with spherical spreading correction (1/r weighting).
    ///
    /// # Arguments
    ///
    /// - `pressure_fields`: Time-resolved pressure fields
    /// - `time_points`: Corresponding time values
    ///
    /// # Returns
    ///
    /// Reconstructed initial pressure distribution
    pub fn time_reversal_reconstruction(
        &self,
        pressure_fields: &[Array3<f64>],
        time_points: &[f64],
    ) -> KwaversResult<Array3<f64>> {
        reconstruction::time_reversal_reconstruction(
            &self.grid,
            pressure_fields,
            time_points,
            self.parameters.speed_of_sound,
            72, // number of detectors
        )
    }

    /// Get grid reference
    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// Get optical properties reference
    pub fn optical_properties(&self) -> &Array3<OpticalPropertyData> {
        &self.optical_properties
    }

    /// Get parameters reference
    pub fn parameters(&self) -> &PhotoacousticParameters {
        &self.parameters
    }

    /// Validate against analytical solution
    ///
    /// Compares computed pressure with analytical photoacoustic generation formula
    /// at the center of the grid.
    ///
    /// # Returns
    ///
    /// Relative error between computed and analytical pressure
    pub fn validate_analytical(&self) -> KwaversResult<f64> {
        // Compare computed pressure with analytical expectation
        let fluence = self.compute_fluence()?;
        let initial_pressure = self.compute_initial_pressure(&fluence)?;

        // For validation, check a known location (center of grid)
        let center_i = self.grid.nx / 2;
        let center_j = self.grid.ny / 2;
        let center_k = self.grid.nz / 2;

        let computed_pressure = initial_pressure.pressure[[center_i, center_j, center_k]];
        let fluence_at_center = fluence[[center_i, center_j, center_k]];
        let props_at_center = &self.optical_properties[[center_i, center_j, center_k]];

        // Analytical pressure using photoacoustic generation formula
        // p = Γ μ_a Φ, where Γ is Grüneisen parameter, μ_a is absorption, Φ is fluence
        let analytical_pressure =
            props_at_center.anisotropy * props_at_center.absorption_coefficient * fluence_at_center;

        // Calculate relative error
        let error = if analytical_pressure > 0.0 {
            ((computed_pressure - analytical_pressure) / analytical_pressure).abs()
        } else {
            0.0 // No error if analytical pressure is zero
        };

        Ok(error)
    }
}
