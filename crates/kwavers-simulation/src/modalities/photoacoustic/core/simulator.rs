//! Core Photoacoustic Simulator — struct definition, construction, and optical methods.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_imaging::photoacoustic::PhotoacousticParameters;
use kwavers_medium::properties::OpticalPropertyData;
use kwavers_medium::Medium;
use kwavers_source::GridSource;
use kwavers_solver::forward::fdtd::{FdtdConfig, FdtdSolver, KSpaceCorrectionMode};
use ndarray::Array3;

use super::super::optics;

/// Photoacoustic Imaging Simulator
///
/// Main orchestrator for photoacoustic imaging simulations. Integrates optical diffusion,
/// acoustic wave propagation, and image reconstruction into a unified pipeline.
#[derive(Debug)]
pub struct PhotoacousticSimulator {
    /// Computational grid
    pub(super) grid: Grid,
    /// Simulation parameters
    pub(super) parameters: PhotoacousticParameters,
    /// Optical properties field
    pub(super) optical_properties: Array3<OpticalPropertyData>,
    /// FDTD acoustic wave solver
    pub(super) fdtd_solver: FdtdSolver,
}

impl PhotoacousticSimulator {
    /// Create new photoacoustic simulator
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
            enable_nonlinear: false,
            kspace_correction: KSpaceCorrectionMode::None,
            nt,
            dt,
            sensor_mask: None,
            geometry: Default::default(),
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compute_fluence_at_wavelength(&self, wavelength_nm: f64) -> KwaversResult<Array3<f64>> {
        optics::compute_fluence_at_wavelength(
            &self.grid,
            &self.optical_properties,
            self.parameters.laser_fluence,
            wavelength_nm,
        )
    }

    /// Compute fluence for all wavelengths in parallel
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compute_multi_wavelength_fluence(&self) -> KwaversResult<Vec<Array3<f64>>> {
        optics::compute_multi_wavelength_fluence(
            &self.grid,
            &self.optical_properties,
            self.parameters.laser_fluence,
            &self.parameters.wavelengths,
        )
    }

    /// Get grid reference
    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// Get optical properties reference
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn optical_properties(&self) -> &Array3<OpticalPropertyData> {
        &self.optical_properties
    }

    /// Get parameters reference
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn parameters(&self) -> &PhotoacousticParameters {
        &self.parameters
    }

    /// Validate against analytical solution
    ///
    /// Compares computed pressure with analytical photoacoustic generation formula
    /// at the center of the grid.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn validate_analytical(&self) -> KwaversResult<f64> {
        let fluence = self.compute_fluence()?;
        let initial_pressure = self.compute_initial_pressure(&fluence)?;

        let center_i = self.grid.nx / 2;
        let center_j = self.grid.ny / 2;
        let center_k = self.grid.nz / 2;

        let computed_pressure = initial_pressure.pressure[[center_i, center_j, center_k]];
        let fluence_at_center = fluence[[center_i, center_j, center_k]];
        let props_at_center = &self.optical_properties[[center_i, center_j, center_k]];

        // Analytical pressure: p = Γ μ_a Φ
        let analytical_pressure =
            props_at_center.anisotropy * props_at_center.absorption_coefficient * fluence_at_center;

        let error = if analytical_pressure > 0.0 {
            ((computed_pressure - analytical_pressure) / analytical_pressure).abs()
        } else {
            0.0
        };

        Ok(error)
    }
}
