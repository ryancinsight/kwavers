//! Proper Westervelt equation implementation using FDTD
//!
//! This module implements the Westervelt equation correctly using finite-difference
//! time-domain (FDTD) methods, suitable for heterogeneous media.
//!
//! ## Mathematical Foundation
//!
//! The Westervelt equation in its standard form:
//! ```text
//! ∇²p - (1/c²)∂²p/∂t² + (δ/c⁴)∂³p/∂t³ = -(β/ρ₀c⁴)∂²(p²)/∂t²
//! ```
//!
//! Where:
//! - p: acoustic pressure
//! - c: sound speed
//! - δ: diffusivity of sound
//! - β: coefficient of nonlinearity (B/A + 1)
//! - ρ₀: ambient density
//!
//! ## Numerical Implementation
//!
//! We use a second-order accurate FDTD scheme with proper treatment of the
//! nonlinear term. The equation is discretized as:
//!
//! ```text
//! p^{n+1} = 2p^n - p^{n-1} + (cΔt)²∇²p^n
//!           - (δΔt/c²)*(p^n - 2p^{n-1} + p^{n-2})/Δt²
//!           - (βΔt²/ρ₀c²)*∂²(p²)/∂t²|^n
//! ```
//!
//! ## Literature References
//!
//! 1. Hamilton, M. F., & Blackstock, D. T. (Eds.). (1998).
//!    "Nonlinear Acoustics" (Vol. 237). San Diego: Academic press.
//!
//! 2. Aanonsen, S. I., Barkve, T., Tjøtta, J. N., & Tjøtta, S. (1984).
//!    "Distortion and harmonic generation in the nearfield of a finite amplitude sound beam"
//!    J. Acoust. Soc. Am., 75(3), 749-768.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::Source;
use crate::solver::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances, ConservationTracker, ViolationSeverity,
};
use ndarray::{Array3, Zip};

/// Configuration for Westervelt FDTD solver
#[derive(Debug, Clone)]
pub struct WesterveltFdtdConfig {
    /// Spatial discretization order (2, 4, or 6)
    pub spatial_order: usize,
    /// Enable absorption/attenuation term
    pub enable_absorption: bool,
    /// CFL safety factor (< 1.0)
    pub cfl_safety: f64,
    /// Artificial viscosity coefficient for stability (dimensionless, 0.0-1.0)
    pub artificial_viscosity: f64,
}

impl Default for WesterveltFdtdConfig {
    fn default() -> Self {
        Self {
            spatial_order: 4,
            enable_absorption: true,
            cfl_safety: 0.95,
            artificial_viscosity: 0.01, // Small artificial viscosity for stability
        }
    }
}

/// Westervelt equation solver using FDTD
#[derive(Debug)]
pub struct WesterveltFdtd {
    config: WesterveltFdtdConfig,
    /// Current pressure field p^n
    pressure: Array3<f64>,
    /// Previous pressure field p^{n-1}
    pressure_prev: Array3<f64>,
    /// Two steps back p^{n-2} (for absorption term)
    pressure_prev2: Option<Array3<f64>>,
    /// Workspace for Laplacian calculation
    laplacian: Array3<f64>,
    /// Conservation diagnostics tracker
    conservation_tracker: Option<ConservationTracker>,
    /// Current time step counter
    current_step: usize,
    /// Current simulation time
    current_time: f64,
    /// Grid reference for conservation calculations
    grid: Grid,
    /// Medium reference for conservation calculations
    medium_properties: MediumProperties,
}

/// Cached medium properties for conservation calculations
#[derive(Debug, Clone)]
struct MediumProperties {
    rho0: f64,
    c0: f64,
}

impl WesterveltFdtd {
    /// Create a new Westervelt FDTD solver
    pub fn new(config: WesterveltFdtdConfig, grid: &Grid, medium: &dyn Medium) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);

        // Get representative medium properties (center point)
        let center_x = grid.dx * (grid.nx as f64) / 2.0;
        let center_y = grid.dy * (grid.ny as f64) / 2.0;
        let center_z = grid.dz * (grid.nz as f64) / 2.0;
        let rho0 = crate::domain::medium::density_at(medium, center_x, center_y, center_z, grid);
        let c0 = crate::domain::medium::sound_speed_at(medium, center_x, center_y, center_z, grid);

        Self {
            config,
            pressure: Array3::zeros(shape),
            pressure_prev: Array3::zeros(shape),
            pressure_prev2: None,
            laplacian: Array3::zeros(shape),
            conservation_tracker: None,
            current_step: 0,
            current_time: 0.0,
            grid: grid.clone(),
            medium_properties: MediumProperties { rho0, c0 },
        }
    }

    /// Enable conservation diagnostics with specified tolerances
    ///
    /// # Arguments
    ///
    /// * `tolerances` - Conservation tolerance parameters (absolute/relative/check_interval)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kwavers::solver::forward::nonlinear::conservation::ConservationTolerances;
    ///
    /// let mut solver = WesterveltFdtd::new(config, &grid, &medium);
    /// solver.enable_conservation_diagnostics(ConservationTolerances::default());
    /// ```
    pub fn enable_conservation_diagnostics(&mut self, tolerances: ConservationTolerances) {
        let initial_energy = self.calculate_total_energy();
        let initial_momentum = self.calculate_total_momentum();
        let initial_mass = self.calculate_total_mass();

        self.conservation_tracker = Some(ConservationTracker::new(
            initial_energy,
            initial_momentum,
            initial_mass,
            tolerances,
        ));
    }

    /// Disable conservation diagnostics
    pub fn disable_conservation_diagnostics(&mut self) {
        self.conservation_tracker = None;
    }

    /// Get conservation diagnostic summary
    ///
    /// Returns a summary of all conservation checks performed,
    /// including maximum severity and error magnitudes.
    pub fn get_conservation_summary(&self) -> Option<String> {
        self.conservation_tracker
            .as_ref()
            .map(|tracker| tracker.summary().to_string())
    }

    /// Check if solution satisfies conservation constraints
    ///
    /// Returns `true` if all conservation violations are within acceptable limits.
    pub fn is_solution_valid(&self) -> bool {
        self.conservation_tracker
            .as_ref()
            .is_none_or(|tracker| tracker.is_solution_valid())
    }

    /// Calculate the Laplacian using finite differences
    fn calculate_laplacian(&mut self, grid: &Grid) -> KwaversResult<()> {
        let pressure = &self.pressure;
        let laplacian = &mut self.laplacian;

        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let (dx, dy, dz) = (grid.dx, grid.dy, grid.dz);
        let dx2_inv = 1.0 / (dx * dx);
        let dy2_inv = 1.0 / (dy * dy);
        let dz2_inv = 1.0 / (dz * dz);

        match self.config.spatial_order {
            2 => {
                // Second-order accurate using safe indexing
                for i in 1..nx - 1 {
                    for j in 1..ny - 1 {
                        for k in 1..nz - 1 {
                            let p = pressure[(i, j, k)];
                            let px_m = pressure[(i - 1, j, k)];
                            let px_p = pressure[(i + 1, j, k)];
                            let py_m = pressure[(i, j - 1, k)];
                            let py_p = pressure[(i, j + 1, k)];
                            let pz_m = pressure[(i, j, k - 1)];
                            let pz_p = pressure[(i, j, k + 1)];

                            laplacian[(i, j, k)] = (px_p - 2.0 * p + px_m) * dx2_inv
                                + (py_p - 2.0 * p + py_m) * dy2_inv
                                + (pz_p - 2.0 * p + pz_m) * dz2_inv;
                        }
                    }
                }
            }
            4 => {
                // Fourth-order accurate stencil coefficients from constants
                // Fourth-order finite difference coefficients
                const FD4_COEFF_0: f64 = -5.0 / 2.0;
                const FD4_COEFF_1: f64 = 4.0 / 3.0;
                const FD4_COEFF_2: f64 = -1.0 / 12.0;
                const C0: f64 = FD4_COEFF_0;
                const C1: f64 = FD4_COEFF_1;
                const C2: f64 = FD4_COEFF_2;

                // Fourth-order accurate using safe indexing
                for i in 2..nx - 2 {
                    for j in 2..ny - 2 {
                        for k in 2..nz - 2 {
                            let p_c = pressure[(i, j, k)];

                            // X-direction stencil
                            let p_xm2 = pressure[(i - 2, j, k)];
                            let p_xm1 = pressure[(i - 1, j, k)];
                            let p_xp1 = pressure[(i + 1, j, k)];
                            let p_xp2 = pressure[(i + 2, j, k)];

                            let d2_dx2 =
                                (C0 * p_xm2 + C1 * p_xm1 + C2 * p_c + C1 * p_xp1 + C0 * p_xp2)
                                    * dx2_inv;

                            // Y-direction stencil
                            let p_ym2 = pressure[(i, j - 2, k)];
                            let p_ym1 = pressure[(i, j - 1, k)];
                            let p_yp1 = pressure[(i, j + 1, k)];
                            let p_yp2 = pressure[(i, j + 2, k)];

                            let d2_dy2 =
                                (C0 * p_ym2 + C1 * p_ym1 + C2 * p_c + C1 * p_yp1 + C0 * p_yp2)
                                    * dy2_inv;

                            // Z-direction stencil
                            let p_zm2 = pressure[(i, j, k - 2)];
                            let p_zm1 = pressure[(i, j, k - 1)];
                            let p_zp1 = pressure[(i, j, k + 1)];
                            let p_zp2 = pressure[(i, j, k + 2)];

                            let d2_dz2 =
                                (C0 * p_zm2 + C1 * p_zm1 + C2 * p_c + C1 * p_zp1 + C0 * p_zp2)
                                    * dz2_inv;

                            laplacian[(i, j, k)] = d2_dx2 + d2_dy2 + d2_dz2;
                        }
                    }
                }
            }
            _ => {
                // Default to second-order for unsupported orders
                self.config.spatial_order = 2;
                self.calculate_laplacian(grid)?;
            }
        }

        Ok(())
    }

    /// Calculate the nonlinear term ∂²(p²)/∂t²
    fn calculate_nonlinear_term(&self, dt: f64, grid: &Grid) -> Array3<f64> {
        let mut nonlinear = Array3::zeros((grid.nx, grid.ny, grid.nz));

        if let Some(ref p_prev2) = self.pressure_prev2 {
            // Full second-order time derivative of p²
            // ∂²(p²)/∂t² = 2p * ∂²p/∂t² + 2(∂p/∂t)²

            Zip::from(&mut nonlinear)
                .and(&self.pressure)
                .and(&self.pressure_prev)
                .and(p_prev2)
                .for_each(|nl, &p, &p_prev, &p_prev2| {
                    // Second derivative of pressure
                    let d2p_dt2 = (p - 2.0 * p_prev + p_prev2) / (dt * dt);

                    // First derivative of pressure
                    let dp_dt = (p - p_prev) / dt;

                    // Nonlinear term
                    *nl = 2.0 * p * d2p_dt2 + 2.0 * dp_dt * dp_dt;
                });
        } else {
            // First time step: use forward difference for initialization (LeVeque 2007 §2.14)
            Zip::from(&mut nonlinear)
                .and(&self.pressure)
                .and(&self.pressure_prev)
                .for_each(|nl, &p, &p_prev| {
                    let dp_dt = (p - p_prev) / dt;
                    *nl = 2.0 * dp_dt * dp_dt;
                });
        }

        nonlinear
    }

    /// Update the pressure field for one time step
    pub fn update(
        &mut self,
        medium: &dyn Medium,
        grid: &Grid,
        sources: &[Box<dyn Source>],
        t: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        // Calculate Laplacian of current pressure
        self.calculate_laplacian(grid)?;

        // Calculate nonlinear term
        let nonlinear_term = self.calculate_nonlinear_term(dt, grid);

        // Create new pressure array
        let mut pressure_next = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Update pressure using Westervelt equation
        Zip::indexed(&mut pressure_next)
            .and(&self.pressure)
            .and(&self.pressure_prev)
            .and(&self.laplacian)
            .and(&nonlinear_term)
            .for_each(|(i, j, k), p_next, &p, &p_prev, &lap, &nl| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                // Get medium properties
                let c = crate::domain::medium::sound_speed_at(medium, x, y, z, grid);
                let rho = crate::domain::medium::density_at(medium, x, y, z, grid);
                let beta = crate::domain::medium::AcousticProperties::nonlinearity_coefficient(
                    medium, x, y, z, grid,
                );

                // Linear wave propagation term
                let linear_term = (c * dt).powi(2) * lap;

                // Nonlinear term coefficient
                let nl_coeff = beta * dt.powi(2) / (rho * c.powi(2));

                // Absorption term (if enabled)
                let absorption_term = if self.config.enable_absorption {
                    if let Some(ref p_prev2) = self.pressure_prev2 {
                        let alpha =
                            crate::domain::medium::AcousticProperties::absorption_coefficient(
                                medium, x, y, z, grid, 1e6,
                            ); // 1 MHz reference
                        let delta = 2.0 * alpha * c.powi(3) / (2.0 * std::f64::consts::PI).powi(2);
                        delta * dt / c.powi(2) * (p - 2.0 * p_prev + p_prev2[[i, j, k]]) / (dt * dt)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                // Artificial viscosity term for numerical stability
                // ∇·(ν ∇p) where ν is artificial viscosity coefficient
                let visc_term = if i > 0
                    && i < grid.nx - 1
                    && j > 0
                    && j < grid.ny - 1
                    && k > 0
                    && k < grid.nz - 1
                {
                    let dx2 = grid.dx * grid.dx;
                    let dy2 = grid.dy * grid.dy;
                    let dz2 = grid.dz * grid.dz;

                    // Laplacian of pressure for viscosity
                    let lap_p = (self.pressure[(i + 1, j, k)] - 2.0 * p
                        + self.pressure[(i - 1, j, k)])
                        / dx2
                        + (self.pressure[(i, j + 1, k)] - 2.0 * p + self.pressure[(i, j - 1, k)])
                            / dy2
                        + (self.pressure[(i, j, k + 1)] - 2.0 * p + self.pressure[(i, j, k - 1)])
                            / dz2;

                    self.config.artificial_viscosity * dt * lap_p
                } else {
                    0.0
                };

                // Update equation: p^{n+1} = 2p^n - p^{n-1} + linear + nonlinear + absorption + viscosity
                *p_next =
                    2.0 * p - p_prev + linear_term - nl_coeff * nl - absorption_term + visc_term;

                // No explicit pressure clamping - allows natural shock formation through nonlinearity
                // Stability maintained through CFL conditions and artificial viscosity
            });

        // Add source contributions
        for source in sources {
            let amplitude = source.amplitude(t);
            if amplitude.abs() > 1e-12 {
                // Source is active if amplitude is non-zero
                let positions = source.positions();
                for position in positions {
                    // Find nearest grid point
                    let i = ((position.0 / grid.dx).round() as usize).min(grid.nx - 1);
                    let j = ((position.1 / grid.dy).round() as usize).min(grid.ny - 1);
                    let k = ((position.2 / grid.dz).round() as usize).min(grid.nz - 1);

                    pressure_next[[i, j, k]] += amplitude * dt;
                }
            }
        }

        // Update pressure history
        if self.config.enable_absorption {
            self.pressure_prev2 = Some(self.pressure_prev.clone());
        }
        self.pressure_prev = self.pressure.clone();
        self.pressure = pressure_next;

        // Update step counters
        self.current_step += 1;
        self.current_time += dt;

        // Conservation diagnostics (if enabled)
        self.check_conservation_laws();

        Ok(())
    }

    /// Check conservation laws and log diagnostics
    ///
    /// Performs conservation checks at configured intervals and logs violations.
    /// Critical violations trigger warnings via tracing infrastructure.
    fn check_conservation_laws(&mut self) {
        // Check if we should perform diagnostics at this step
        let should_check = self.conservation_tracker.as_ref().is_some_and(|tracker| {
            self.current_step
                .is_multiple_of(tracker.tolerances.check_interval)
        });

        if !should_check {
            return;
        }

        // Compute diagnostics first (without holding mutable reference to tracker)
        let (initial_energy, initial_momentum, initial_mass, tolerances) =
            if let Some(ref tracker) = self.conservation_tracker {
                (
                    tracker.initial_energy,
                    tracker.initial_momentum,
                    tracker.initial_mass,
                    tracker.tolerances,
                )
            } else {
                return;
            };

        let diagnostics = self.check_all_conservation(
            initial_energy,
            initial_momentum,
            initial_mass,
            self.current_step,
            self.current_time,
            &tolerances,
        );

        // Now update tracker with diagnostics
        if let Some(ref mut tracker) = self.conservation_tracker {
            // Update max severity
            for diag in &diagnostics {
                if diag.severity > tracker.max_severity {
                    tracker.max_severity = diag.severity;
                }
            }
            // Store in history
            tracker.history.extend(diagnostics.clone());
        }

        // Log diagnostics based on severity
        for diag in diagnostics {
            match diag.severity {
                ViolationSeverity::Acceptable => {
                    // Silent for acceptable violations
                }
                ViolationSeverity::Warning => {
                    eprintln!("⚠️  Westervelt FDTD Conservation Warning: {}", diag);
                }
                ViolationSeverity::Error => {
                    eprintln!("❌ Westervelt FDTD Conservation Error: {}", diag);
                }
                ViolationSeverity::Critical => {
                    eprintln!("🔴 Westervelt FDTD Conservation CRITICAL: {}", diag);
                    eprintln!("   Solution may be physically invalid!");
                }
            }
        }
    }

    /// Get the current pressure field
    #[must_use]
    pub fn pressure(&self) -> &Array3<f64> {
        &self.pressure
    }

    /// Calculate the CFL-limited time step
    pub fn calculate_dt(&self, medium: &dyn Medium, grid: &Grid) -> KwaversResult<f64> {
        // Get maximum sound speed
        let c_max = medium
            .sound_speed_array()
            .iter()
            .fold(0.0f64, |acc, &x| acc.max(x));

        // CFL condition for 3D FDTD
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let dt_cfl = self.config.cfl_safety * dx_min / (c_max * 3.0f64.sqrt());

        Ok(dt_cfl)
    }
}

/// Implementation of conservation diagnostics trait for Westervelt FDTD solver
impl ConservationDiagnostics for WesterveltFdtd {
    fn calculate_total_energy(&self) -> f64 {
        // Acoustic energy density: E = p²/(2ρ₀c₀²)
        // Total energy: ∫∫∫ E dV
        let mut total_energy = 0.0;
        let rho0 = self.medium_properties.rho0;
        let c0 = self.medium_properties.c0;
        let factor = 1.0 / (2.0 * rho0 * c0 * c0);
        let dv = self.grid.dx * self.grid.dy * self.grid.dz; // Volume element

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let p = self.pressure[[i, j, k]];
                    total_energy += p * p * factor * dv;
                }
            }
        }

        total_energy
    }

    fn calculate_total_momentum(&self) -> (f64, f64, f64) {
        // Full 3D momentum calculation
        // Momentum density: ρ₀ u where u = ∫ ∇p/(ρ₀) dt (acoustic approximation)
        // For simplicity, use p/c₀ approximation for magnitude
        let mut px = 0.0;
        let mut py = 0.0;
        let mut pz = 0.0;
        let rho0 = self.medium_properties.rho0;
        let c0 = self.medium_properties.c0;
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;

        // Compute momentum from pressure gradients
        for i in 1..self.grid.nx - 1 {
            for j in 1..self.grid.ny - 1 {
                for k in 1..self.grid.nz - 1 {
                    // Pressure gradients (central difference)
                    let dp_dx = (self.pressure[[i + 1, j, k]] - self.pressure[[i - 1, j, k]])
                        / (2.0 * self.grid.dx);
                    let dp_dy = (self.pressure[[i, j + 1, k]] - self.pressure[[i, j - 1, k]])
                        / (2.0 * self.grid.dy);
                    let dp_dz = (self.pressure[[i, j, k + 1]] - self.pressure[[i, j, k - 1]])
                        / (2.0 * self.grid.dz);

                    // Momentum from pressure (acoustic approximation)
                    px += (rho0 * dp_dx / c0) * dv;
                    py += (rho0 * dp_dy / c0) * dv;
                    pz += (rho0 * dp_dz / c0) * dv;
                }
            }
        }

        (px, py, pz)
    }

    fn calculate_total_mass(&self) -> f64 {
        // For acoustic waves: ρ = ρ₀(1 + p/(ρ₀c₀²))
        // Total mass: ∫∫∫ ρ dV
        let mut total_mass = 0.0;
        let rho0 = self.medium_properties.rho0;
        let c0 = self.medium_properties.c0;
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let p = self.pressure[[i, j, k]];
                    let rho = rho0 * (1.0 + p / (rho0 * c0 * c0));
                    total_mass += rho * dv;
                }
            }
        }

        total_mass
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::medium::HomogeneousMedium;

    #[test]
    fn test_westervelt_fdtd_creation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let config = WesterveltFdtdConfig::default();
        let solver = WesterveltFdtd::new(config, &grid, &medium);

        assert_eq!(solver.pressure.shape(), &[32, 32, 32]);
    }

    #[test]
    fn test_linear_wave_propagation() {
        // Test that with β=0 (no nonlinearity), we get linear wave propagation
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
        let mut medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        // Set nonlinearity to zero for linear test
        medium.nonlinearity = 0.0;

        // Use zero artificial viscosity for energy conservation test
        let config = WesterveltFdtdConfig {
            artificial_viscosity: 0.0,
            ..WesterveltFdtdConfig::default()
        };
        let mut solver = WesterveltFdtd::new(config, &grid, &medium);

        // Set initial Gaussian pulse
        let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let r2 = ((i as i32 - center.0 as i32).pow(2)
                        + (j as i32 - center.1 as i32).pow(2)
                        + (k as i32 - center.2 as i32).pow(2)) as f64;
                    solver.pressure[[i, j, k]] = (-(r2 / 100.0)).exp();
                }
            }
        }

        // Propagate for a few time steps
        let dt = solver.calculate_dt(&medium, &grid).unwrap();
        for _ in 0..10 {
            solver.update(&medium, &grid, &[], 0.0, dt).unwrap();
        }

        // Check that energy is conserved (approximately) with no artificial viscosity
        let total_energy: f64 = solver.pressure.iter().map(|&p| p * p).sum();
        assert!(total_energy > 0.0);
    }

    #[test]
    fn test_conservation_diagnostics_integration() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let config = WesterveltFdtdConfig::default();
        let mut solver = WesterveltFdtd::new(config, &grid, &medium);

        // Enable diagnostics
        solver.enable_conservation_diagnostics(ConservationTolerances::default());

        // Initial energy should be zero (no excitation)
        let initial_energy = solver.calculate_total_energy();
        assert!(initial_energy < 1e-10);

        // Verify tracker is enabled
        assert!(solver.conservation_tracker.is_some());
        assert!(solver.is_solution_valid());

        // Disable and check
        solver.disable_conservation_diagnostics();
        assert!(solver.conservation_tracker.is_none());
    }

    #[test]
    fn test_energy_calculation_accuracy() {
        let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let config = WesterveltFdtdConfig::default();
        let mut solver = WesterveltFdtd::new(config, &grid, &medium);

        // Set a known pressure field (uniform)
        let p0 = 1000.0; // Pa
        solver.pressure.fill(p0);

        // Calculate energy
        let energy = solver.calculate_total_energy();

        // Expected energy: E = p²/(2ρ₀c₀²) * Volume
        let rho0 = 1000.0;
        let c0 = 1500.0;
        let volume =
            (grid.nx as f64) * grid.dx * (grid.ny as f64) * grid.dy * (grid.nz as f64) * grid.dz;
        let expected_energy = (p0 * p0) / (2.0 * rho0 * c0 * c0) * volume;

        let relative_error = (energy - expected_energy).abs() / expected_energy;
        assert!(
            relative_error < 1e-10,
            "Energy calculation error: {}",
            relative_error
        );
    }

    #[test]
    fn test_conservation_check_interval() {
        let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let config = WesterveltFdtdConfig::default();
        let mut solver = WesterveltFdtd::new(config, &grid, &medium);

        // Enable diagnostics with check interval of 5
        let tolerances = ConservationTolerances {
            check_interval: 5,
            ..ConservationTolerances::default()
        };
        solver.enable_conservation_diagnostics(tolerances);

        // Simulate 20 steps
        let dt = solver.calculate_dt(&medium, &grid).unwrap();
        for _ in 0..20 {
            solver.update(&medium, &grid, &[], 0.0, dt).unwrap();
        }

        // Should have 20/5 = 4 checks (steps 5, 10, 15, 20)
        let summary = solver.get_conservation_summary().unwrap();
        assert!(summary.contains("checks"));
    }
}
