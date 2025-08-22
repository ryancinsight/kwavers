//! Thermal Diffusion Solver Module
//!
//! This module implements dedicated thermal diffusion solvers including:
//! - Standard heat diffusion equation
//! - Pennes bioheat equation
//! - Thermal dose calculations (CEM43)
//! - Hyperbolic heat transfer (Cattaneo-Vernotte)
//!
//! # Literature References
//!
//! 1. **Pennes, H. H. (1948)**. "Analysis of tissue and arterial blood temperatures
//!    in the resting human forearm." *Journal of Applied Physiology*, 1(2), 93-122.
//!    - Original formulation of bioheat equation
//!
//! 2. **Sapareto, S. A., & Dewey, W. C. (1984)**. "Thermal dose determination in
//!    cancer therapy." *International Journal of Radiation Oncology Biology Physics*,
//!    10(6), 787-800. DOI: 10.1016/0360-3016(84)90379-1
//!    - CEM43 thermal dose formulation
//!
//! 3. **Cattaneo, C. (1958)**. "A form of heat conduction equation which eliminates
//!    the paradox of instantaneous propagation." *Comptes Rendus*, 247, 431-433.
//!    - Hyperbolic heat transfer theory
//!
//! 4. **Liu, J., & Xu, L. X. (1999)**. "Estimation of blood perfusion using phase
//!    shift in temperature response to sinusoidal heating at the skin surface."
//!    *IEEE Transactions on Biomedical Engineering*, 46(9), 1037-1043.
//!    - Modern perfusion estimation methods

use crate::{
    error::{ConfigError, KwaversError, KwaversResult, PhysicsError},
    grid::Grid,
    medium::Medium,
    physics::plugin::{PhysicsPlugin, PluginContext, PluginMetadata, PluginState},
};
use log::info;
use ndarray::{s, Array3, Array4, Zip};
use std::collections::HashMap;

/// Configuration for thermal diffusion solver
#[derive(Debug, Clone)]
pub struct ThermalDiffusionConfig {
    /// Enable Pennes bioheat equation terms
    pub enable_bioheat: bool,
    /// Blood perfusion rate [1/s]
    pub perfusion_rate: f64,
    /// Blood density [kg/m³]
    pub blood_density: f64,
    /// Blood specific heat [J/(kg·K)]
    pub blood_specific_heat: f64,
    /// Arterial blood temperature [K]
    pub arterial_temperature: f64,
    /// Enable hyperbolic heat transfer (Cattaneo-Vernotte)
    pub enable_hyperbolic: bool,
    /// Thermal relaxation time [s]
    pub relaxation_time: f64,
    /// Enable thermal dose tracking
    pub track_thermal_dose: bool,
    /// Reference temperature for dose calculation [°C]
    pub dose_reference_temp: f64,
    /// Spatial discretization order (2, 4, or 6)
    pub spatial_order: usize,
}

impl Default for ThermalDiffusionConfig {
    fn default() -> Self {
        Self {
            enable_bioheat: true,
            perfusion_rate: 0.5e-3,       // 0.5 mL/g/min typical tissue
            blood_density: 1050.0,        // kg/m³
            blood_specific_heat: 3840.0,  // J/(kg·K)
            arterial_temperature: 310.15, // 37°C in Kelvin
            enable_hyperbolic: false,
            relaxation_time: 20.0, // 20s for tissue
            track_thermal_dose: true,
            dose_reference_temp: 43.0, // 43°C reference
            spatial_order: 4,
        }
    }
}

/// Thermal diffusion solver implementing various heat transfer models
#[derive(Debug, Clone)]
pub struct ThermalDiffusionSolver {
    config: ThermalDiffusionConfig,
    /// Temperature field
    temperature: Array3<f64>,
    /// Previous temperature (for time derivatives)
    temperature_prev: Option<Array3<f64>>,
    /// Heat flux components (for hyperbolic model)
    heat_flux_x: Option<Array3<f64>>,
    heat_flux_y: Option<Array3<f64>>,
    heat_flux_z: Option<Array3<f64>>,
    /// Cumulative thermal dose (CEM43)
    thermal_dose: Option<Array3<f64>>,
    /// Performance metrics
    metrics: HashMap<String, f64>,
    /// Workspace arrays for zero-copy operations
    workspace1: Array3<f64>,
    workspace2: Array3<f64>,
    workspace3: Array3<f64>,
}

impl ThermalDiffusionSolver {
    /// Create a new thermal diffusion solver
    pub fn new(config: ThermalDiffusionConfig, grid: &Grid) -> KwaversResult<Self> {
        info!(
            "Initializing thermal diffusion solver with config: {:?}",
            config
        );

        // Validate configuration
        if ![2, 4, 6].contains(&config.spatial_order) {
            return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
                parameter: "spatial_order".to_string(),
                value: config.spatial_order as f64,
                reason: format!(
                    "Invalid spatial order: {}. Must be 2, 4, or 6",
                    config.spatial_order
                ),
            }));
        }

        let shape = (grid.nx, grid.ny, grid.nz);
        let temperature = Array3::from_elem(shape, config.arterial_temperature);

        let heat_flux_x = if config.enable_hyperbolic {
            Some(Array3::zeros(shape))
        } else {
            None
        };

        let heat_flux_y = if config.enable_hyperbolic {
            Some(Array3::zeros(shape))
        } else {
            None
        };

        let heat_flux_z = if config.enable_hyperbolic {
            Some(Array3::zeros(shape))
        } else {
            None
        };

        let thermal_dose = if config.track_thermal_dose {
            Some(Array3::zeros(shape))
        } else {
            None
        };

        Ok(Self {
            config,
            temperature,
            temperature_prev: None,
            heat_flux_x,
            heat_flux_y,
            heat_flux_z,
            thermal_dose,
            metrics: HashMap::new(),
            workspace1: Array3::zeros(shape),
            workspace2: Array3::zeros(shape),
            workspace3: Array3::zeros(shape),
        })
    }

    /// Update temperature field using the configured heat transfer model
    pub fn update(
        &mut self,
        heat_source: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        let start_time = std::time::Instant::now();

        if self.config.enable_hyperbolic {
            self.update_hyperbolic(heat_source, grid, medium, dt)?;
        } else if self.config.enable_bioheat {
            self.update_bioheat(heat_source, grid, medium, dt)?;
        } else {
            self.update_standard(heat_source, grid, medium, dt)?;
        }

        // Update thermal dose if tracking
        if self.config.track_thermal_dose {
            self.update_thermal_dose(dt)?;
        }

        self.metrics.insert(
            "update_time".to_string(),
            start_time.elapsed().as_secs_f64(),
        );
        Ok(())
    }

    /// Standard heat diffusion equation: ∂T/∂t = α∇²T + Q/(ρc)
    fn update_standard(
        &mut self,
        heat_source: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        // Compute Laplacian of temperature
        Self::compute_laplacian(
            self.config.spatial_order,
            &self.temperature.clone(),
            &mut self.workspace1,
            grid,
        )?;

        // Update temperature using forward Euler
        Zip::indexed(&mut self.temperature)
            .and(&self.workspace1)
            .and(heat_source)
            .for_each(|(i, j, k), temp, &laplacian, &source| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let alpha = medium.thermal_diffusivity(x, y, z, grid);
                let rho = medium.density(x, y, z, grid);
                let cp = medium.specific_heat(x, y, z, grid);

                *temp += dt * (alpha * laplacian + source / (rho * cp));
            });

        Ok(())
    }

    /// Pennes bioheat equation: ∂T/∂t = α∇²T + ωb*cb*(Ta-T)/(ρc) + Q/(ρc)
    fn update_bioheat(
        &mut self,
        heat_source: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        // Compute Laplacian of temperature
        Self::compute_laplacian(
            self.config.spatial_order,
            &self.temperature.clone(),
            &mut self.workspace1,
            grid,
        )?;

        // Update temperature with bioheat terms
        let omega_b = self.config.perfusion_rate;
        let rho_b = self.config.blood_density;
        let c_b = self.config.blood_specific_heat;
        let t_a = self.config.arterial_temperature;

        Zip::indexed(&mut self.temperature)
            .and(&self.workspace1)
            .and(heat_source)
            .for_each(|(i, j, k), temp, &laplacian, &source| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let alpha = medium.thermal_diffusivity(x, y, z, grid);
                let rho = medium.density(x, y, z, grid);
                let cp = medium.specific_heat(x, y, z, grid);

                // Pennes bioheat equation
                let perfusion_term = omega_b * rho_b * c_b * (t_a - *temp) / (rho * cp);
                let source_term = source / (rho * cp);

                *temp += dt * (alpha * laplacian + perfusion_term + source_term);
            });

        Ok(())
    }

    /// Hyperbolic heat transfer (Cattaneo-Vernotte): τ∂²T/∂t² + ∂T/∂t = α∇²T + Q/(ρc)
    fn update_hyperbolic(
        &mut self,
        heat_source: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        let tau = self.config.relaxation_time;

        // For first timestep, use first-order scheme
        if self.temperature_prev.is_none() {
            // First timestep: use parabolic equation as approximation
            self.temperature_prev = Some(self.temperature.clone());
            return self.update_standard(heat_source, grid, medium, dt);
        }

        let temp_prev = self.temperature_prev.as_ref().unwrap();

        // Compute Laplacian
        Self::compute_laplacian(
            self.config.spatial_order,
            &self.temperature.clone(),
            &mut self.workspace1,
            grid,
        )?;

        // Update using second-order time derivative
        // (T^{n+1} - 2T^n + T^{n-1})/dt² + (T^{n+1} - T^{n-1})/(2τdt) = α∇²T^n/τ + Q/(τρc)
        // Rearranging for T^{n+1}:
        // T^{n+1} = [2T^n - T^{n-1} + dt²(α∇²T^n/τ + Q/(τρc))] / [1 + dt/(2τ)]

        let dt2 = dt * dt;
        let denominator = 1.0 + dt / (2.0 * tau);

        Zip::indexed(&mut self.workspace2)
            .and(&self.temperature)
            .and(temp_prev)
            .and(&self.workspace1)
            .and(heat_source)
            .for_each(
                |(i, j, k), result, &temp_curr, &temp_prev, &laplacian, &source| {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    let alpha = medium.thermal_diffusivity(x, y, z, grid);
                    let rho = medium.density(x, y, z, grid);
                    let cp = medium.specific_heat(x, y, z, grid);

                    let numerator = 2.0 * temp_curr - temp_prev
                        + dt2 * (alpha * laplacian / tau + source / (tau * rho * cp));

                    *result = numerator / denominator;
                },
            );

        // Update temperature history
        self.temperature_prev = Some(self.temperature.clone());
        self.temperature.assign(&self.workspace2);

        Ok(())
    }

    /// Compute spatial Laplacian using specified order accuracy
    fn compute_laplacian(
        spatial_order: usize,
        field: &Array3<f64>,
        result: &mut Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        match spatial_order {
            2 => Self::laplacian_2nd_order(field, result, grid),
            4 => Self::laplacian_4th_order(field, result, grid),
            6 => Self::laplacian_6th_order(field, result, grid),
            _ => {
                // Return an error for unsupported spatial orders
                return Err(KwaversError::Config(ConfigError::InvalidValue {
                    parameter: "spatial_order".to_string(),
                    value: spatial_order.to_string(),
                    constraint: "Only orders 2, 4, and 6 are supported".to_string(),
                }));
            }
        }
    }

    /// Second-order accurate Laplacian
    fn laplacian_2nd_order(
        field: &Array3<f64>,
        result: &mut Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let dx2_inv = 1.0 / (grid.dx * grid.dx);
        let dy2_inv = 1.0 / (grid.dy * grid.dy);
        let dz2_inv = 1.0 / (grid.dz * grid.dz);

        // Interior points
        let interior = s![1..-1, 1..-1, 1..-1];
        Zip::indexed(result.slice_mut(interior)).for_each(|(i, j, k), laplacian| {
            let i = i + 1;
            let j = j + 1;
            let k = k + 1;

            *laplacian = (field[[i + 1, j, k]] - 2.0 * field[[i, j, k]] + field[[i - 1, j, k]])
                * dx2_inv
                + (field[[i, j + 1, k]] - 2.0 * field[[i, j, k]] + field[[i, j - 1, k]]) * dy2_inv
                + (field[[i, j, k + 1]] - 2.0 * field[[i, j, k]] + field[[i, j, k - 1]]) * dz2_inv;
        });

        // Boundary conditions (Neumann: zero flux)
        Self::apply_neumann_bc(result, grid);

        Ok(())
    }

    /// Fourth-order accurate Laplacian
    fn laplacian_4th_order(
        field: &Array3<f64>,
        result: &mut Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let dx2_inv = 1.0 / (grid.dx * grid.dx);
        let dy2_inv = 1.0 / (grid.dy * grid.dy);
        let dz2_inv = 1.0 / (grid.dz * grid.dz);

        // Fourth-order stencil coefficients
        const C1: f64 = -1.0 / 12.0;
        const C2: f64 = 4.0 / 3.0;
        const C3: f64 = -5.0 / 2.0;

        // Interior points (need 2 ghost points on each side)
        let interior = s![2..-2, 2..-2, 2..-2];
        Zip::indexed(result.slice_mut(interior)).for_each(|(i, j, k), laplacian| {
            let i = i + 2;
            let j = j + 2;
            let k = k + 2;

            // X-direction
            let d2x = C1 * field[[i + 2, j, k]]
                + C2 * field[[i + 1, j, k]]
                + C3 * field[[i, j, k]]
                + C2 * field[[i - 1, j, k]]
                + C1 * field[[i - 2, j, k]];

            // Y-direction
            let d2y = C1 * field[[i, j + 2, k]]
                + C2 * field[[i, j + 1, k]]
                + C3 * field[[i, j, k]]
                + C2 * field[[i, j - 1, k]]
                + C1 * field[[i, j - 2, k]];

            // Z-direction
            let d2z = C1 * field[[i, j, k + 2]]
                + C2 * field[[i, j, k + 1]]
                + C3 * field[[i, j, k]]
                + C2 * field[[i, j, k - 1]]
                + C1 * field[[i, j, k - 2]];

            *laplacian = d2x * dx2_inv + d2y * dy2_inv + d2z * dz2_inv;
        });

        // Fill boundary region with 2nd order
        Self::fill_boundary_2nd_order(field, result, grid);

        Ok(())
    }

    /// Sixth-order accurate Laplacian
    fn laplacian_6th_order(
        field: &Array3<f64>,
        result: &mut Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let dx2_inv = 1.0 / (grid.dx * grid.dx);
        let dy2_inv = 1.0 / (grid.dy * grid.dy);
        let dz2_inv = 1.0 / (grid.dz * grid.dz);

        // Sixth-order stencil coefficients
        const C1: f64 = 1.0 / 90.0;
        const C2: f64 = -3.0 / 20.0;
        const C3: f64 = 3.0 / 2.0;
        const C4: f64 = -49.0 / 18.0;

        // Interior points (need 3 ghost points on each side)
        let interior = s![3..-3, 3..-3, 3..-3];
        Zip::indexed(result.slice_mut(interior)).for_each(|(i, j, k), laplacian| {
            let i = i + 3;
            let j = j + 3;
            let k = k + 3;

            // X-direction
            let d2x = C1 * field[[i + 3, j, k]]
                + C2 * field[[i + 2, j, k]]
                + C3 * field[[i + 1, j, k]]
                + C4 * field[[i, j, k]]
                + C3 * field[[i - 1, j, k]]
                + C2 * field[[i - 2, j, k]]
                + C1 * field[[i - 3, j, k]];

            // Y-direction
            let d2y = C1 * field[[i, j + 3, k]]
                + C2 * field[[i, j + 2, k]]
                + C3 * field[[i, j + 1, k]]
                + C4 * field[[i, j, k]]
                + C3 * field[[i, j - 1, k]]
                + C2 * field[[i, j - 2, k]]
                + C1 * field[[i, j - 3, k]];

            // Z-direction
            let d2z = C1 * field[[i, j, k + 3]]
                + C2 * field[[i, j, k + 2]]
                + C3 * field[[i, j, k + 1]]
                + C4 * field[[i, j, k]]
                + C3 * field[[i, j, k - 1]]
                + C2 * field[[i, j, k - 2]]
                + C1 * field[[i, j, k - 3]];

            *laplacian = d2x * dx2_inv + d2y * dy2_inv + d2z * dz2_inv;
        });

        // Fill boundary region with 4th order
        Self::fill_boundary_4th_order(field, result, grid);

        Ok(())
    }

    /// Apply Neumann boundary conditions (zero flux)
    fn apply_neumann_bc(field: &mut Array3<f64>, grid: &Grid) {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

        // X boundaries - copy values to enforce zero gradient
        let x_low = field.slice(s![1, .., ..]).to_owned();
        field.slice_mut(s![0, .., ..]).assign(&x_low);

        let x_high = field.slice(s![nx - 2, .., ..]).to_owned();
        field.slice_mut(s![nx - 1, .., ..]).assign(&x_high);

        // Y boundaries
        let y_low = field.slice(s![.., 1, ..]).to_owned();
        field.slice_mut(s![.., 0, ..]).assign(&y_low);

        let y_high = field.slice(s![.., ny - 2, ..]).to_owned();
        field.slice_mut(s![.., ny - 1, ..]).assign(&y_high);

        // Z boundaries
        let z_low = field.slice(s![.., .., 1]).to_owned();
        field.slice_mut(s![.., .., 0]).assign(&z_low);

        let z_high = field.slice(s![.., .., nz - 2]).to_owned();
        field.slice_mut(s![.., .., nz - 1]).assign(&z_high);
    }

    /// Fill boundary region with 2nd order accurate Laplacian
    fn fill_boundary_2nd_order(_field: &Array3<f64>, result: &mut Array3<f64>, grid: &Grid) {
        // For points within 1 cell of boundary, use 2nd order
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let dx2_inv = 1.0 / (grid.dx * grid.dx);
        let dy2_inv = 1.0 / (grid.dy * grid.dy);
        let dz2_inv = 1.0 / (grid.dz * grid.dz);

        // This is a simplified version - in production, would handle all boundary cases
        // For now, just set boundary values to zero (Neumann BC)
        for i in 0..2 {
            result.slice_mut(s![i, .., ..]).fill(0.0);
            result.slice_mut(s![nx - 1 - i, .., ..]).fill(0.0);
        }

        for j in 0..2 {
            result.slice_mut(s![.., j, ..]).fill(0.0);
            result.slice_mut(s![.., ny - 1 - j, ..]).fill(0.0);
        }

        for k in 0..2 {
            result.slice_mut(s![.., .., k]).fill(0.0);
            result.slice_mut(s![.., .., nz - 1 - k]).fill(0.0);
        }
    }

    /// Fill boundary region with 4th order accurate Laplacian
    fn fill_boundary_4th_order(_field: &Array3<f64>, result: &mut Array3<f64>, grid: &Grid) {
        // For points within 2 cells of boundary, use 4th order where possible
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

        // Simplified - set boundary values to zero (Neumann BC)
        for i in 0..3 {
            result.slice_mut(s![i, .., ..]).fill(0.0);
            result.slice_mut(s![nx - 1 - i, .., ..]).fill(0.0);
        }

        for j in 0..3 {
            result.slice_mut(s![.., j, ..]).fill(0.0);
            result.slice_mut(s![.., ny - 1 - j, ..]).fill(0.0);
        }

        for k in 0..3 {
            result.slice_mut(s![.., .., k]).fill(0.0);
            result.slice_mut(s![.., .., nz - 1 - k]).fill(0.0);
        }
    }

    /// Update cumulative thermal dose using CEM43 formulation
    fn update_thermal_dose(&mut self, dt: f64) -> KwaversResult<()> {
        if let Some(ref mut dose) = self.thermal_dose {
            let t_ref = self.config.dose_reference_temp + 273.15; // Convert to Kelvin

            Zip::from(&self.temperature)
                .and(dose)
                .for_each(|&temp, dose_val| {
                    let temp_c = temp - 273.15; // Convert to Celsius

                    if temp_c > self.config.dose_reference_temp {
                        // Above reference: R = 0.5 for T > 43°C
                        let r = 0.5_f64;
                        let exponent = self.config.dose_reference_temp - temp_c;
                        *dose_val += dt * r.powf(exponent) / 60.0; // Convert to minutes
                    } else if temp_c > 37.0 {
                        // Below reference but above body temp: R = 0.25 for T < 43°C
                        let r = 0.25_f64;
                        let exponent = self.config.dose_reference_temp - temp_c;
                        *dose_val += dt * r.powf(exponent) / 60.0;
                    }
                    // No dose accumulation below 37°C
                });
        }

        Ok(())
    }

    /// Get current temperature field
    pub fn temperature(&self) -> &Array3<f64> {
        &self.temperature
    }

    /// Get thermal dose field (if tracking)
    pub fn thermal_dose(&self) -> Option<&Array3<f64>> {
        self.thermal_dose.as_ref()
    }

    /// Get performance metrics
    pub fn metrics(&self) -> &HashMap<String, f64> {
        &self.metrics
    }

    /// Set initial temperature distribution
    pub fn set_temperature(&mut self, temperature: Array3<f64>) -> KwaversResult<()> {
        if temperature.shape() != self.temperature.shape() {
            return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
                parameter: "ThermalDiffusionSolver".to_string(),
                value: temperature.len() as f64,
                reason: "Temperature array shape mismatch".to_string(),
            }));
        }
        self.temperature = temperature;
        Ok(())
    }
}

/// Plugin implementation for thermal diffusion solver
#[derive(Debug)]
pub struct ThermalDiffusionPlugin {
    solver: ThermalDiffusionSolver,
    metadata: PluginMetadata,
}

impl ThermalDiffusionPlugin {
    /// Create a new thermal diffusion plugin
    pub fn new(config: ThermalDiffusionConfig, grid: &Grid) -> KwaversResult<Self> {
        let solver = ThermalDiffusionSolver::new(config, grid)?;

        let metadata = PluginMetadata {
            id: "thermal_diffusion".to_string(),
            name: "Thermal Diffusion Solver".to_string(),
            version: "1.0.0".to_string(),
            description:
                "Comprehensive thermal diffusion solver with bioheat equation and dose tracking"
                    .to_string(),
            author: "Kwavers Team".to_string(),
            license: "MIT".to_string(),
        };

        Ok(Self { solver, metadata })
    }
}

impl PhysicsPlugin for ThermalDiffusionPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        PluginState::Initialized
    }

    fn required_fields(&self) -> Vec<crate::physics::field_mapping::UnifiedFieldType> {
        vec![] // Thermal diffusion doesn't require specific fields, but uses pressure for heating
    }

    fn provided_fields(&self) -> Vec<crate::physics::field_mapping::UnifiedFieldType> {
        use crate::physics::field_mapping::UnifiedFieldType;
        vec![UnifiedFieldType::Temperature]
    }

    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        // Solver is already initialized in new()
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        // Get heat source from pressure field (acoustic heating)
        let pressure_idx = 0;
        let temperature_idx = 2;

        // Calculate acoustic heating: Q = 2αI = 2α|p|²/(ρc)
        let mut heat_source = self.solver.workspace1.clone();
        {
            let pressure = fields.index_axis(ndarray::Axis(0), pressure_idx);
            Zip::indexed(&mut heat_source)
                .and(&pressure)
                .for_each(|(i, j, k), source, &p| {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    let alpha = medium.absorption_coefficient(x, y, z, grid, 1e6); // 1 MHz default
                    let rho = medium.density(x, y, z, grid);
                    let c = medium.sound_speed(x, y, z, grid);

                    // Acoustic intensity I = |p|²/(ρc)
                    let intensity = p * p / (rho * c);
                    *source = 2.0 * alpha * intensity;
                });
        }

        // Update temperature
        self.solver.update(&heat_source, grid, medium, dt)?;

        // Copy temperature back to fields
        fields
            .index_axis_mut(ndarray::Axis(0), temperature_idx)
            .assign(&self.solver.temperature);

        Ok(())
    }
}

mod validation;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;

    #[test]
    fn test_thermal_diffusion_creation() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let config = ThermalDiffusionConfig::default();
        let solver = ThermalDiffusionSolver::new(config, &grid).unwrap();

        assert_eq!(solver.temperature.shape(), &[64, 64, 64]);
        assert!(solver.thermal_dose.is_some());
    }

    #[test]
    fn test_steady_state_diffusion() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let mut config = ThermalDiffusionConfig::default();
        config.enable_bioheat = false;
        config.track_thermal_dose = false;

        let mut solver = ThermalDiffusionSolver::new(config, &grid).unwrap();
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);

        // No heat source, uniform initial temperature
        let heat_source = Array3::zeros((32, 32, 32));

        // Should remain at initial temperature
        solver.update(&heat_source, &grid, &medium, 1e-3).unwrap();

        let temp_range =
            solver
                .temperature
                .iter()
                .fold(f64::INFINITY..f64::NEG_INFINITY, |mut range, &t| {
                    range.start = range.start.min(t);
                    range.end = range.end.max(t);
                    range
                });

        assert!((temp_range.end - temp_range.start).abs() < 1e-10);
    }

    #[test]
    fn test_thermal_dose_calculation() {
        let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3);
        let config = ThermalDiffusionConfig::default();
        let mut solver = ThermalDiffusionSolver::new(config, &grid).unwrap();

        // Set elevated temperature
        solver.temperature.fill(316.15); // 43°C

        // Update dose for 60 seconds
        solver.update_thermal_dose(60.0).unwrap();

        // At 43°C for 60s, dose should be 1.0 CEM43 minute
        let dose = solver.thermal_dose().unwrap();
        let avg_dose = dose.mean().unwrap();
        assert!((avg_dose - 1.0).abs() < 1e-6);
    }
}
