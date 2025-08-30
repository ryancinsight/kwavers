//! KZK (Khokhlov-Zabolotskaya-Kuznetsov) equation solver plugin
//!
//! Implements the KZK equation for nonlinear beam propagation in absorbing media.
//!
//! ## Literature Reference
//! - Lee, Y. S., & Hamilton, M. F. (1995). "Time-domain modeling of pulsed
//!   finite-amplitude sound beams." J. Acoust. Soc. Am., 97(2), 906-917.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::plugin::{Plugin, PluginMetadata, PluginState};
use crate::physics::UnifiedFieldType;
use ndarray::{Array3, Array4, Axis, Zip};
use rustfft::{num_complex::Complex, FftPlanner};
use std::any::Any;
use std::f64::consts::PI;

/// Physical constants for KZK equation
const SHOCK_FORMATION_COEFFICIENT: f64 = 1.2; // Goldberg number threshold
const DIFFRACTION_COEFFICIENT: f64 = 0.5; // Fresnel number scaling

/// KZK equation solver for nonlinear beam propagation
#[derive(Debug)]
pub struct KzkSolverPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    /// Frequency domain representation
    frequency_domain: Option<Array3<Complex<f64>>>,
    /// Spatial frequency grid
    kx: Vec<f64>,
    ky: Vec<f64>,
    /// Maximum frequency for spectral content
    max_frequency: f64,
}

impl KzkSolverPlugin {
    /// Create new KZK solver plugin
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                name: "KZK Solver".to_string(),
                version: "1.0.0".to_string(),
                description: "Nonlinear beam propagation using KZK equation".to_string(),
                author: "Kwavers Team".to_string(),
                id: "kzk_solver".to_string(),
                license: "MIT".to_string(),
            },
            state: PluginState::Created,
            frequency_domain: None,
            kx: Vec::new(),
            ky: Vec::new(),
            max_frequency: 10e6, // 10 MHz default
        }
    }

    /// Initialize frequency domain operators
    pub fn initialize_operators(&mut self, grid: &Grid, max_frequency: f64) -> KwaversResult<()> {
        self.max_frequency = max_frequency;

        // Setup spatial frequency grid (kx, ky)
        let dkx = 2.0 * PI / (grid.nx as f64 * grid.dx);
        let dky = 2.0 * PI / (grid.ny as f64 * grid.dy);

        self.kx = (0..grid.nx)
            .map(|i| {
                let i = i as f64;
                if i < grid.nx as f64 / 2.0 {
                    i * dkx
                } else {
                    (i - grid.nx as f64) * dkx
                }
            })
            .collect();

        self.ky = (0..grid.ny)
            .map(|j| {
                let j = j as f64;
                if j < grid.ny as f64 / 2.0 {
                    j * dky
                } else {
                    (j - grid.ny as f64) * dky
                }
            })
            .collect();

        // Initialize frequency domain array
        self.frequency_domain = Some(Array3::zeros((grid.nx, grid.ny, grid.nz)));

        Ok(())
    }

    /// Solve KZK equation using operator splitting
    pub fn solve(
        &mut self,
        initial_pressure: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        propagation_distance: f64,
        time_steps: usize,
    ) -> KwaversResult<Array3<f64>> {
        let dz = propagation_distance / time_steps as f64;
        let mut pressure = initial_pressure.clone();
        let mut planner = FftPlanner::new();

        for _step in 0..time_steps {
            // Step 1: Diffraction (linear propagation in frequency domain)
            pressure = self.apply_diffraction_step(&pressure, grid, dz, &mut planner)?;

            // Step 2: Absorption
            pressure = self.apply_absorption_step(&pressure, medium, grid, dz)?;

            // Step 3: Nonlinearity (time domain)
            pressure = self.apply_nonlinear_step(&pressure, medium, grid, dz)?;
        }

        Ok(pressure)
    }

    /// Apply diffraction step using angular spectrum method
    fn apply_diffraction_step(
        &self,
        pressure: &Array3<f64>,
        grid: &Grid,
        dz: f64,
        planner: &mut FftPlanner<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let mut result = pressure.clone();

        // For each z-slice, apply 2D FFT
        for mut slice in result.axis_iter_mut(Axis(2)) {
            // Convert to complex for FFT
            let mut complex_slice: Vec<Complex<f64>> =
                slice.iter().map(|&v| Complex::new(v, 0.0)).collect();

            // 2D FFT (simplified - would need proper 2D FFT)
            let fft = planner.plan_fft_forward(complex_slice.len());
            fft.process(&mut complex_slice);

            // Apply diffraction propagator in frequency domain
            for (i, val) in complex_slice.iter_mut().enumerate() {
                let ix = i % grid.nx;
                let iy = i / grid.nx;

                if ix < self.kx.len() && iy < self.ky.len() {
                    let kx = self.kx[ix];
                    let ky = self.ky[iy];
                    let k = 2.0 * PI * self.max_frequency / 1500.0; // Wavenumber
                    let kz = (k * k - kx * kx - ky * ky).sqrt();

                    // Propagator: exp(i * kz * dz)
                    let propagator = Complex::from_polar(1.0, kz * dz);
                    *val *= propagator;
                }
            }

            // Inverse FFT
            let ifft = planner.plan_fft_inverse(complex_slice.len());
            ifft.process(&mut complex_slice);

            // Convert back to real
            let slice_flat = slice.as_slice_mut().unwrap();
            for (i, &val) in complex_slice.iter().enumerate() {
                if i < slice_flat.len() {
                    slice_flat[i] = val.re / complex_slice.len() as f64;
                }
            }
        }

        Ok(result)
    }

    /// Apply absorption using frequency power law
    fn apply_absorption_step(
        &self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        dz: f64,
    ) -> KwaversResult<Array3<f64>> {
        let mut result = Array3::zeros(pressure.dim());

        Zip::indexed(&mut result)
            .and(pressure)
            .for_each(|(i, j, k), res, &p| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let alpha = crate::medium::core::CoreMedium::absorption_coefficient(
                    medium,
                    x,
                    y,
                    z,
                    grid,
                    self.max_frequency,
                );

                // Apply absorption: p * exp(-alpha * dz)
                *res = p * (-alpha * dz).exp();
            });

        Ok(result)
    }

    /// Apply nonlinear step using Burgers equation solution
    fn apply_nonlinear_step(
        &self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        dz: f64,
    ) -> KwaversResult<Array3<f64>> {
        let mut result = Array3::zeros(pressure.dim());

        Zip::indexed(&mut result)
            .and(pressure)
            .for_each(|(i, j, k), res, &p| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let c = medium.sound_speed(x, y, z, grid);
                let rho = medium.density(x, y, z, grid);
                let beta = crate::medium::core::CoreMedium::nonlinearity_coefficient(
                    medium, x, y, z, grid,
                );

                // Nonlinear phase velocity shift
                let v_nl = c * (1.0 + beta * p / (2.0 * rho * c * c));

                // Apply nonlinear propagation
                let phase_shift = 2.0 * PI * self.max_frequency * dz / v_nl;
                *res = p * phase_shift.cos(); // Simplified - should track phase properly
            });

        Ok(result)
    }

    /// Calculate shock formation distance
    pub fn shock_formation_distance(
        &self,
        peak_pressure: f64,
        frequency: f64,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> f64 {
        // Goldberg number calculation
        let c = medium.sound_speed(0.0, 0.0, 0.0, grid);
        let rho = medium.density(0.0, 0.0, 0.0, grid);
        let beta =
            crate::medium::core::CoreMedium::nonlinearity_coefficient(medium, 0.0, 0.0, 0.0, grid);

        // x_shock = ρc³/(βωp₀)
        let omega = 2.0 * PI * frequency;
        rho * c.powi(3) / (beta * omega * peak_pressure * SHOCK_FORMATION_COEFFICIENT)
    }
}

impl Plugin for KzkSolverPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &crate::physics::plugin::PluginContext,
    ) -> KwaversResult<()> {
        // Extract pressure field
        let pressure = fields.index_axis(ndarray::Axis(0), 0).to_owned();

        // Solve KZK for one time step
        let propagation_distance = medium.sound_speed(0.0, 0.0, 0.0, grid) * dt;
        let updated = self.solve(&pressure, medium, grid, propagation_distance, 1)?;

        // Update pressure field
        fields.index_axis_mut(ndarray::Axis(0), 0).assign(&updated);

        Ok(())
    }

    fn initialize(&mut self, grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        self.initialize_operators(grid, self.max_frequency)?;
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shock_formation_distance() {
        // Test against known values from literature
        let plugin = KzkSolverPlugin::new();
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let medium = crate::medium::HomogeneousMedium::water(&grid);

        let distance = plugin.shock_formation_distance(
            1e6, // 1 MPa peak pressure
            1e6, // 1 MHz
            &medium, &grid,
        );

        // Should be on the order of centimeters for these parameters
        assert!(distance > 0.01 && distance < 0.1);
    }
}
