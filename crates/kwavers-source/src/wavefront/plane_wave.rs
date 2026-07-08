//! Plane wave source implementation
//!
//! This source generates a plane wave propagating in a specified direction.
//! Commonly used for testing and validation against analytical solutions.

use crate::{Source, SourceField};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_grid::Grid;
use kwavers_signal::Signal;
use leto::Array3;
use std::fmt::Debug;
use std::sync::Arc;

/// Injection mode for plane wave sources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InjectionMode {
    /// Inject at boundary plane only (correct arrival timing)
    #[default]
    BoundaryOnly,
    /// Pre-populate spatial pattern across grid (legacy mode)
    FullGrid,
}

/// Plane wave source configuration
#[derive(Debug, Clone)]
pub struct PlaneWaveSourceConfig {
    /// Direction vector (unit vector)
    pub direction: (f64, f64, f64),
    /// Wavelength in meters
    pub wavelength: f64,
    /// Phase offset in radians
    pub phase: f64,
    /// Source field type (pressure or velocity)
    pub source_type: SourceField,
    /// Injection mode (boundary-only or full-grid)
    pub injection_mode: InjectionMode,
}

impl Default for PlaneWaveSourceConfig {
    fn default() -> Self {
        Self {
            direction: (1.0, 0.0, 0.0), // Default: propagate along x-axis
            wavelength: 1.5e-3,         // Default: 1mm wavelength (1.5MHz in water)
            phase: 0.0,
            source_type: SourceField::Pressure,
            injection_mode: InjectionMode::default(),
        }
    }
}

/// Plane wave source implementation
#[derive(Debug)]
pub struct PlaneWaveSource {
    config: PlaneWaveSourceConfig,
    signal: Arc<dyn Signal>,
    wave_number: f64,
}

impl PlaneWaveSource {
    /// Create a new plane wave source
    pub fn new(config: PlaneWaveSourceConfig, signal: Arc<dyn Signal>) -> Self {
        let wave_number = TWO_PI / config.wavelength;
        Self {
            config,
            signal,
            wave_number,
        }
    }

    /// Create a plane wave source with default configuration
    pub fn new_default(signal: Arc<dyn Signal>) -> Self {
        Self::new(PlaneWaveSourceConfig::default(), signal)
    }

    /// Get the wave number (k = 2π/λ)
    #[must_use]
    pub fn wave_number(&self) -> f64 {
        self.wave_number
    }

    /// Get the propagation direction
    #[must_use]
    pub fn direction(&self) -> (f64, f64, f64) {
        self.config.direction
    }
}

impl Source for PlaneWaveSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros([grid.nx, grid.ny, grid.nz]);
        self.create_mask_into(grid, &mut mask);
        mask
    }

    fn create_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.shape(), [grid.nx, grid.ny, grid.nz]);
        mask.fill(0.0);

        self.add_mask_into(grid, mask);
    }

    fn add_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.shape(), [grid.nx, grid.ny, grid.nz]);
        if grid.nx == 0 || grid.ny == 0 || grid.nz == 0 {
            return;
        }

        match self.config.injection_mode {
            InjectionMode::BoundaryOnly => {
                // Determine which boundary based on dominant direction component
                let (dx, dy, dz) = self.config.direction;
                let abs_dx = dx.abs();
                let abs_dy = dy.abs();
                let abs_dz = dz.abs();

                if abs_dx >= abs_dy && abs_dx >= abs_dz {
                    // X-direction: inject at x=0 or x=max
                    let plane_idx = if dx > 0.0 { 0 } else { grid.nx - 1 };
                    for j in 0..grid.ny {
                        for k in 0..grid.nz {
                            mask[[plane_idx, j, k]] += 1.0;
                        }
                    }
                } else if abs_dy >= abs_dz {
                    // Y-direction: inject at y=0 or y=max
                    let plane_idx = if dy > 0.0 { 0 } else { grid.ny - 1 };
                    for i in 0..grid.nx {
                        for k in 0..grid.nz {
                            mask[[i, plane_idx, k]] += 1.0;
                        }
                    }
                } else {
                    // Z-direction: inject at z=0 or z=max
                    let plane_idx = if dz > 0.0 { 0 } else { grid.nz - 1 };
                    for i in 0..grid.nx {
                        for j in 0..grid.ny {
                            mask[[i, j, plane_idx]] += 1.0;
                        }
                    }
                }
            }
            InjectionMode::FullGrid => {
                // Apply spatial phase variation based on wave propagation
                for ([i, j, k], val) in mask.indexed_iter_mut().expect("valid array layout") {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    // Spatial phase: k • r = kx*x + ky*y + kz*z
                    let spatial_phase = self.wave_number
                        * (self.config.direction.2.mul_add(
                            z,
                            self.config
                                .direction
                                .0
                                .mul_add(x, self.config.direction.1 * y),
                        ) + self.config.phase);

                    *val += spatial_phase.cos(); // Spatial variation of the wave
                }
            }
        }
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        // Plane waves don't have specific positions - they affect the entire domain
        vec![]
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }

    fn source_type(&self) -> SourceField {
        self.config.source_type
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, _grid: &Grid) -> f64 {
        // Plane wave source term: A * cos(k•r - ωt + φ)
        let spatial_phase = self.wave_number
            * (self.config.direction.2.mul_add(
                z,
                self.config
                    .direction
                    .0
                    .mul_add(x, self.config.direction.1 * y),
            ) + self.config.phase);

        let temporal_phase = self.signal.amplitude(t);
        spatial_phase.cos() * temporal_phase
    }
}

/// Builder pattern for plane wave source
#[derive(Debug, Default)]
pub struct PlaneWaveBuilder {
    config: PlaneWaveSourceConfig,
}

impl PlaneWaveBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn direction(mut self, direction: (f64, f64, f64)) -> Self {
        self.config.direction = direction;
        self
    }

    #[must_use]
    pub fn wavelength(mut self, wavelength: f64) -> Self {
        self.config.wavelength = wavelength;
        self
    }

    #[must_use]
    pub fn phase(mut self, phase: f64) -> Self {
        self.config.phase = phase;
        self
    }

    #[must_use]
    pub fn source_type(mut self, source_type: SourceField) -> Self {
        self.config.source_type = source_type;
        self
    }

    #[must_use]
    pub fn injection_mode(mut self, mode: InjectionMode) -> Self {
        self.config.injection_mode = mode;
        self
    }

    pub fn build(self, signal: Arc<dyn Signal>) -> PlaneWaveSource {
        PlaneWaveSource::new(self.config, signal)
    }
}
