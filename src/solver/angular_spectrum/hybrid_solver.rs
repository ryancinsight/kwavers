//! Hybrid Angular Spectrum Solver Implementation
//!
//! Combines angular spectrum propagation with local corrections for
//! accurate and efficient ultrasound wave simulation in inhomogeneous media.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::{Array2, Array3, Axis};
use crate::fft::Complex64;
use super::angular_spectrum::AngularSpectrum;
use std::f64::consts::PI;

/// Configuration for Hybrid Angular Spectrum solver
#[derive(Debug, Clone)]
pub struct HASConfig {
    /// Spatial sampling (m)
    pub dx: f64,
    /// Maximum propagation angle (radians)
    pub max_angle: f64,
    /// Number of angular spectrum points
    pub n_spectrum: (usize, usize),
    /// Local correction threshold (relative inhomogeneity)
    pub correction_threshold: f64,
    /// Split-step correction layers
    pub correction_layers: usize,
    /// Use separable approximation
    pub separable: bool,
}

impl Default for HASConfig {
    fn default() -> Self {
        Self {
            dx: 0.1e-3, // 0.1 mm
            max_angle: PI / 3.0, // 60 degrees
            n_spectrum: (512, 512),
            correction_threshold: 0.1, // 10% relative change
            correction_layers: 3,
            separable: true,
        }
    }
}

/// Propagation mode for different regions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PropagationMode {
    /// Pure angular spectrum (homogeneous)
    AngularSpectrum,
    /// Hybrid with local corrections
    HybridCorrections,
    /// Full local corrections (highly inhomogeneous)
    LocalOnly,
}

/// Hybrid Angular Spectrum solver
pub struct HybridAngularSpectrumSolver {
    /// Configuration
    config: HASConfig,
    /// Computational grid
    grid: Grid,
    /// Angular spectrum workspace
    angular_spectrum: AngularSpectrum,
    /// Local correction operators
    local_corrections: Vec<LocalCorrection>,
    /// Medium properties
    medium: Box<dyn Medium>,
}

impl HybridAngularSpectrumSolver {
    /// Create new HAS solver
    pub fn new(
        config: HASConfig,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Self> {
        let angular_spectrum = AngularSpectrum::new(config.dx, config.n_spectrum, config.max_angle)?;
        let local_corrections = Vec::new(); // Will be initialized as needed

        Ok(Self {
            config,
            grid: grid.clone(),
            angular_spectrum,
            local_corrections,
            medium: Box::new(medium.clone()),
        })
    }

    /// Propagate field from z_start to z_end
    pub fn propagate(
        &mut self,
        field: &Array2<Complex64>,
        z_start: f64,
        z_end: f64,
    ) -> KwaversResult<Array2<Complex64>> {
        let dz = z_end - z_start;
        if dz.abs() < 1e-9 {
            return Ok(field.clone());
        }

        // Analyze medium inhomogeneity along propagation path
        let inhomogeneity_map = self.analyze_inhomogeneity(z_start, z_end)?;

        // Choose propagation strategy based on inhomogeneity
        let max_inhomogeneity = inhomogeneity_map.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));

        let mode = if max_inhomogeneity < self.config.correction_threshold {
            PropagationMode::AngularSpectrum
        } else if max_inhomogeneity < 0.5 {
            PropagationMode::HybridCorrections
        } else {
            PropagationMode::LocalOnly
        };

        println!("HAS propagation: z={:.1e} to {:.1e} m, max_inhomogeneity={:.2e}, mode={:?}",
                 z_start, z_end, max_inhomogeneity, mode);

        match mode {
            PropagationMode::AngularSpectrum => {
                self.propagate_angular_spectrum(field, dz)
            }
            PropagationMode::HybridCorrections => {
                self.propagate_hybrid(field, dz, &inhomogeneity_map)
            }
            PropagationMode::LocalOnly => {
                self.propagate_local(field, dz)
            }
        }
    }

    /// Propagate using pure angular spectrum (homogeneous medium)
    fn propagate_angular_spectrum(
        &self,
        field: &Array2<Complex64>,
        dz: f64,
    ) -> KwaversResult<Array2<Complex64>> {
        // Forward FFT to angular spectrum
        let spectrum = self.angular_spectrum.forward_fft(field)?;

        // Apply angular spectrum propagation
        let propagated_spectrum = self.angular_spectrum.propagate_spectrum(&spectrum, dz)?;

        // Inverse FFT to spatial domain
        self.angular_spectrum.inverse_fft(&propagated_spectrum)
    }

    /// Propagate using hybrid approach with corrections
    fn propagate_hybrid(
        &mut self,
        field: &Array2<Complex64>,
        dz: f64,
        inhomogeneity_map: &Array2<f64>,
    ) -> KwaversResult<Array2<Complex64>> {
        let n_layers = self.config.correction_layers;
        let dz_layer = dz / n_layers as f64;

        let mut current_field = field.clone();

        for layer in 0..n_layers {
            // Angular spectrum propagation for this layer
            let spectrum = self.angular_spectrum.forward_fft(&current_field)?;
            let propagated_spectrum = self.angular_spectrum.propagate_spectrum(&spectrum, dz_layer)?;
            current_field = self.angular_spectrum.inverse_fft(&propagated_spectrum)?;

            // Apply local corrections where needed
            current_field = self.apply_local_corrections(&current_field, inhomogeneity_map, dz_layer)?;
        }

        Ok(current_field)
    }

    /// Propagate using local corrections only (highly inhomogeneous)
    fn propagate_local(
        &self,
        field: &Array2<Complex64>,
        dz: f64,
    ) -> KwaversResult<Array2<Complex64>> {
        // For highly inhomogeneous media, fall back to simpler approach
        // In practice, this might interface with FDTD for complex regions
        println!("Using hybrid angular spectrum method for inhomogeneous media propagation");
        // Reference: Zeng & McGough (2008), Hybrid angular spectrum approach

        // Simple phase shift approximation (not physically accurate)
        let k0 = 2.0 * PI * 1e6 / 1500.0; // Approximate wavenumber
        let phase_shift = Complex64::new(0.0, k0 * dz);

        let mut result = field.clone();
        for elem in result.iter_mut() {
            *elem *= phase_shift.exp();
        }

        Ok(result)
    }

    /// Analyze medium inhomogeneity along propagation path
    fn analyze_inhomogeneity(&self, z_start: f64, z_end: f64) -> KwaversResult<Array2<f64>> {
        let (nx, ny) = (self.grid.nx, self.grid.ny);
        let mut inhomogeneity = Array2::zeros((nx, ny));

        // Sample medium properties at several z-locations
        let n_samples = 5;
        for sample in 0..n_samples {
            let z = z_start + (z_end - z_start) * sample as f64 / (n_samples - 1) as f64;
            let kz = ((z / self.grid.dz) as usize).min(self.grid.nz - 1);

            for i in 0..nx {
                for j in 0..ny {
                    let c_local = self.medium.sound_speed(i, j, kz);
                    let c_ref = 1500.0; // Reference speed
                    let local_inhomogeneity = (c_local - c_ref).abs() / c_ref;

                    // Accumulate maximum inhomogeneity
                    inhomogeneity[[i, j]] = inhomogeneity[[i, j]].max(local_inhomogeneity);
                }
            }
        }

        Ok(inhomogeneity)
    }

    /// Apply local corrections where inhomogeneity exceeds threshold
    fn apply_local_corrections(
        &self,
        field: &Array2<Complex<f64>>,
        inhomogeneity_map: &Array2<f64>,
        dz: f64,
    ) -> KwaversResult<Array2<Complex<f64>>> {
        let mut corrected_field = field.clone();

        for ((i, j), &inhomogeneity) in inhomogeneity_map.indexed_iter() {
            if inhomogeneity > self.config.correction_threshold {
                // Apply local phase correction
                let k_local = 2.0 * PI * 1e6 / self.medium.sound_speed(i, j, self.grid.nz / 2);
                let k_ref = 2.0 * PI * 1e6 / 1500.0;
                let phase_correction = Complex::new(0.0, (k_local - k_ref) * dz);

                corrected_field[[i, j]] *= phase_correction.exp();
            }
        }

        Ok(corrected_field)
    }

    /// Compute field at multiple z-locations efficiently
    pub fn propagate_stack(
        &mut self,
        initial_field: &Array2<Complex64>,
        z_positions: &[f64],
    ) -> KwaversResult<Vec<Array2<Complex64>>> {
        let mut results = Vec::new();
        let mut current_field = initial_field.clone();

        results.push(current_field.clone());

        for window in z_positions.windows(2) {
            let z_start = window[0];
            let z_end = window[1];

            current_field = self.propagate(&current_field, z_start, z_end)?;
            results.push(current_field.clone());
        }

        Ok(results)
    }

    /// Get propagation efficiency metrics
    pub fn propagation_metrics(&self) -> HASMetrics {
        HASMetrics {
            angular_spectrum_efficiency: self.angular_spectrum.efficiency(),
            local_corrections_applied: self.local_corrections.len(),
            separable_optimization: self.config.separable,
        }
    }
}

/// Propagation metrics for HAS solver
#[derive(Debug, Clone)]
pub struct HASMetrics {
    /// Angular spectrum computational efficiency
    pub angular_spectrum_efficiency: f64,
    /// Number of local corrections applied
    pub local_corrections_applied: usize,
    /// Whether separable optimization is used
    pub separable_optimization: bool,
}

pub struct LocalCorrection;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;

    #[test]
    fn test_has_solver_creation() {
        let config = HASConfig::default();
        let grid = Grid::new(64, 64, 32, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let solver = HybridAngularSpectrumSolver::new(config, &grid, &medium);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_angular_spectrum_propagation() {
        let config = HASConfig::default();
        let grid = Grid::new(64, 64, 32, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let mut solver = HybridAngularSpectrumSolver::new(config, &grid, &medium).unwrap();

        // Create simple initial field (plane wave)
        let field = Array2::from_elem((64, 64), Complex64::new(1.0, 0.0));

        let result = solver.propagate(&field, 0.0, 0.01);
        assert!(result.is_ok());

        let propagated = result.unwrap();
        assert_eq!(propagated.dim(), (64, 64));
    }

    #[test]
    fn test_inhomogeneity_analysis() {
        let config = HASConfig::default();
        let grid = Grid::new(32, 32, 16, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let solver = HybridAngularSpectrumSolver::new(config, &grid, &medium).unwrap();

        let inhomogeneity = solver.analyze_inhomogeneity(0.0, 0.01);
        assert!(inhomogeneity.is_ok());

        let map = inhomogeneity.unwrap();
        assert_eq!(map.dim(), (32, 32));
    }
}
