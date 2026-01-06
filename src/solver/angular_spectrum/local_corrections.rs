//! Local Corrections for Hybrid Angular Spectrum Method
//!
//! Implements local correction operators for handling inhomogeneities
//! in hybrid angular spectrum wave propagation.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::fft::Complex64;
use ndarray::{Array2, Array3};

/// Types of local corrections available
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CorrectionType {
    /// Born approximation for weak inhomogeneities
    BornApproximation,
    /// Rytov approximation for phase corrections
    RytovApproximation,
    /// Kirchhoff diffraction for strong inhomogeneities
    KirchhoffDiffraction,
    /// Split-step beam propagation
    SplitStep,
}

/// Configuration for local corrections
#[derive(Debug, Clone)]
pub struct CorrectionConfig {
    /// Type of correction to apply
    pub correction_type: CorrectionType,
    /// Correction radius (in grid cells)
    pub correction_radius: usize,
    /// Maximum correction strength
    pub max_correction_strength: f64,
    /// Adaptive correction threshold
    pub adaptive_threshold: f64,
}

/// Local correction operator
#[derive(Debug)]
pub struct LocalCorrection {
    /// Correction type
    correction_type: CorrectionType,
    /// Spatial extent of correction
    radius: usize,
    /// Correction strength
    strength: f64,
    /// Position in grid coordinates
    position: (usize, usize, usize),
    /// Local correction kernel
    kernel: Array3<Complex64>,
}

impl LocalCorrection {
    /// Create new local correction
    pub fn new(
        correction_type: CorrectionType,
        radius: usize,
        strength: f64,
        position: (usize, usize, usize),
        grid: &Grid,
    ) -> KwaversResult<Self> {
        let kernel = Self::compute_correction_kernel(correction_type, radius, strength, grid)?;

        Ok(Self {
            correction_type,
            radius,
            strength,
            position,
            kernel,
        })
    }

    /// Apply correction to field
    pub fn apply(&self, field: &mut Array3<Complex64>, grid: &Grid) {
        let (px, py, pz) = self.position;
        let radius = self.radius as i32;

        // Apply correction within the kernel region
        for i in -radius..=radius {
            for j in -radius..=radius {
                for k in -radius..=radius {
                    let x = px as i32 + i;
                    let y = py as i32 + j;
                    let z = pz as i32 + k;

                    // Check bounds
                    if x >= 0 && x < grid.nx as i32 &&
                       y >= 0 && y < grid.ny as i32 &&
                       z >= 0 && z < grid.nz as i32 {
                        let kernel_idx = ((i + radius) as usize,
                                        (j + radius) as usize,
                                        (k + radius) as usize);

                        if let Some(kernel_val) = self.kernel.get(kernel_idx) {
                            field[[x as usize, y as usize, z as usize]] *= kernel_val;
                        }
                    }
                }
            }
        }
    }

    /// Compute correction kernel based on type
    fn compute_correction_kernel(
        correction_type: CorrectionType,
        radius: usize,
        strength: f64,
        grid: &Grid,
    ) -> KwaversResult<Array3<Complex64>> {
        let kernel_size = 2 * radius + 1;
        let mut kernel = Array3::zeros((kernel_size, kernel_size, kernel_size));

        match correction_type {
            CorrectionType::BornApproximation => {
                Self::born_approximation_kernel(&mut kernel, radius, strength, grid)?;
            }
            CorrectionType::RytovApproximation => {
                Self::rytov_approximation_kernel(&mut kernel, radius, strength, grid)?;
            }
            CorrectionType::KirchhoffDiffraction => {
                Self::kirchhoff_diffraction_kernel(&mut kernel, radius, strength, grid)?;
            }
            CorrectionType::SplitStep => {
                Self::split_step_kernel(&mut kernel, radius, strength, grid)?;
            }
        }

        Ok(kernel)
    }

    /// Born approximation kernel for weak inhomogeneities
    fn born_approximation_kernel(
        kernel: &mut Array3<Complex64>,
        radius: usize,
        strength: f64,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let k0 = 2.0 * std::f64::consts::PI / 0.0005; // Wavenumber (approximate)

        for i in 0..kernel.nrows() {
            for j in 0..kernel.ncols() {
                for k in 0..kernel.dim().2 {
                    let dx = (i as i32 - radius as i32) as f64 * grid.dx;
                    let dy = (j as i32 - radius as i32) as f64 * grid.dy;
                    let dz = (k as i32 - radius as i32) as f64 * grid.dz;

                    let r = (dx*dx + dy*dy + dz*dz).sqrt();

                    if r > 0.0 {
                        // Born approximation: exp(i k0 r) / r
                        let phase = Complex64::new(0.0, k0 * r);
                        kernel[[i, j, k]] = Complex64::new(strength, 0.0) * phase.exp() / r;
                    }
                }
            }
        }

        Ok(())
    }

    /// Rytov approximation kernel for phase corrections
    fn rytov_approximation_kernel(
        kernel: &mut Array3<Complex64>,
        radius: usize,
        strength: f64,
        grid: &Grid,
    ) -> KwaversResult<()> {
        // Rytov approximation: phase-only correction
        for i in 0..kernel.nrows() {
            for j in 0..kernel.ncols() {
                for k in 0..kernel.dim().2 {
                    let dx = (i as i32 - radius as i32) as f64 * grid.dx;
                    let dy = (j as i32 - radius as i32) as f64 * grid.dy;
                    let dz = (k as i32 - radius as i32) as f64 * grid.dz;

                    let r_squared = dx*dx + dy*dy + dz*dz;
                    let phase_correction = strength * r_squared;

                    kernel[[i, j, k]] = Complex64::new(0.0, phase_correction).exp();
                }
            }
        }

        Ok(())
    }

    /// Kirchhoff diffraction kernel for strong inhomogeneities
    fn kirchhoff_diffraction_kernel(
        kernel: &mut Array3<Complex64>,
        radius: usize,
        strength: f64,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let k0 = 2.0 * std::f64::consts::PI / 0.0005; // Wavenumber

        for i in 0..kernel.nrows() {
            for j in 0..kernel.ncols() {
                for k in 0..kernel.dim().2 {
                    let dx = (i as i32 - radius as i32) as f64 * grid.dx;
                    let dy = (j as i32 - radius as i32) as f64 * grid.dy;
                    let dz = (k as i32 - radius as i32) as f64 * grid.dz;

                    let r = (dx*dx + dy*dy + dz*dz).sqrt();

                    if r > 0.0 {
                        // Kirchhoff diffraction: (i k0 / (4π r)) * exp(i k0 r) * obliquity_factor
                        let obliquity = 1.0; // Simplified - would compute actual obliquity
                        let phase = Complex64::new(0.0, k0 * r);
                        kernel[[i, j, k]] = Complex64::new(0.0, k0 / (4.0 * std::f64::consts::PI * r))
                                           * phase.exp() * obliquity * strength;
                    }
                }
            }
        }

        Ok(())
    }

    /// Split-step beam propagation kernel
    fn split_step_kernel(
        kernel: &mut Array3<Complex64>,
        radius: usize,
        strength: f64,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let dz = grid.dz;
        let k0 = 2.0 * std::f64::consts::PI / 0.0005;

        // Split-step propagation: exp(i k0 z) * exp(-i k0² z / (2k)) for paraxial approximation
        for i in 0..kernel.nrows() {
            for j in 0..kernel.ncols() {
                for k in 0..kernel.dim().2 {
                    let x = (i as i32 - radius as i32) as f64 * grid.dx;
                    let y = (j as i32 - radius as i32) as f64 * grid.dy;

                    let transverse_k_squared = (2.0 * std::f64::consts::PI * x / (grid.nx as f64 * grid.dx)).powi(2) +
                                             (2.0 * std::f64::consts::PI * y / (grid.ny as f64 * grid.dy)).powi(2);

                    // Paraxial phase correction
                    let phase_correction = -transverse_k_squared * dz / (2.0 * k0);
                    let longitudinal_phase = k0 * dz * (k as f64 - radius as f64);

                    let total_phase = longitudinal_phase + phase_correction;
                    kernel[[i, j, k]] = Complex64::new(0.0, total_phase * strength).exp();
                }
            }
        }

        Ok(())
    }

    /// Get correction strength
    pub fn strength(&self) -> f64 {
        self.strength
    }

    /// Get correction radius
    pub fn radius(&self) -> usize {
        self.radius
    }

    /// Get correction position
    pub fn position(&self) -> (usize, usize, usize) {
        self.position
    }
}

/// Manager for local corrections in hybrid method
pub struct CorrectionManager {
    corrections: Vec<LocalCorrection>,
    config: CorrectionConfig,
}

impl CorrectionManager {
    /// Create new correction manager
    pub fn new(config: CorrectionConfig) -> Self {
        Self {
            corrections: Vec::new(),
            config,
        }
    }

    /// Add correction at specified position
    pub fn add_correction(&mut self, position: (usize, usize, usize), strength: f64, grid: &Grid) -> KwaversResult<()> {
        let correction = LocalCorrection::new(
            self.config.correction_type,
            self.config.correction_radius,
            strength.min(self.config.max_correction_strength),
            position,
            grid,
        )?;

        self.corrections.push(correction);
        Ok(())
    }

    /// Apply all corrections to field
    pub fn apply_corrections(&self, field: &mut Array3<Complex64>, grid: &Grid) {
        for correction in &self.corrections {
            correction.apply(field, grid);
        }
    }

    /// Detect regions needing corrections based on medium inhomogeneity
    pub fn detect_corrections(&mut self, medium_variation: &Array3<f64>, grid: &Grid) -> KwaversResult<()> {
        let threshold = self.config.adaptive_threshold;

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    if medium_variation[[i, j, k]] > threshold {
                        self.add_correction((i, j, k), medium_variation[[i, j, k]], grid)?;
                    }
                }
            }
        }

        println!("Added {} local corrections based on medium inhomogeneity", self.corrections.len());
        Ok(())
    }

    /// Get number of active corrections
    pub fn num_corrections(&self) -> usize {
        self.corrections.len()
    }

    /// Clear all corrections
    pub fn clear_corrections(&mut self) {
        self.corrections.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_correction_creation() {
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();

        let correction = LocalCorrection::new(
            CorrectionType::BornApproximation,
            3,
            0.1,
            (16, 16, 16),
            &grid,
        );

        assert!(correction.is_ok());
        let corr = correction.unwrap();
        assert_eq!(corr.radius(), 3);
        assert_eq!(corr.strength(), 0.1);
    }

    #[test]
    fn test_correction_manager() {
        let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();
        let config = CorrectionConfig {
            correction_type: CorrectionType::RytovApproximation,
            correction_radius: 2,
            max_correction_strength: 1.0,
            adaptive_threshold: 0.1,
        };

        let mut manager = CorrectionManager::new(config);

        // Test adding correction
        let result = manager.add_correction((8, 8, 8), 0.5, &grid);
        assert!(result.is_ok());
        assert_eq!(manager.num_corrections(), 1);

        // Test clearing corrections
        manager.clear_corrections();
        assert_eq!(manager.num_corrections(), 0);
    }

    #[test]
    fn test_correction_types() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();

        // Test all correction types
        for correction_type in [
            CorrectionType::BornApproximation,
            CorrectionType::RytovApproximation,
            CorrectionType::KirchhoffDiffraction,
            CorrectionType::SplitStep,
        ] {
            let correction = LocalCorrection::new(correction_type, 2, 0.1, (4, 4, 4), &grid);
            assert!(correction.is_ok());
        }
    }
}








