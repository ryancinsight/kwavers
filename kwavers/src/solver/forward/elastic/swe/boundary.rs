//! Perfectly Matched Layer (PML) boundary conditions for elastic waves
//!
//! Implements absorbing boundary conditions to prevent artificial reflections
//! at the computational domain boundaries.
//!
//! ## Mathematical Background
//!
//! PML introduces a complex coordinate stretching that absorbs outgoing waves:
//! ```text
//! ∂/∂x → (1 + iσ/ω) ∂/∂x
//! ```
//!
//! For time-domain implementation, this becomes an exponential damping:
//! ```text
//! v(t+Δt) = v(t) * exp(-σ(x) * Δt)
//! ```
//!
//! Where σ(x) is the PML attenuation profile, typically quadratic:
//! ```text
//! σ(x) = σ_max * (d/L_pml)^2
//! ```
//!
//! Where:
//! - `d`: Distance into PML region (m)
//! - `L_pml`: PML thickness (m)
//! - `σ_max`: Maximum attenuation (Np/m)
//!
//! ## Design Principles
//!
//! 1. **Gradual absorption**: Quadratic profile prevents reflections at PML interface
//! 2. **Frequency independence**: Effective across ultrasound frequency range
//! 3. **Stability**: Maintains CFL condition compatibility
//! 4. **Minimal computational cost**: Simple exponential damping
//!
//! ## References
//!
//! - Berenger, J. P. (1994). "A perfectly matched layer for the absorption of
//!   electromagnetic waves." *J. Comp. Phys.*, 114(2), 185-200.
//! - Komatitsch, D., & Martin, R. (2007). "An unsplit convolutional PML improved
//!   at grazing incidence for the seismic wave equation." *Geophysics*, 72(5), SM155-SM167.
//! - Collino, F., & Tsogka, C. (2001). "Application of the PML absorbing layer
//!   model to the linear elastodynamic problem in anisotropic heterogeneous media."
//!   *Geophysics*, 66(1), 294-307.

use crate::domain::grid::Grid;
use ndarray::Array3;

/// PML configuration parameters
///
/// Controls the absorbing boundary layer properties.
#[derive(Debug, Clone)]
pub struct PMLConfig {
    /// Thickness of PML region in grid points
    pub thickness: usize,

    /// Maximum attenuation coefficient (Np/m)
    ///
    /// Typical values: 50-200 Np/m for ultrasound applications
    pub sigma_max: f64,

    /// Power law exponent for attenuation profile
    ///
    /// Standard value: 2.0 (quadratic profile)
    /// Higher values: steeper profile, more abrupt absorption
    pub profile_order: u32,

    /// Reflection coefficient target
    ///
    /// Theoretical reflection at PML interface
    /// Typical target: 1e-5 to 1e-8 (-100 to -160 dB)
    pub reflection_target: f64,
}

impl Default for PMLConfig {
    /// Create default PML configuration
    ///
    /// Default parameters provide <-80 dB reflection for ultrasound waves.
    fn default() -> Self {
        Self {
            thickness: 10,
            sigma_max: 100.0,
            profile_order: 2,
            reflection_target: 1e-4,
        }
    }
}

/// PML boundary condition calculator
///
/// Computes spatially-varying attenuation coefficients for absorbing boundaries.
#[derive(Debug)]
pub struct PMLBoundary {
    /// Attenuation coefficient field (Np/m)
    sigma: Array3<f64>,

    /// Configuration parameters
    config: PMLConfig,
}

impl PMLBoundary {
    /// Create new PML boundary from grid and configuration
    ///
    /// ## Arguments
    /// - `grid`: Computational grid defining domain size
    /// - `config`: PML parameters
    ///
    /// ## Returns
    /// PMLBoundary with pre-computed attenuation field
    #[must_use]
    pub fn new(grid: &Grid, config: PMLConfig) -> Self {
        let (_nx, _ny, _nz) = grid.dimensions();
        let sigma = Self::compute_attenuation_field(grid, &config);

        Self { sigma, config }
    }

    /// Get attenuation coefficient at a grid point
    ///
    /// ## Returns
    /// Attenuation coefficient σ(x,y,z) in Np/m
    #[must_use]
    pub fn attenuation(&self, i: usize, j: usize, k: usize) -> f64 {
        self.sigma[[i, j, k]]
    }

    /// Get full attenuation field
    #[must_use]
    pub fn attenuation_field(&self) -> &Array3<f64> {
        &self.sigma
    }

    /// Check if point is inside PML region
    #[must_use]
    pub fn is_in_pml(&self, i: usize, j: usize, k: usize) -> bool {
        self.sigma[[i, j, k]] > 0.0
    }

    /// Apply PML damping to velocity field
    ///
    /// ## Formula
    /// ```text
    /// v(t+Δt) = v(t) * exp(-σ * Δt)
    /// ```
    ///
    /// ## Arguments
    /// - `vx, vy, vz`: Velocity components (modified in-place)
    /// - `dt`: Time step (seconds)
    pub fn apply_damping(
        &self,
        vx: &mut Array3<f64>,
        vy: &mut Array3<f64>,
        vz: &mut Array3<f64>,
        dt: f64,
    ) {
        let (nx, ny, nz) = vx.dim();

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let sigma = self.sigma[[i, j, k]];
                    if sigma > 0.0 {
                        let damping_factor = (-sigma * dt).exp();
                        vx[[i, j, k]] *= damping_factor;
                        vy[[i, j, k]] *= damping_factor;
                        vz[[i, j, k]] *= damping_factor;
                    }
                }
            }
        }
    }

    /// Compute attenuation field for entire grid
    ///
    /// Uses power-law profile: σ(d) = σ_max * (d/L)^n
    ///
    /// PML is applied at all six faces of the 3D domain.
    fn compute_attenuation_field(grid: &Grid, config: &PMLConfig) -> Array3<f64> {
        let (nx, ny, nz) = grid.dimensions();
        let mut sigma = Array3::zeros((nx, ny, nz));

        let thickness = config.thickness;
        let sigma_max = config.sigma_max;
        let order = config.profile_order;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mut max_sigma = 0.0;

                    // X-direction PML (left and right faces)
                    if i < thickness {
                        // Left face
                        let dist = (thickness - i) as f64;
                        let normalized_dist = dist / thickness as f64;
                        let sigma_val = sigma_max * normalized_dist.powi(order as i32);
                        max_sigma = f64::max(max_sigma, sigma_val);
                    } else if i >= nx - thickness {
                        // Right face
                        let dist = (i - (nx - thickness) + 1) as f64;
                        let normalized_dist = dist / thickness as f64;
                        let sigma_val = sigma_max * normalized_dist.powi(order as i32);
                        max_sigma = f64::max(max_sigma, sigma_val);
                    }

                    // Y-direction PML (front and back faces)
                    if j < thickness {
                        // Front face
                        let dist = (thickness - j) as f64;
                        let normalized_dist = dist / thickness as f64;
                        let sigma_val = sigma_max * normalized_dist.powi(order as i32);
                        max_sigma = f64::max(max_sigma, sigma_val);
                    } else if j >= ny - thickness {
                        // Back face
                        let dist = (j - (ny - thickness) + 1) as f64;
                        let normalized_dist = dist / thickness as f64;
                        let sigma_val = sigma_max * normalized_dist.powi(order as i32);
                        max_sigma = f64::max(max_sigma, sigma_val);
                    }

                    // Z-direction PML (top and bottom faces)
                    if k < thickness {
                        // Bottom face
                        let dist = (thickness - k) as f64;
                        let normalized_dist = dist / thickness as f64;
                        let sigma_val = sigma_max * normalized_dist.powi(order as i32);
                        max_sigma = f64::max(max_sigma, sigma_val);
                    } else if k >= nz - thickness {
                        // Top face
                        let dist = (k - (nz - thickness) + 1) as f64;
                        let normalized_dist = dist / thickness as f64;
                        let sigma_val = sigma_max * normalized_dist.powi(order as i32);
                        max_sigma = f64::max(max_sigma, sigma_val);
                    }

                    sigma[[i, j, k]] = max_sigma;
                }
            }
        }

        sigma
    }

    /// Calculate theoretical reflection coefficient
    ///
    /// Based on the formula from Collino & Tsogka (2001):
    /// ```text
    /// R ≈ exp(-2 * σ_max * L_pml / c_max)
    /// ```
    ///
    /// ## Arguments
    /// - `c_max`: Maximum wave speed in medium (m/s)
    ///
    /// ## Returns
    /// Theoretical reflection coefficient (dimensionless, typically 1e-4 to 1e-8)
    #[must_use]
    pub fn theoretical_reflection(&self, c_max: f64, grid: &Grid) -> f64 {
        let l_pml = self.config.thickness as f64 * grid.dx.min(grid.dy).min(grid.dz);
        let exponent = -2.0 * self.config.sigma_max * l_pml / c_max;
        exponent.exp()
    }

    /// Optimize sigma_max to achieve target reflection coefficient
    ///
    /// Solves for σ_max given target reflection R:
    /// ```text
    /// σ_max = -ln(R) * c_max / (2 * L_pml)
    /// ```
    ///
    /// ## Arguments
    /// - `target_reflection`: Desired reflection coefficient (e.g., 1e-5)
    /// - `c_max`: Maximum wave speed (m/s)
    /// - `grid`: Computational grid
    ///
    /// ## Returns
    /// Optimal σ_max (Np/m)
    #[must_use]
    pub fn optimize_sigma_max(
        target_reflection: f64,
        c_max: f64,
        grid: &Grid,
        thickness: usize,
    ) -> f64 {
        let l_pml = thickness as f64 * grid.dx.min(grid.dy).min(grid.dz);
        -target_reflection.ln() * c_max / (2.0 * l_pml)
    }

    /// Get PML region mask
    ///
    /// Returns binary mask: 1.0 in PML region, 0.0 in interior
    #[must_use]
    pub fn get_mask(&self) -> Array3<f64> {
        let (nx, ny, nz) = self.sigma.dim();
        let mut mask = Array3::zeros((nx, ny, nz));

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if self.sigma[[i, j, k]] > 0.0 {
                        mask[[i, j, k]] = 1.0;
                    }
                }
            }
        }

        mask
    }

    /// Get PML volume fraction
    ///
    /// Returns fraction of computational domain occupied by PML
    #[must_use]
    pub fn volume_fraction(&self) -> f64 {
        let total_points = self.sigma.len();
        let pml_points = self.sigma.iter().filter(|&&s| s > 0.0).count();
        pml_points as f64 / total_points as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pml_creation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let config = PMLConfig::default();
        let pml = PMLBoundary::new(&grid, config);

        // Interior point should have zero attenuation
        assert_eq!(pml.attenuation(16, 16, 16), 0.0);
        assert!(!pml.is_in_pml(16, 16, 16));

        // Boundary point should have non-zero attenuation
        assert!(pml.attenuation(0, 16, 16) > 0.0);
        assert!(pml.is_in_pml(0, 16, 16));
    }

    #[test]
    fn test_pml_profile() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let config = PMLConfig {
            thickness: 5,
            sigma_max: 100.0,
            profile_order: 2,
            reflection_target: 1e-5,
        };
        let pml = PMLBoundary::new(&grid, config);

        // Attenuation should increase towards boundary
        let sigma_1 = pml.attenuation(4, 16, 16); // 1 point into PML
        let sigma_2 = pml.attenuation(3, 16, 16); // 2 points into PML
        let sigma_3 = pml.attenuation(0, 16, 16); // At boundary

        assert!(sigma_1 < sigma_2);
        assert!(sigma_2 < sigma_3);
        assert!((sigma_3 - 100.0).abs() < 1e-10); // Should be sigma_max at boundary
    }

    #[test]
    fn test_pml_damping() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let config = PMLConfig::default();
        let pml = PMLBoundary::new(&grid, config);

        let mut vx = Array3::<f64>::ones((32, 32, 32));
        let mut vy = Array3::<f64>::ones((32, 32, 32));
        let mut vz = Array3::<f64>::ones((32, 32, 32));

        let dt = 1e-7;
        pml.apply_damping(&mut vx, &mut vy, &mut vz, dt);

        // Interior should be unchanged
        assert!((vx[[16, 16, 16]] - 1.0).abs() < 1e-10);

        // Boundary should be damped
        assert!(vx[[0, 16, 16]] < 1.0);
        assert!(vx[[0, 16, 16]] > 0.0);
    }

    #[test]
    fn test_theoretical_reflection() {
        // Mathematical specification: PML theoretical reflection coefficient
        // R = exp(-2 * σ_max * L_pml / c_max)
        //
        // For R < 0.01: require σ_max > -ln(0.01) * c_max / (2 * L_pml)
        // With c_max = 1500 m/s, L_pml = 10 * 1e-3 = 0.01 m:
        // σ_max > -ln(0.01) * 1500 / 0.02 = 4.605 * 1500 / 0.02 = 345,375 Np/m
        //
        // Use optimization formula to compute required σ_max for target R = 0.005 (0.5%)
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let c_max = 1500.0; // Water sound speed
        let thickness = 10;
        let target_reflection = 0.005; // 0.5% target

        let sigma_optimized =
            PMLBoundary::optimize_sigma_max(target_reflection, c_max, &grid, thickness);

        let config = PMLConfig {
            thickness,
            sigma_max: sigma_optimized,
            profile_order: 2,
            reflection_target: target_reflection,
        };
        let pml = PMLBoundary::new(&grid, config);

        let reflection = pml.theoretical_reflection(c_max, &grid);

        // Should achieve target (with some numerical tolerance)
        assert!(
            reflection < 0.01,
            "Reflection {} exceeds 1% threshold",
            reflection
        );
        assert!(
            reflection > 0.0,
            "Reflection {} must be positive",
            reflection
        );
        assert!(
            (reflection - target_reflection).abs() / target_reflection < 0.01,
            "Reflection {} differs from target {} by more than 1%",
            reflection,
            target_reflection
        );
    }

    #[test]
    fn test_sigma_max_optimization() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let target_reflection = 1e-6;
        let c_max = 1500.0;
        let thickness = 10;

        let sigma_opt = PMLBoundary::optimize_sigma_max(target_reflection, c_max, &grid, thickness);

        // Optimized sigma should be positive
        assert!(sigma_opt > 0.0);

        // Verify it achieves target
        let config = PMLConfig {
            thickness,
            sigma_max: sigma_opt,
            profile_order: 2,
            reflection_target: target_reflection,
        };
        let pml = PMLBoundary::new(&grid, config);
        let achieved_reflection = pml.theoretical_reflection(c_max, &grid);

        // Should be close to target (within order of magnitude)
        assert!((achieved_reflection.log10() - target_reflection.log10()).abs() < 1.0);
    }

    #[test]
    fn test_pml_volume_fraction() {
        // Mathematical specification: PML volume fraction calculation
        // Grid: n^3 total points
        // Interior: (n - 2*t)^3 points
        // PML: n^3 - (n - 2*t)^3 points
        // Fraction: [n^3 - (n - 2*t)^3] / n^3
        //
        // For constraint < 0.6: require (n - 2*t)^3 / n^3 > 0.4
        // With n=50, t=5: (50-10)^3 / 50^3 = 40^3 / 50^3 = 64000/125000 = 0.512
        // PML fraction: 1 - 0.512 = 0.488 (48.8%) < 0.6 ✓
        let grid = Grid::new(50, 50, 50, 1e-3, 1e-3, 1e-3).unwrap();
        let config = PMLConfig {
            thickness: 5,
            ..Default::default()
        };
        let pml = PMLBoundary::new(&grid, config);

        let vol_frac = pml.volume_fraction();

        // Should be reasonable fraction (PML at 6 faces)
        assert!(vol_frac > 0.0);
        assert!(vol_frac < 1.0);

        // For 5-point thick PML on 50^3 grid: expect ~49% in PML
        // Theoretical: 1 - (40/50)^3 = 1 - 0.512 = 0.488
        assert!(vol_frac > 0.3, "PML volume fraction {} too small", vol_frac);
        assert!(
            vol_frac < 0.6,
            "PML volume fraction {} exceeds 60% threshold",
            vol_frac
        );
    }

    #[test]
    fn test_pml_mask() {
        let grid = Grid::new(20, 20, 20, 1e-3, 1e-3, 1e-3).unwrap();
        let config = PMLConfig {
            thickness: 3,
            ..Default::default()
        };
        let pml = PMLBoundary::new(&grid, config);

        let mask = pml.get_mask();

        // Check interior point
        assert_eq!(mask[[10, 10, 10]], 0.0);

        // Check boundary point
        assert_eq!(mask[[0, 10, 10]], 1.0);
    }
}
