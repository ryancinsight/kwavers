//! Modified Born Series for Viscoacoustic Media
//!
//! This module implements the modified Born series method for viscoacoustic
//! wave propagation, extending the standard Born approximation to include
//! absorption and dispersion effects from viscosity and thermal conduction.
//!
//! ## Mathematical Foundation
//!
//! The viscoacoustic wave equation includes absorption and dispersion:
//! ```text
//! ∇²p - (1/c₀²)∂²p/∂t² + (δ/c₀⁴)∂³p/∂t³ = -β ∂²(p²)/∂t² + S
//! ```
//!
//! Where δ is the diffusivity of sound, related to absorption.
//!
//! ## Modified Born Approximation
//!
//! The modified Born series accounts for absorption in the scattering process:
//! ```text
//! p = pⁱ + pˢ
//! ∇²pˢ + k²(1 + V - iα)pˢ = -k²(1 - iα)V pⁱ
//! ```
//!
//! Where α is the absorption coefficient.
//!
//! ## Higher-Order Modified Born Series
//!
//! The nth-order term includes absorption effects:
//! ```text
//! ∇²pₙ + k²(1 + V - iα)pₙ = -k²(1 - iα)V p_{n-1}
//! ```
//!
//! ## Absorption-Dispersion Relationship
//!
//! The absorption coefficient α is frequency-dependent:
//! ```text
//! α(ω) = (ω² δ)/(2 c₀³)
//! ```
//!
//! Where δ is the acoustic diffusivity.
//!
//! ## References
//!
//! Sun, Y., et al. (2025). "A viscoacoustic wave equation solver using
//! modified Born series"

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::{Array3, ArrayView3, ArrayViewMut3};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Modified Born series solver for viscoacoustic media
#[derive(Debug)]
pub struct ModifiedBornSolver {
    /// Solver configuration
    config: super::BornConfig,
    /// Computational grid
    grid: Grid,
    /// Workspace for computations
    workspace: super::BornWorkspace,
    /// Absorption coefficient field
    absorption_field: Array3<Complex64>,
    /// Diffusivity field
    diffusivity_field: Array3<f64>,
}

impl ModifiedBornSolver {
    /// Create a new modified Born solver
    pub fn new(config: super::BornConfig, grid: Grid) -> Self {
        let workspace = super::BornWorkspace::new(grid.nx, grid.ny, grid.nz);
        let shape = (grid.nx, grid.ny, grid.nz);

        Self {
            config,
            grid,
            workspace,
            absorption_field: Array3::zeros(shape),
            diffusivity_field: Array3::zeros(shape),
        }
    }

    /// Precompute absorption and diffusivity fields
    pub fn precompute_viscoacoustic_properties<M: Medium>(
        &mut self,
        frequency: f64,
        medium: &M,
    ) -> KwaversResult<()> {
        let omega = 2.0 * std::f64::consts::PI * frequency;

        // Compute absorption coefficient α(ω) = (ω² δ)/(2 c₀³)
        // and diffusivity δ for each grid point
        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let c0 = medium.sound_speed(i, j, k);
                    let diffusivity = self.compute_diffusivity(medium, i, j, k);

                    self.diffusivity_field[[i, j, k]] = diffusivity;

                    // Absorption coefficient with frequency dependence
                    let absorption = (omega * omega * diffusivity) / (2.0 * c0 * c0 * c0);
                    self.absorption_field[[i, j, k]] = Complex64::new(0.0, absorption);
                }
            }
        }

        Ok(())
    }

    /// Solve viscoacoustic Helmholtz equation using modified Born series
    pub fn solve<M: Medium>(
        &mut self,
        wavenumber: f64,
        frequency: f64,
        medium: &M,
        incident_field: ArrayView3<Complex64>,
        mut result: ArrayViewMut3<Complex64>,
    ) -> KwaversResult<ModifiedBornStats> {
        // Precompute viscoacoustic properties if not done
        self.precompute_viscoacoustic_properties(frequency, medium)?;

        self.workspace.clear();

        // Initialize with incident field
        self.workspace.field_workspace.assign(&incident_field);

        let mut total_field = incident_field.to_owned();
        let mut stats = ModifiedBornStats::default();

        // Compute Born series terms
        for order in 1..=self.config.max_order as usize {
            let scattering_field =
                self.compute_modified_born_term(wavenumber, frequency, medium, total_field.view())?;

            // Add scattering contribution
            total_field += &scattering_field;

            // Check convergence
            let residual = self.compute_viscoacoustic_residual(
                wavenumber,
                frequency,
                medium,
                total_field.view(),
            );

            stats.orders_computed = order;
            stats.final_residual = residual;

            if residual < self.config.tolerance {
                stats.converged = true;
                break;
            }

            if order >= 10 {
                // Prevent excessive iterations
                break;
            }
        }

        result.assign(&total_field);
        Ok(stats)
    }

    /// Compute one term of the modified Born series
    fn compute_modified_born_term<M: Medium>(
        &mut self,
        wavenumber: f64,
        frequency: f64,
        medium: &M,
        current_field: ArrayView3<Complex64>,
    ) -> KwaversResult<Array3<Complex64>> {
        let k_squared = wavenumber * wavenumber;
        let mut scattering_field =
            Array3::<Complex64>::zeros((self.grid.nx, self.grid.ny, self.grid.nz));

        // Compute modified scattering source: -k²(1 - iα)V ψ
        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    // Compute heterogeneity potential V
                    let v = self.compute_heterogeneity_potential(medium, i, j, k);

                    // Get absorption coefficient
                    let absorption = self.absorption_field[[i, j, k]];

                    // Modified Born source term: -k²(1 - iα)V ψ
                    let source_term = -k_squared
                        * (Complex64::new(1.0, 0.0) - absorption)
                        * v
                        * current_field[[i, j, k]];

                    self.workspace.heterogeneity_workspace[[i, j, k]] = source_term;
                }
            }
        }

        // Apply viscoacoustic Green's function
        self.apply_viscoacoustic_green(wavenumber, frequency)?;

        scattering_field.assign(&self.workspace.green_workspace);

        Ok(scattering_field)
    }

    /// Compute heterogeneity potential V = 1 - (ρ c²)/(ρ₀ c₀²)
    fn compute_heterogeneity_potential<M: Medium>(
        &self,
        medium: &M,
        i: usize,
        j: usize,
        k: usize,
    ) -> f64 {
        let c_local = medium.sound_speed(i, j, k);
        let rho_local = medium.density(i, j, k);

        // Reference values
        let c0 = 1500.0; // m/s
        let rho0 = 1000.0; // kg/m³

        1.0 - (rho_local * c_local * c_local) / (rho0 * c0 * c0)
    }

    /// Compute acoustic diffusivity δ from medium properties
    fn compute_diffusivity<M: Medium>(&self, medium: &M, i: usize, j: usize, k: usize) -> f64 {
        // Simplified diffusivity calculation
        // In practice, this should include viscosity, thermal conductivity, etc.
        // δ = (4/3) μ_B / ρ + thermal terms

        let rho = medium.density(i, j, k);
        let viscosity = 0.001; // Pa·s (water)
        let thermal_conductivity = 0.6; // W/(m·K)
        let heat_capacity = 4186.0; // J/(kg·K)
        let _temperature = 293.0; // K

        // Bulk viscosity contribution
        let viscous_diffusivity = (4.0 / 3.0) * viscosity / rho;

        // Thermal contribution (simplified)
        let thermal_diffusivity = thermal_conductivity / (rho * heat_capacity);

        viscous_diffusivity + thermal_diffusivity
    }

    /// Apply viscoacoustic Green's function
    fn apply_viscoacoustic_green(&mut self, wavenumber: f64, _frequency: f64) -> KwaversResult<()> {
        // Apply viscoacoustic Green's function with absorption
        // For efficiency, use local approximation similar to other solvers

        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;

        // Clear result array
        self.workspace
            .green_workspace
            .fill(Complex64::new(0.0, 0.0));

        // Sequential computation for now to avoid borrowing issues
        // TODO: Implement proper parallel Green's function computation
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let source_val = self.workspace.heterogeneity_workspace[[i, j, k]];
                    let absorption = self.absorption_field[[i, j, k]];

                    // Self-contribution with absorption regularization
                    let k_complex = Complex64::new(wavenumber, absorption.im);
                    let self_green = Complex64::new(0.5, 0.0) / k_complex.norm_sqr();
                    self.workspace.green_workspace[[i, j, k]] += self_green * source_val;

                    // Contributions from nearest neighbors with absorption
                    let neighbors = [
                        (i.saturating_sub(1), j, k),
                        ((i + 1).min(nx - 1), j, k),
                        (i, j.saturating_sub(1), k),
                        (i, (j + 1).min(ny - 1), k),
                        (i, j, k.saturating_sub(1)),
                        (i, j, (k + 1).min(nz - 1)),
                    ];

                    for (ni, nj, nk) in neighbors {
                        let dx = (ni as f64 - i as f64) * self.grid.dx;
                        let dy = (nj as f64 - j as f64) * self.grid.dy;
                        let dz = (nk as f64 - k as f64) * self.grid.dz;
                        let r = (dx * dx + dy * dy + dz * dz).sqrt();

                        if r > 1e-12 {
                            // Viscoacoustic Green's function: exp(ik_complex * r)/(4πr)
                            let kr_real = wavenumber * r;
                            let kr_imag = absorption.im * r;
                            let exp_factor = Complex64::from_polar(1.0, kr_real)
                                * Complex64::exp(Complex64::new(0.0, -kr_imag));
                            let green_val = exp_factor / (4.0 * PI * r);
                            self.workspace.green_workspace[[ni, nj, nk]] += green_val * source_val;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute viscoacoustic residual for convergence check
    fn compute_viscoacoustic_residual<M: Medium>(
        &self,
        wavenumber: f64,
        _frequency: f64,
        medium: &M,
        field: ArrayView3<Complex64>,
    ) -> f64 {
        let mut residual_sum = 0.0;

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let field_val = field[[i, j, k]];

                    // Compute Laplacian of the field
                    let laplacian = self.compute_laplacian(field.view(), i, j, k);

                    // Compute viscoacoustic Helmholtz residual
                    let absorption = self.absorption_field[[i, j, k]];
                    let heterogeneity = self.compute_heterogeneity_potential(medium, i, j, k);

                    // Viscoacoustic Helmholtz equation: ∇²ψ + k²(1 + V - iα)ψ = 0
                    let k_complex = Complex64::new(wavenumber, absorption.im);
                    let helmholtz_term = k_complex
                        * k_complex
                        * (Complex64::new(1.0 + heterogeneity, 0.0) - absorption);

                    let residual = laplacian + helmholtz_term * field_val;
                    residual_sum += residual.norm_sqr();
                }
            }
        }

        (residual_sum / (self.grid.nx * self.grid.ny * self.grid.nz) as f64).sqrt()
    }

    /// Compute Laplacian of a field using finite differences
    fn compute_laplacian(
        &self,
        field: ArrayView3<Complex64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Complex64 {
        // Use high-order finite differences for better accuracy
        let d2x = self.fourth_order_derivative_x(field, i, j, k);
        let d2y = self.fourth_order_derivative_y(field, i, j, k);
        let d2z = self.fourth_order_derivative_z(field, i, j, k);

        d2x + d2y + d2z
    }

    /// Compute second derivative in x-direction
    fn second_derivative_x(
        &self,
        field: ArrayView3<Complex64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Complex64 {
        let dx2 = self.grid.dx * self.grid.dx;

        if i == 0 {
            // Forward difference at boundary
            let f0 = field[[0, j, k]];
            let f1 = field[[1, j, k]];
            let f2 = field[[2, j, k]];
            (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f2) / dx2
        } else if i == self.grid.nx - 1 {
            // Backward difference at boundary
            let f0 = field[[i, j, k]];
            let fm1 = field[[i - 1, j, k]];
            let fm2 = field[[i - 2, j, k]];
            (2.0 * f0 - 5.0 * fm1 + 4.0 * fm2 - fm2) / dx2
        } else {
            // Central difference
            let fm1 = field[[i - 1, j, k]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i + 1, j, k]];
            (fm1 - 2.0 * f0 + fp1) / dx2
        }
    }

    /// Compute second derivative in y-direction
    fn second_derivative_y(
        &self,
        field: ArrayView3<Complex64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Complex64 {
        let dy2 = self.grid.dy * self.grid.dy;

        if j == 0 {
            let f0 = field[[i, 0, k]];
            let f1 = field[[i, 1, k]];
            let f2 = field[[i, 2, k]];
            (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f2) / dy2
        } else if j == self.grid.ny - 1 {
            let f0 = field[[i, j, k]];
            let fm1 = field[[i, j - 1, k]];
            let fm2 = field[[i, j - 2, k]];
            (2.0 * f0 - 5.0 * fm1 + 4.0 * fm2 - fm2) / dy2
        } else {
            let fm1 = field[[i, j - 1, k]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i, j + 1, k]];
            (fm1 - 2.0 * f0 + fp1) / dy2
        }
    }

    /// Compute second derivative in z-direction
    fn second_derivative_z(
        &self,
        field: ArrayView3<Complex64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Complex64 {
        let dz2 = self.grid.dz * self.grid.dz;

        if k == 0 {
            let f0 = field[[i, j, 0]];
            let f1 = field[[i, j, 1]];
            let f2 = field[[i, j, 2]];
            (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f2) / dz2
        } else if k == self.grid.nz - 1 {
            let f0 = field[[i, j, k]];
            let fm1 = field[[i, j, k - 1]];
            let fm2 = field[[i, j, k - 2]];
            (2.0 * f0 - 5.0 * fm1 + 4.0 * fm2 - fm2) / dz2
        } else {
            let fm1 = field[[i, j, k - 1]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i, j, k + 1]];
            (fm1 - 2.0 * f0 + fp1) / dz2
        }
    }

    /// Compute fourth-order second derivative in x-direction
    fn fourth_order_derivative_x(
        &self,
        field: ArrayView3<Complex64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Complex64 {
        let dx2 = self.grid.dx * self.grid.dx;

        // Fourth-order central difference: (1/12)(-f_{i-2} + 16f_{i-1} - 30f_i + 16f_{i+1} - f_{i+2})
        if i >= 2 && i < self.grid.nx - 2 {
            let fm2 = field[[i - 2, j, k]];
            let fm1 = field[[i - 1, j, k]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i + 1, j, k]];
            let fp2 = field[[i + 2, j, k]];

            (-fm2 + 16.0 * fm1 - 30.0 * f0 + 16.0 * fp1 - fp2) / (12.0 * dx2)
        } else {
            // Fallback to second-order for boundary points
            self.second_derivative_x(field, i, j, k)
        }
    }

    /// Compute fourth-order second derivative in y-direction
    fn fourth_order_derivative_y(
        &self,
        field: ArrayView3<Complex64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Complex64 {
        let dy2 = self.grid.dy * self.grid.dy;

        // Fourth-order central difference: (1/12)(-f_{j-2} + 16f_{j-1} - 30f_j + 16f_{j+1} - f_{j+2})
        if j >= 2 && j < self.grid.ny - 2 {
            let fm2 = field[[i, j - 2, k]];
            let fm1 = field[[i, j - 1, k]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i, j + 1, k]];
            let fp2 = field[[i, j + 2, k]];

            (-fm2 + 16.0 * fm1 - 30.0 * f0 + 16.0 * fp1 - fp2) / (12.0 * dy2)
        } else {
            // Fallback to second-order for boundary points
            self.second_derivative_y(field, i, j, k)
        }
    }

    /// Compute fourth-order second derivative in z-direction
    fn fourth_order_derivative_z(
        &self,
        field: ArrayView3<Complex64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Complex64 {
        let dz2 = self.grid.dz * self.grid.dz;

        // Fourth-order central difference: (1/12)(-f_{k-2} + 16f_{k-1} - 30f_k + 16f_{k+1} - f_{k+2})
        if k >= 2 && k < self.grid.nz - 2 {
            let fm2 = field[[i, j, k - 2]];
            let fm1 = field[[i, j, k - 1]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i, j, k + 1]];
            let fp2 = field[[i, j, k + 2]];

            (-fm2 + 16.0 * fm1 - 30.0 * f0 + 16.0 * fp1 - fp2) / (12.0 * dz2)
        } else {
            // Fallback to second-order for boundary points
            self.second_derivative_z(field, i, j, k)
        }
    }
}

/// Statistics from modified Born series solution
#[derive(Debug, Clone, Default)]
pub struct ModifiedBornStats {
    /// Number of Born series orders computed
    pub orders_computed: usize,
    /// Final residual value
    pub final_residual: f64,
    /// Whether convergence was achieved
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::solver::forward::helmholtz::BornConfig;

    #[test]
    fn test_modified_born_creation() {
        let config = BornConfig::default();
        let grid = Grid::new(16, 16, 16, 0.1, 0.1, 0.1).unwrap();

        let solver = ModifiedBornSolver::new(config, grid);
        assert_eq!(solver.config.max_iterations, 50);
        assert_eq!(solver.grid.nx, 16);
    }
}
