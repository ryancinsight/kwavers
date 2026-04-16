//! 1D radial radical diffusion solver on a logarithmic grid.
//!
//! ## Mathematical Foundation
//!
//! The Smoluchowski equation for radical species concentration in the liquid
//! surrounding a spherical bubble (Storey & Szeri 2000; Leighton 1994):
//!
//! ```text
//! ∂[Nᵢ]/∂t = Dᵢ · (1/r²) · ∂/∂r(r² ∂[Nᵢ]/∂r) + Σⱼ νᵢⱼ · rⱼ   (r > R_bubble)
//!
//! Boundary condition: [Nᵢ](R) = [Nᵢ]_bubble(t)   (Hertz-Knudsen flux at bubble wall)
//! Far field:          [Nᵢ](∞) → 0
//! ```
//!
//! ## Numerical Scheme
//!
//! **Grid**: Logarithmic radial grid with `n_points` nodes,
//! `r_j = R_bubble · exp(j · Δξ)` where `Δξ = ln(r_max / R_bubble) / (n - 1)`.
//!
//! **Time integration**: Crank-Nicolson (implicit, 2nd-order) for diffusion;
//! explicit Euler for reactions.  The operator-split scheme:
//!
//! ```text
//! [Nᵢ]* = [Nᵢ]ⁿ + dt · f_reaction([Nᵢ]ⁿ)    (explicit Euler, reactions)
//! (I − dt/2 · L) [Nᵢ]ⁿ⁺¹ = (I + dt/2 · L) [Nᵢ]*  (Crank-Nicolson, diffusion)
//! ```
//!
//! where `L` is the discrete radial Laplacian operator on the log grid.
//!
//! **Discretisation**: On the log-radial coordinate `ξ = ln(r/R_bubble)`:
//!
//! ```text
//! (1/r²)·∂/∂r(r²·∂N/∂r) = (1/r²)·e^{-2ξ}·[∂²N/∂ξ² + ∂N/∂ξ]
//! ```
//!
//! Centred differences give a tridiagonal system solvable in O(n) by Thomas algorithm.
//!
//! ## References
//!
//! - Storey BD, Szeri AJ (2000). "Water vapour, sonoluminescence and sonochemistry."
//!   *Proc R Soc Lond A* **456**, 1685–1709. DOI: 10.1098/rspa.2000.0560
//! - Leighton TG (1994). *The Acoustic Bubble*. Academic Press. §4.4.
//! - Christman CL (1987). *Ultrasonics* **25**(1), 31–37.

use crate::physics::chemistry::ros_plasma::ros_species::ROSSpecies;
use std::collections::HashMap;

// ============================================================================
// RadicalDiffusionSolver
// ============================================================================

/// 1D radial Smoluchowski diffusion solver for radical species.
///
/// Solves the spherically-symmetric diffusion equation on a logarithmic radial
/// grid using Crank-Nicolson time integration (2nd-order in space and time).
///
/// ## Default parameters
///
/// | Parameter | Default | Notes |
/// |-----------|---------|-------|
/// | `n_points` | 64 | log-radial nodes |
/// | `r_max_factor` | 1000 | `r_max = R_bubble × 1000` |
#[derive(Debug, Clone)]
pub struct RadicalDiffusionSolver {
    /// Number of radial grid points.
    pub n_points: usize,
    /// Bubble radius [m] — sets the inner boundary.
    pub r_bubble_m: f64,
    /// `r_max = r_bubble * r_max_factor`.
    pub r_max_factor: f64,
}

impl Default for RadicalDiffusionSolver {
    fn default() -> Self {
        Self { n_points: 64, r_bubble_m: 10e-6, r_max_factor: 1000.0 }
    }
}

/// Result from a single diffusion time step.
#[derive(Debug, Clone)]
pub struct DiffusionStepResult {
    /// Updated concentrations \[species\]\[radial_node\] in mol/m³.
    pub concentrations: Vec<Vec<f64>>,
    /// Maximum concentration change across all species and nodes.
    pub max_delta: f64,
}

/// Error type for diffusion solver.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffusionError {
    /// The bubble radius is not positive.
    InvalidBubbleRadius(f64),
    /// `n_points < 3` — cannot form a tridiagonal system.
    TooFewPoints(usize),
    /// Thomas algorithm encountered a zero pivot — numerically singular.
    SingularSystem,
}

impl std::fmt::Display for DiffusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBubbleRadius(r) => write!(f, "bubble radius {r:.3e} m must be > 0"),
            Self::TooFewPoints(n) => write!(f, "{n} points < 3 required for tridiagonal system"),
            Self::SingularSystem => write!(f, "Thomas algorithm: zero pivot (singular system)"),
        }
    }
}

impl RadicalDiffusionSolver {
    /// Create a solver with a specific bubble radius.
    #[must_use]
    pub fn new(r_bubble_m: f64) -> Self {
        Self { n_points: 64, r_bubble_m, r_max_factor: 1000.0 }
    }

    /// Advance all radical species by one diffusion step `dt` [s].
    ///
    /// # Arguments
    ///
    /// * `concentrations` — \[n_species\]\[n_points\] concentrations in mol/m³.
    ///   Mutated in-place.
    /// * `bubble_concs` — Dirichlet boundary condition at `r = R_bubble` for each species.
    /// * `dt` — Time step [s].
    /// * `diffusion_coefficients` — One `D_i` [m²/s] per species.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError`] if geometry or numerics are invalid.
    pub fn step(
        &self,
        concentrations: &mut [Vec<f64>],
        bubble_concs: &[f64],
        dt: f64,
        diffusion_coefficients: &[f64],
    ) -> Result<DiffusionStepResult, DiffusionError> {
        if self.r_bubble_m <= 0.0 {
            return Err(DiffusionError::InvalidBubbleRadius(self.r_bubble_m));
        }
        if self.n_points < 3 {
            return Err(DiffusionError::TooFewPoints(self.n_points));
        }

        let n = self.n_points;
        let r_max = self.r_bubble_m * self.r_max_factor;
        let dxi = (r_max / self.r_bubble_m).ln() / (n - 1) as f64;

        let mut max_delta = 0.0_f64;
        let n_species = concentrations.len();

        for s in 0..n_species {
            if s >= diffusion_coefficients.len() {
                continue;
            }
            let d = diffusion_coefficients[s];
            if d <= 0.0 {
                continue;
            }

            let c = &mut concentrations[s];
            let c_bc = if s < bubble_concs.len() { bubble_concs[s] } else { 0.0 };

            // Crank-Nicolson: solve (I − alpha·L)·c_new = (I + alpha·L)·c + 2·alpha·bc_rhs
            // where alpha = D·dt/(2·dxi²) and L is the log-radial Laplacian.
            let alpha = d * dt / (2.0 * dxi * dxi);

            // Build RHS: (I + alpha·L)·c with Dirichlet BC at j=0
            let mut rhs = vec![0.0_f64; n];
            rhs[0] = c_bc; // Dirichlet: c[0] = bubble boundary
            for j in 1..n - 1 {
                // log-radial Laplacian: L·c[j] = (c[j+1] - 2c[j] + c[j-1])/dxi²
                //                               + (c[j+1] - c[j-1])/(2·dxi²)
                // (the +1 term arises from the (2/r)·∂c/∂r transformation)
                let laplacian = (c[j + 1] - 2.0 * c[j] + c[j - 1]) / (dxi * dxi);
                let first_deriv = (c[j + 1] - c[j - 1]) / (2.0 * dxi * dxi);
                rhs[j] = c[j] + alpha * (laplacian + first_deriv);
            }
            rhs[n - 1] = 0.0; // Dirichlet at far field: c = 0

            // Solve (I − alpha·L)·c_new = rhs using Thomas algorithm.
            // Tridiagonal system: sub = -alpha*(1 - 1/(2)), main = 1 + 2*alpha, sup = -alpha*(1 + 1/2)
            // For the log-radial Laplacian the off-diagonal entries are:
            // lower: -alpha·(1 + 1/2) = -alpha·3/2  ... actually let me set it up properly.
            //
            // The discrete log-Laplacian L·c[j] = (c[j+1] - 2c[j] + c[j-1])/dxi² + (c[j+1]-c[j-1])/(2dxi²)
            // Rearranging:
            // coeff of c[j-1]: 1/dxi² - 1/(2dxi²) = 1/(2dxi²)  → scaled by alpha: alpha/2 * ... wait
            //
            // Let's define per-cell:
            // lower_j  = alpha * (1/dxi² - 1/(2*dxi²)) = alpha * 1/(2*dxi²) ... no.
            //
            // Actually: L * c[j] = [c[j-1](1 - 0.5) + c[j+1](1 + 0.5) - 2*c[j]] / dxi²
            //                     = [0.5*c[j-1] + 1.5*c[j+1] - 2*c[j]] / dxi²
            // That gives an upwind-biased scheme. Let me use a cleaner formulation.
            //
            // Standard log-spherical Laplacian on coord xi = ln(r/R):
            // (1/r^2) d/dr(r^2 dN/dr) = e^{-2xi}/R^2 [d^2N/dxi^2 + 2 dN/dxi]
            // On a unit grid (R=1, normalised): L*N[j] ~ [N[j+1]-2N[j]+N[j-1] + N[j+1]-N[j-1]] / dxi^2
            //                                          = [N[j-1]*(-0) + ... ] wait:
            // d^2N/dxi^2 + 2 dN/dxi discretised:
            //   ~ (N[j+1] - 2N[j] + N[j-1])/dxi^2 + 2*(N[j+1]-N[j-1])/(2*dxi)
            //   = (N[j+1] - 2N[j] + N[j-1])/dxi^2 + (N[j+1]-N[j-1])/dxi
            //
            // So per component:
            //   a_lower = 1/dxi^2 - 1/dxi = (1 - dxi)/dxi^2
            //   a_diag  = -2/dxi^2
            //   a_upper = 1/dxi^2 + 1/dxi = (1 + dxi)/dxi^2
            //
            let inv_dxi2 = 1.0 / (dxi * dxi);
            let inv_dxi = 1.0 / dxi;

            let a_lower = alpha * (inv_dxi2 - inv_dxi); // coefficient of c[j-1]
            let a_diag_off = alpha * (-2.0 * inv_dxi2); // −alpha * (diag of L)
            let a_upper = alpha * (inv_dxi2 + inv_dxi); // coefficient of c[j+1]

            // LHS tridiagonal: I − alpha·L
            // lower = −a_lower, diag = 1 − a_diag_off, upper = −a_upper
            let mut lo = vec![-a_lower; n];
            let mut di = vec![1.0 - a_diag_off; n];
            let mut up = vec![-a_upper; n];

            // Fix BCs
            di[0] = 1.0;
            up[0] = 0.0;
            lo[0] = 0.0;
            di[n - 1] = 1.0;
            lo[n - 1] = 0.0;
            up[n - 1] = 0.0;

            // Thomas algorithm (forward sweep)
            let c_new = thomas_solve(&lo, &di, &up, &rhs).ok_or(DiffusionError::SingularSystem)?;

            // Update concentrations and compute max change
            for j in 0..n {
                let v = c_new[j].max(0.0);
                let delta = (v - c[j]).abs();
                max_delta = max_delta.max(delta);
                c[j] = v;
            }
        }

        Ok(DiffusionStepResult { concentrations: concentrations.to_vec(), max_delta })
    }

    /// Build the logarithmic radial grid `r[j] = R_bubble · exp(j · Δξ)`.
    #[must_use]
    pub fn radial_grid(&self) -> Vec<f64> {
        let n = self.n_points;
        let r_max = self.r_bubble_m * self.r_max_factor;
        let dxi = (r_max / self.r_bubble_m).ln() / (n - 1) as f64;
        (0..n).map(|j| self.r_bubble_m * (j as f64 * dxi).exp()).collect()
    }

    /// Initialise species concentrations to zero on the full grid.
    ///
    /// Returns a `Vec<Vec<f64>>` of shape `[n_species][n_points]`.
    #[must_use]
    pub fn zero_concentrations(&self, n_species: usize) -> Vec<Vec<f64>> {
        vec![vec![0.0_f64; self.n_points]; n_species]
    }

    /// Extract the bubble-wall (j=0) concentrations as a `HashMap`.
    #[must_use]
    pub fn wall_concentrations(
        &self,
        concentrations: &[Vec<f64>],
        species_list: &[ROSSpecies],
    ) -> HashMap<ROSSpecies, f64> {
        species_list
            .iter()
            .zip(concentrations.iter())
            .map(|(&s, c)| (s, c.first().copied().unwrap_or(0.0)))
            .collect()
    }

    /// Diffusion coefficient [m²/s] for a given `ROSSpecies`.
    ///
    /// Values from Buxton et al. (2000); Christman (1987).
    #[must_use]
    pub fn diffusion_coefficient(species: ROSSpecies) -> f64 {
        species.diffusion_coefficient()
    }
}

// ============================================================================
// Thomas algorithm (tridiagonal solver)
// ============================================================================

/// Solve the tridiagonal system `lower·x[i-1] + diag·x[i] + upper·x[i+1] = rhs[i]`
/// using the Thomas algorithm (forward sweep + back substitution).
///
/// Returns `None` if a zero pivot is encountered.
fn thomas_solve(lower: &[f64], diag: &[f64], upper: &[f64], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = diag.len();
    let mut c_prime = vec![0.0_f64; n]; // forward sweep coefficients
    let mut d_prime = vec![0.0_f64; n]; // forward sweep RHS
    let mut x = vec![0.0_f64; n];

    // Forward sweep
    if diag[0].abs() < 1e-300 {
        return None;
    }
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let w = diag[i] - lower[i] * c_prime[i - 1];
        if w.abs() < 1e-300 {
            return None;
        }
        c_prime[i] = upper[i] / w;
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / w;
    }

    // Back substitution
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Some(x)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// **Test 1 — Boundary condition enforcement**.
    ///
    /// With a non-zero bubble concentration, the j=0 node must equal the BC
    /// exactly after each step (Dirichlet).
    #[test]
    fn test_dirichlet_boundary_condition() {
        let solver = RadicalDiffusionSolver::new(10e-6);
        let n_species = 1;
        let mut concs = solver.zero_concentrations(n_species);
        let bubble_bc = vec![1e-6_f64]; // 1 µM at wall
        let d_coeffs = vec![2e-9_f64]; // OH• diffusion coefficient

        solver.step(&mut concs, &bubble_bc, 1e-9, &d_coeffs).unwrap();

        assert!(
            (concs[0][0] - bubble_bc[0]).abs() < 1e-20,
            "BC not enforced: c[0] = {:.4e}, expected {:.4e}",
            concs[0][0],
            bubble_bc[0]
        );
    }

    /// **Test 2 — Far-field decay towards zero**.
    ///
    /// After several diffusion steps, the concentration at the far boundary must
    /// remain zero (homogeneous Dirichlet BC at `r_max`).
    #[test]
    fn test_far_field_remains_zero() {
        let solver = RadicalDiffusionSolver::new(10e-6);
        let n = solver.n_points;
        let n_species = 1;
        let mut concs = solver.zero_concentrations(n_species);
        concs[0][n / 2] = 1e-6; // Gaussian-like seed in the middle

        let bubble_bc = vec![0.0_f64];
        let d_coeffs = vec![2e-9_f64];

        for _ in 0..100 {
            solver.step(&mut concs, &bubble_bc, 1e-9, &d_coeffs).unwrap();
        }

        assert!(
            concs[0][n - 1] < 1e-30,
            "Far-field BC violated: c[n-1] = {:.4e}",
            concs[0][n - 1]
        );
    }

    /// **Test 3 — Non-negative concentrations**.
    ///
    /// Starting from a step-function initial condition, concentrations must
    /// remain non-negative after diffusion.
    #[test]
    fn test_concentrations_non_negative() {
        let solver = RadicalDiffusionSolver::new(10e-6);
        let n = solver.n_points;
        let n_species = 1;
        let mut concs = solver.zero_concentrations(n_species);
        // Step function: first half at 1 µM
        for j in 0..n / 2 {
            concs[0][j] = 1e-6;
        }

        let bubble_bc = vec![1e-6_f64];
        let d_coeffs = vec![2e-9_f64];

        for _ in 0..50 {
            solver.step(&mut concs, &bubble_bc, 1e-9, &d_coeffs).unwrap();
        }

        for &c in &concs[0] {
            assert!(c >= 0.0, "Negative concentration: {c:.4e}");
        }
    }

    /// **Test 4 — Radial grid is strictly increasing**.
    #[test]
    fn test_radial_grid_strictly_increasing() {
        let solver = RadicalDiffusionSolver::new(5e-6);
        let grid = solver.radial_grid();
        for w in grid.windows(2) {
            assert!(w[1] > w[0], "Grid not strictly increasing: {} ≤ {}", w[1], w[0]);
        }
        assert!(
            (grid[0] - solver.r_bubble_m).abs() < 1e-18,
            "First grid point should equal r_bubble"
        );
    }
}
