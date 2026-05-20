use super::linear::thomas_solve;
use super::{DiffusionError, DiffusionStepResult, RadicalDiffusionSolver};

impl RadicalDiffusionSolver {
    /// Advance all radical species by one diffusion step `dt` (s).
    ///
    /// # Theory — Spherical Diffusion on a Logarithmic Grid
    ///
    /// The Smoluchowski equation in spherical coordinates (Storey & Szeri 2000):
    ///
    /// ```text
    /// ∂c/∂t = D · (1/r²) · ∂/∂r(r² · ∂c/∂r)
    /// ```
    ///
    /// Under the change of variable ξ = ln(r/R_bubble) this becomes:
    ///
    /// ```text
    /// ∂c/∂t = (D / r²) · (∂²c/∂ξ² + ∂c/∂ξ)
    ///       = α(r) · (∂²c/∂ξ² + ∂c/∂ξ)
    /// ```
    ///
    /// where `α(r) = D/r²` is the **spatially varying** diffusion coefficient.
    /// On the j-th log-grid node `r_j = R · exp(j · Δξ)`:
    ///
    /// ```text
    /// α_j = D / r_j²
    /// ```
    ///
    /// **Crank-Nicolson half-step coefficients (per node j):**
    ///
    /// ```text
    /// λ_j = α_j · dt / 2
    ///
    /// RHS[j] = c[j] + λ_j · (∂²c/∂ξ² + ∂c/∂ξ)   [explicit half, old c]
    ///
    /// LHS: (1 + 2λ_j/Δξ²) · c_new[j]
    ///      − λ_j · (1/Δξ² − 1/(2Δξ)) · c_new[j−1]
    ///      − λ_j · (1/Δξ² + 1/(2Δξ)) · c_new[j+1] = RHS[j]
    /// ```
    ///
    /// where the `1/(2Δξ)` factor comes from the central-difference approximation
    /// of `∂c/∂ξ` (compare with the previous incorrect `1/Δξ` = twice too large).
    ///
    /// # Arguments
    ///
    /// * `concentrations` - `[n_species][n_points]` concentrations in mol/m^3.
    ///   Mutated in place.
    /// * `bubble_concs` - Dirichlet boundary condition at `r = R_bubble` for each species.
    /// * `dt` - Time step (s).
    /// * `diffusion_coefficients` - One `D_i` [m^2/s] per species.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError`] when geometry or numerics are invalid.
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
        let inv_dxi = 1.0 / dxi;
        let inv_dxi2 = inv_dxi * inv_dxi;

        // Pre-compute per-node radii: r_j = R_bubble * exp(j * dxi)
        let r_nodes: Vec<f64> = (0..n)
            .map(|j| self.r_bubble_m * (j as f64 * dxi).exp())
            .collect();

        let mut max_delta = 0.0_f64;

        for (species_index, concentration) in concentrations.iter_mut().enumerate() {
            let Some(&diffusion_coefficient) = diffusion_coefficients.get(species_index) else {
                continue;
            };
            if diffusion_coefficient <= 0.0 {
                continue;
            }

            let wall_concentration = bubble_concs.get(species_index).copied().unwrap_or(0.0);

            // λ_j = D * dt / (2 * r_j²)  — spatially varying CN half-step coefficient
            let lambda: Vec<f64> = r_nodes
                .iter()
                .map(|&r| diffusion_coefficient * dt / (2.0 * r * r))
                .collect();

            let rhs = build_rhs(concentration, wall_concentration, &lambda, inv_dxi, inv_dxi2, n);
            let (lower, diagonal, upper) =
                build_lhs(&lambda, inv_dxi, inv_dxi2, n);

            let solved = thomas_solve(&lower, &diagonal, &upper, &rhs)
                .ok_or(DiffusionError::SingularSystem)?;

            for (current, next) in concentration.iter_mut().zip(solved) {
                let projected = next.max(0.0);
                max_delta = max_delta.max((projected - *current).abs());
                *current = projected;
            }
        }

        Ok(DiffusionStepResult {
            concentrations: concentrations.to_vec(),
            max_delta,
        })
    }
}

fn build_rhs(
    concentration: &[f64],
    wall_concentration: f64,
    lambda: &[f64],
    inv_dxi: f64,
    inv_dxi2: f64,
    n: usize,
) -> Vec<f64> {
    let mut rhs = vec![0.0_f64; n];
    rhs[0] = wall_concentration;

    for j in 1..n - 1 {
        // ∂²c/∂ξ² via central difference
        let laplacian = (concentration[j + 1] - 2.0 * concentration[j] + concentration[j - 1])
            * inv_dxi2;
        // ∂c/∂ξ via central difference (coefficient 1/(2Δξ) = 0.5 * inv_dxi)
        let first_derivative = (concentration[j + 1] - concentration[j - 1]) * 0.5 * inv_dxi;
        // Explicit half: c[j] + λ_j * (∂²c/∂ξ² + ∂c/∂ξ)
        rhs[j] = lambda[j].mul_add(laplacian + first_derivative, concentration[j]);
    }

    rhs[n - 1] = 0.0;
    rhs
}

/// Build the tridiagonal CN-implicit matrix for the log-grid spherical diffusion.
///
/// The LHS operator is `(I − λ_j · L_ξ)` where:
///
/// ```text
/// L_ξ c[j] = c[j-1]·(1/Δξ² − 1/(2Δξ))
///            − c[j]·(2/Δξ²)
///            + c[j+1]·(1/Δξ² + 1/(2Δξ))
/// ```
///
/// Rows 0 and n−1 are set to identity (Dirichlet boundary conditions).
fn build_lhs(lambda: &[f64], inv_dxi: f64, inv_dxi2: f64, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut lower = vec![0.0_f64; n];
    let mut diagonal = vec![1.0_f64; n];
    let mut upper = vec![0.0_f64; n];

    // Interior nodes: per-point λ_j coefficient
    for j in 1..n - 1 {
        let lam = lambda[j];
        // c_new[j-1] coefficient: −λ_j · (1/Δξ² − 1/(2Δξ))
        lower[j] = -lam * (inv_dxi2 - 0.5 * inv_dxi);
        // c_new[j] coefficient: 1 + λ_j · (2/Δξ²)
        diagonal[j] = 1.0 + lam * (2.0 * inv_dxi2);
        // c_new[j+1] coefficient: −λ_j · (1/Δξ² + 1/(2Δξ))
        upper[j] = -lam * (inv_dxi2 + 0.5 * inv_dxi);
    }

    // Dirichlet boundaries: rows 0 and n-1 are identity
    // (already set by initialization above)

    (lower, diagonal, upper)
}

