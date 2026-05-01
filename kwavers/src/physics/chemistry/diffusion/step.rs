use super::linear::thomas_solve;
use super::{DiffusionError, DiffusionStepResult, RadicalDiffusionSolver};

impl RadicalDiffusionSolver {
    /// Advance all radical species by one diffusion step `dt` [s].
    ///
    /// # Arguments
    ///
    /// * `concentrations` - `[n_species][n_points]` concentrations in mol/m^3.
    ///   Mutated in place.
    /// * `bubble_concs` - Dirichlet boundary condition at `r = R_bubble` for each species.
    /// * `dt` - Time step [s].
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

        let mut max_delta = 0.0_f64;

        for (species_index, concentration) in concentrations.iter_mut().enumerate() {
            let Some(&diffusion_coefficient) = diffusion_coefficients.get(species_index) else {
                continue;
            };
            if diffusion_coefficient <= 0.0 {
                continue;
            }

            let wall_concentration = bubble_concs.get(species_index).copied().unwrap_or(0.0);
            let alpha = diffusion_coefficient * dt / (2.0 * dxi * dxi);

            let rhs = build_rhs(concentration, wall_concentration, alpha, dxi, n);
            let (lower, diagonal, upper) = build_lhs(alpha, inv_dxi, inv_dxi2, n);

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
    alpha: f64,
    dxi: f64,
    n: usize,
) -> Vec<f64> {
    let mut rhs = vec![0.0_f64; n];
    rhs[0] = wall_concentration;

    let inv_dxi2 = 1.0 / (dxi * dxi);
    for j in 1..n - 1 {
        let laplacian =
            (concentration[j + 1] - 2.0 * concentration[j] + concentration[j - 1]) * inv_dxi2;
        let first_derivative = (concentration[j + 1] - concentration[j - 1]) / (2.0 * dxi * dxi);
        rhs[j] = concentration[j] + alpha * (laplacian + first_derivative);
    }

    rhs[n - 1] = 0.0;
    rhs
}

fn build_lhs(alpha: f64, inv_dxi: f64, inv_dxi2: f64, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let lower_coefficient = alpha * (inv_dxi2 - inv_dxi);
    let diagonal_coefficient = alpha * (-2.0 * inv_dxi2);
    let upper_coefficient = alpha * (inv_dxi2 + inv_dxi);

    let mut lower = vec![-lower_coefficient; n];
    let mut diagonal = vec![1.0 - diagonal_coefficient; n];
    let mut upper = vec![-upper_coefficient; n];

    diagonal[0] = 1.0;
    upper[0] = 0.0;
    lower[0] = 0.0;
    diagonal[n - 1] = 1.0;
    lower[n - 1] = 0.0;
    upper[n - 1] = 0.0;

    (lower, diagonal, upper)
}
