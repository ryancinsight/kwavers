use ndarray::{Array1, Array2};

use super::sampling::CollocationSampler;

/// Adaptive mesh refinement for PINN training
///
/// Refines collocation point distribution based on PDE residual magnitude.
/// Regions with high residuals get more points in subsequent training epochs.
pub struct AdaptiveRefinement {
    sampler: CollocationSampler,
    points: Array2<f64>,
    residuals: Array1<f64>,
    threshold: f64,
}

impl std::fmt::Debug for AdaptiveRefinement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveRefinement")
            .field("sampler", &"CollocationSampler")
            .field(
                "points",
                &format!("{}x{}", self.points.nrows(), self.points.ncols()),
            )
            .field("residuals", &format!("{} values", self.residuals.len()))
            .field("threshold", &self.threshold)
            .finish()
    }
}

impl AdaptiveRefinement {
    #[must_use]
    pub fn new(sampler: CollocationSampler, initial_points: Array2<f64>, threshold: f64) -> Self {
        let n_points = initial_points.nrows();
        Self {
            sampler,
            points: initial_points,
            residuals: Array1::zeros(n_points),
            threshold,
        }
    }

    /// Update residuals.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn update_residuals(&mut self, residuals: Array1<f64>) {
        assert_eq!(
            residuals.len(),
            self.points.nrows(),
            "Residual count mismatch"
        );
        self.residuals = residuals;
    }

    /// Refine mesh by adding points near high-residual regions
    pub fn refine(&mut self, refinement_factor: f64) -> Array2<f64> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = if let Some(s) = self.sampler.seed {
            ChaCha8Rng::seed_from_u64(s)
        } else {
            ChaCha8Rng::from_entropy()
        };

        let mut refined_points = self.points.clone().to_owned();
        let dim = self.points.ncols();

        for (i, &residual) in self.residuals.iter().enumerate() {
            if residual > self.threshold {
                let n_new = (refinement_factor * residual / self.threshold).ceil() as usize;

                for _ in 0..n_new {
                    let mut new_point = Array1::zeros(dim);

                    let perturbation_scale = 0.1;
                    for d in 0..dim {
                        let delta = rng.gen_range(-perturbation_scale..perturbation_scale);
                        new_point[d] = self.points[[i, d]] + delta;
                    }

                    let mut temp = Array2::zeros((refined_points.nrows() + 1, dim));
                    for row in 0..refined_points.nrows() {
                        for col in 0..dim {
                            temp[[row, col]] = refined_points[[row, col]];
                        }
                    }
                    for col in 0..dim {
                        temp[[refined_points.nrows(), col]] = new_point[col];
                    }
                    refined_points = temp;
                }
            }
        }

        self.points = refined_points.clone();
        self.residuals = Array1::zeros(self.points.nrows());

        refined_points
    }
}
