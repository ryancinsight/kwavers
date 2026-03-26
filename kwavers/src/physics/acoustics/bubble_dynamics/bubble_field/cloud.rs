use rand::prelude::*;
use rand_distr::{LogNormal, Normal, Uniform};

use super::super::bubble_state::{BubbleParameters, BubbleState};
use super::core::BubbleField;
use super::distributions::{SizeDistribution, SpatialDistribution};

/// Bubble cloud with size distribution
#[derive(Debug)]
pub struct BubbleCloud {
    /// Base bubble field
    pub field: BubbleField,
    /// Size distribution parameters
    pub size_distribution: SizeDistribution,
    /// Spatial distribution
    pub spatial_distribution: SpatialDistribution,
}

impl BubbleCloud {
    /// Create new bubble cloud
    #[must_use]
    pub fn new(
        grid_shape: (usize, usize, usize),
        params: BubbleParameters,
        size_dist: SizeDistribution,
        spatial_dist: SpatialDistribution,
    ) -> Self {
        Self {
            field: BubbleField::new(grid_shape, params),
            size_distribution: size_dist,
            spatial_distribution: spatial_dist,
        }
    }

    /// Generate bubble cloud with specified density
    pub fn generate(&mut self, bubble_density: f64, grid_spacing: (f64, f64, f64)) {
        let mut rng = thread_rng();

        let volume = grid_spacing.0
            * grid_spacing.1
            * grid_spacing.2
            * (self.field.grid_shape.0 * self.field.grid_shape.1 * self.field.grid_shape.2) as f64;
        let n_bubbles = (bubble_density * volume) as usize;

        for _ in 0..n_bubbles {
            // Generate position
            let (i, j, k) = self.generate_position(&mut rng);

            // Generate size
            let radius = self.generate_radius(&mut rng);

            // Create bubble with custom radius
            let mut params = self.field.bubble_parameters.clone();
            params.r0 = radius;
            let state = BubbleState::new(&params);

            self.field.add_bubble(i, j, k, state);
        }
    }

    fn generate_position(&self, rng: &mut impl Rng) -> (usize, usize, usize) {
        let shape = self.field.grid_shape;

        match &self.spatial_distribution {
            SpatialDistribution::Uniform => (
                rng.gen_range(0..shape.0),
                rng.gen_range(0..shape.1),
                rng.gen_range(0..shape.2),
            ),
            SpatialDistribution::Gaussian { center, std_dev } => {
                // Generate Gaussian-distributed position
                let normal = Normal::new(0.0, *std_dev).unwrap();
                let dx: f64 = rng.sample(normal);
                let dy: f64 = rng.sample(normal);
                let dz: f64 = rng.sample(normal);

                let i = ((center.0 + dx) * shape.0 as f64).round() as usize;
                let j = ((center.1 + dy) * shape.1 as f64).round() as usize;
                let k = ((center.2 + dz) * shape.2 as f64).round() as usize;

                (
                    i.clamp(0, shape.0 - 1),
                    j.clamp(0, shape.1 - 1),
                    k.clamp(0, shape.2 - 1),
                )
            }
            SpatialDistribution::Cluster { centers, radius } => {
                // Choose random cluster center
                let center = centers.choose(rng).unwrap();

                // Generate position within cluster
                let uniform = Uniform::new(-*radius, *radius);
                let dx: f64 = rng.sample(uniform);
                let dy: f64 = rng.sample(uniform);
                let dz: f64 = rng.sample(uniform);

                let i = ((center.0 + dx) * shape.0 as f64).round() as usize;
                let j = ((center.1 + dy) * shape.1 as f64).round() as usize;
                let k = ((center.2 + dz) * shape.2 as f64).round() as usize;

                (
                    i.clamp(0, shape.0 - 1),
                    j.clamp(0, shape.1 - 1),
                    k.clamp(0, shape.2 - 1),
                )
            }
        }
    }

    fn generate_radius(&self, rng: &mut impl Rng) -> f64 {
        match &self.size_distribution {
            SizeDistribution::Uniform { min, max } => rng.gen_range(*min..*max),
            SizeDistribution::LogNormal { mean, std_dev } => {
                let normal = LogNormal::new(mean.ln(), *std_dev / *mean).unwrap();
                rng.sample(normal)
            }
            SizeDistribution::PowerLaw { min, max, exponent } => {
                let u: f64 = rng.r#gen();
                let alpha = exponent + 1.0;

                (u * (max.powf(alpha) - min.powf(alpha)) + min.powf(alpha)).powf(1.0 / alpha)
            }
        }
    }
}
