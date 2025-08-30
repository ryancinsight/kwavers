//! Bubble field management
//!
//! Manages collections of bubbles in the simulation domain

use super::adaptive_integration::integrate_bubble_dynamics_adaptive;
use super::bubble_state::{BubbleParameters, BubbleState};
use super::rayleigh_plesset::KellerMiksisModel;
use ndarray::Array3;
use rand::prelude::*;
use rand_distr::{LogNormal, Normal, Uniform};
use std::collections::HashMap;

/// Single bubble or bubble cloud field
#[derive(Debug))]
pub struct BubbleField {
    /// Bubble states indexed by grid position
    pub bubbles: HashMap<(usize, usize, usize), BubbleState>,
    /// Solver for bubble dynamics
    solver: KellerMiksisModel,
    /// Default bubble parameters
    bubble_parameters: BubbleParameters,
    /// Grid dimensions
    grid_shape: (usize, usize, usize),
    /// Time history for selected bubbles
    pub time_history: Vec<f64>,
    pub radius_history: Vec<Vec<f64>>,
    pub temperature_history: Vec<Vec<f64>>,
}

impl BubbleField {
    /// Create new bubble field
    pub fn new(grid_shape: (usize, usize, usize), params: BubbleParameters) -> Self {
        Self {
            bubbles: HashMap::new(),
            solver: KellerMiksisModel::new(params.clone()),
            bubble_parameters: params,
            grid_shape,
            time_history: Vec::new(),
            radius_history: Vec::new(),
            temperature_history: Vec::new(),
        }
    }

    /// Add a single bubble at grid position
    pub fn add_bubble(&mut self, i: usize, j: usize, k: usize, state: BubbleState) {
        self.bubbles.insert((i, j, k), state);
    }

    /// Add bubble at center of grid
    pub fn add_center_bubble(&mut self, params: &BubbleParameters) {
        let center = (
            self.grid_shape.0 / 2,
            self.grid_shape.1 / 2,
            self.grid_shape.2 / 2,
        );
        let state = BubbleState::new(params);
        self.add_bubble(center.0, center.1, center.2, state);
    }

    /// Update all bubbles for one time step
    pub fn update(
        &mut self,
        pressure_field: &Array3<f64>,
        dp_dt_field: &Array3<f64>,
        dt: f64,
        t: f64,
    ) {
        // Update each bubble
        for ((i, j, k), state) in self.bubbles.iter_mut() {
            let p_acoustic = pressure_field[[*i, *j, *k];
            let dp_dt = dp_dt_field[[*i, *j, *k];

            // Use adaptive integration (no Mutex needed anymore)
            if let Err(e) =
                integrate_bubble_dynamics_adaptive(&self.solver, state, p_acoustic, dp_dt, dt, t)
            {
                eprintln!(
                    "Bubble dynamics integration failed at position ({}, {}, {}): {:?}",
                    i, j, k, e
                );
            }
        }

        // Record history for tracking
        self.record_history(t);
    }

    /// Record time history of bubble states
    fn record_history(&mut self, t: f64) {
        self.time_history.push(t);

        // Initialize history vectors if needed
        if self.radius_history.is_empty() {
            for _ in 0..self.bubbles.len() {
                self.radius_history.push(Vec::new());
                self.temperature_history.push(Vec::new());
            }
        }

        // Record each bubble's state
        for (idx, (_, state)) in self.bubbles.iter().enumerate() {
            self.radius_history[idx].push(state.radius);
            self.temperature_history[idx].push(state.temperature);
        }
    }

    /// Get bubble state fields for physics modules
    pub fn get_state_fields(&self) -> BubbleStateFields {
        let shape = self.grid_shape;
        let mut fields = BubbleStateFields::new(shape);

        for ((i, j, k), state) in &self.bubbles {
            fields.radius[[*i, *j, *k] = state.radius;
            fields.temperature[[*i, *j, *k] = state.temperature;
            fields.pressure[[*i, *j, *k] = state.pressure_internal;
            fields.velocity[[*i, *j, *k] = state.wall_velocity;
            fields.is_collapsing[[*i, *j, *k] = state.is_collapsing as i32 as f64;
            fields.compression_ratio[[*i, *j, *k] = state.compression_ratio;
        }

        fields
    }

    /// Get statistics about bubble field
    pub fn get_statistics(&self) -> BubbleFieldStats {
        let mut stats = BubbleFieldStats::default();

        for state in self.bubbles.values() {
            stats.total_bubbles += 1;
            if state.is_collapsing {
                stats.collapsing_bubbles += 1;
            }
            stats.max_temperature = stats.max_temperature.max(state.temperature);
            stats.max_compression = stats.max_compression.max(state.compression_ratio);
            stats.total_collapses += state.collapse_count;
        }

        stats
    }
}

/// Bubble state fields for interfacing with physics modules
#[derive(Debug))]
pub struct BubbleStateFields {
    pub radius: Array3<f64>,
    pub temperature: Array3<f64>,
    pub pressure: Array3<f64>,
    pub velocity: Array3<f64>,
    pub is_collapsing: Array3<f64>,
    pub compression_ratio: Array3<f64>,
}

impl BubbleStateFields {
    pub fn new(shape: (usize, usize, usize)) -> Self {
        Self {
            radius: Array3::zeros(shape),
            temperature: Array3::from_elem(shape, 293.15),
            pressure: Array3::from_elem(shape, 101325.0),
            velocity: Array3::zeros(shape),
            is_collapsing: Array3::zeros(shape),
            compression_ratio: Array3::from_elem(shape, 1.0),
        }
    }
}

/// Statistics about bubble field
#[derive(Debug, Default)]
pub struct BubbleFieldStats {
    pub total_bubbles: usize,
    pub collapsing_bubbles: usize,
    pub max_temperature: f64,
    pub max_compression: f64,
    pub total_collapses: u32,
}

/// Bubble cloud with size distribution
#[derive(Debug))]
pub struct BubbleCloud {
    /// Base bubble field
    pub field: BubbleField,
    /// Size distribution parameters
    pub size_distribution: SizeDistribution,
    /// Spatial distribution
    pub spatial_distribution: SpatialDistribution,
}

/// Size distribution types
#[derive(Debug, Clone))]
pub enum SizeDistribution {
    Uniform { min: f64, max: f64 },
    LogNormal { mean: f64, std_dev: f64 },
    PowerLaw { min: f64, max: f64, exponent: f64 },
}

/// Spatial distribution types
#[derive(Debug, Clone))]
pub enum SpatialDistribution {
    Uniform,
    Gaussian {
        center: (f64, f64, f64),
        std_dev: f64,
    },
    Cluster {
        centers: Vec<(f64, f64, f64)>,
        radius: f64,
    },
}

impl BubbleCloud {
    /// Create new bubble cloud
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
                let u: f64 = rng.gen();
                let alpha = exponent + 1.0;

                (u * (max.powf(alpha) - min.powf(alpha)) + min.powf(alpha)).powf(1.0 / alpha)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bubble_field_creation() {
        let params = BubbleParameters::default();
        let mut field = BubbleField::new((10, 10, 10), params.clone());

        field.add_center_bubble(&params);
        assert_eq!(field.bubbles.len(), 1);
        assert!(field.bubbles.contains_key(&(5, 5, 5)));
    }

    #[test]
    fn test_bubble_cloud_generation() {
        let params = BubbleParameters::default();
        let size_dist = SizeDistribution::Uniform {
            min: 1e-6,
            max: 10e-6,
        };
        let spatial_dist = SpatialDistribution::Uniform;

        let mut cloud = BubbleCloud::new((20, 20, 20), params, size_dist, spatial_dist);
        cloud.generate(1e12, (1e-3, 1e-3, 1e-3)); // Higher density and larger grid spacing

        assert!(cloud.field.bubbles.len() > 0);
    }
}
