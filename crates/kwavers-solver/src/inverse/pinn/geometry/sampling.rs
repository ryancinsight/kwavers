use leto::Array2;

use kwavers_grid::geometry::{GeometricDomain, PointLocation};

/// Sampling strategy for PINN collocation points
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollocationSamplingStrategy {
    /// Uniform random sampling (baseline)
    Uniform,
    /// Latin hypercube sampling (better space-filling)
    LatinHypercube,
    /// Sobol quasi-random sequence (low-discrepancy)
    Sobol,
    /// Adaptive refinement based on residual magnitude
    AdaptiveRefinement,
}

/// Collocation point sampler for PINN training
///
/// Generates interior, boundary, and interface points according to a
/// specified sampling strategy.
pub struct CollocationSampler {
    pub domain: Box<dyn GeometricDomain>,
    pub strategy: CollocationSamplingStrategy,
    pub seed: Option<u64>,
}

impl std::fmt::Debug for CollocationSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollocationSampler")
            .field("domain", &"<dyn GeometricDomain>")
            .field("strategy", &self.strategy)
            .field("seed", &self.seed)
            .finish()
    }
}

impl CollocationSampler {
    #[must_use]
    pub fn new(
        domain: Box<dyn GeometricDomain>,
        strategy: CollocationSamplingStrategy,
        seed: Option<u64>,
    ) -> Self {
        Self {
            domain,
            strategy,
            seed,
        }
    }

    #[must_use]
    pub fn sample_interior(&self, n_points: usize) -> Array2<f64> {
        match self.strategy {
            CollocationSamplingStrategy::Uniform => {
                self.domain.sample_interior(n_points, self.seed)
            }
            CollocationSamplingStrategy::LatinHypercube => {
                self.latin_hypercube_sample(n_points, false)
            }
            CollocationSamplingStrategy::Sobol => self.sobol_sample(n_points, false),
            CollocationSamplingStrategy::AdaptiveRefinement => {
                self.domain.sample_interior(n_points, self.seed)
            }
        }
    }

    #[must_use]
    pub fn sample_boundary(&self, n_points: usize) -> Array2<f64> {
        match self.strategy {
            CollocationSamplingStrategy::Uniform => {
                self.domain.sample_boundary(n_points, self.seed)
            }
            CollocationSamplingStrategy::LatinHypercube => {
                self.latin_hypercube_sample(n_points, true)
            }
            CollocationSamplingStrategy::Sobol => self.sobol_sample(n_points, true),
            CollocationSamplingStrategy::AdaptiveRefinement => {
                self.domain.sample_boundary(n_points, self.seed)
            }
        }
    }

    fn latin_hypercube_sample(&self, n_points: usize, boundary: bool) -> Array2<f64> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = if let Some(s) = self.seed {
            ChaCha8Rng::seed_from_u64(s)
        } else {
            ChaCha8Rng::from_entropy()
        };

        let bbox = self.domain.bounding_box();
        let dim = (bbox.len()) / 2;

        let mut points = Array2::zeros([n_points, dim]);

        for d in 0..dim {
            let mut perm: Vec<usize> = (0..n_points).collect();
            for i in (1..n_points).rev() {
                let j = rng.gen_range(0..=i);
                perm.swap(i, j);
            }

            for (i, &p) in perm.iter().enumerate() {
                let u = (p as f64 + rng.gen::<f64>()) / n_points as f64;
                points[[i, d]] = u.mul_add(bbox[2 * d + 1] - bbox[2 * d], bbox[2 * d]);
            }
        }

        let mut valid_points = Vec::new();
        let tolerance = 1e-8;
        for i in 0..n_points {
            let mut point = vec![0.0; dim];
            for d in 0..dim {
                point[d] = points[[i, d]];
            }
            let loc = self.domain.classify_point(&point, tolerance);
            let include = (boundary && loc == PointLocation::Boundary)
                || (!boundary && loc == PointLocation::Interior);
            if include {
                valid_points.push(point);
            }
        }

        let n_valid = (valid_points.len()).min(n_points);
        let mut result = Array2::zeros([n_valid, dim]);
        for (i, point) in valid_points.iter().take(n_valid).enumerate() {
            for (j, &coord) in point.iter().enumerate() {
                result[[i, j]] = coord;
            }
        }

        result
    }

    fn sobol_sample(&self, n_points: usize, boundary: bool) -> Array2<f64> {
        if boundary {
            return self.domain.sample_boundary(n_points, self.seed);
        }

        let bbox = self.domain.bounding_box();
        let dim = (bbox.len()) / 2;

        let unit_points = sobol_unit_hypercube_points(n_points, dim, self.seed);
        let mut points = Array2::zeros([n_points, dim]);
        for (i, u) in unit_points.iter().enumerate() {
            for d in 0..dim {
                points[[i, d]] = u[d].mul_add(bbox[2 * d + 1] - bbox[2 * d], bbox[2 * d]);
            }
        }

        points
    }
}

/// Sobol unit hypercube points.
/// # Panics
/// - Panics if an internal precondition is violated.
///
pub(super) fn sobol_unit_hypercube_points(
    n_points: usize,
    dim: usize,
    seed: Option<u64>,
) -> Vec<Vec<f64>> {
    assert!(
        (1..=3).contains(&dim),
        "Sobol sequence is supported only for 1..=3 dimensions"
    );

    let skip = seed.map_or(0usize, |s| (s % 1024) as usize);
    let target = n_points + skip;

    let direction_numbers = sobol_direction_numbers(dim);
    let mut x: Vec<u32> = vec![0; dim];

    let mut result = Vec::with_capacity(n_points);
    for i in 1..=target {
        let c = (i as u32).trailing_zeros() as usize;
        for d in 0..dim {
            x[d] ^= direction_numbers[d][c];
        }

        if i > skip {
            let point = x
                .iter()
                .map(|&v| (v as f64) / (u32::MAX as f64 + 1.0))
                .collect::<Vec<f64>>();
            result.push(point);
        }
    }

    debug_assert_eq!((result.len()), n_points);
    result
}

fn sobol_direction_numbers(dim: usize) -> Vec<[u32; 32]> {
    const MAX_BITS: usize = 32;
    let mut dirs: Vec<[u32; MAX_BITS]> = Vec::with_capacity(dim);

    let mut v1 = [0u32; MAX_BITS];
    for (i, v) in v1.iter_mut().enumerate() {
        *v = 1u32 << (31 - i);
    }
    dirs.push(v1);

    if dim >= 2 {
        dirs.push(sobol_direction_numbers_from_params(1, 0, &[1]));
    }
    if dim >= 3 {
        dirs.push(sobol_direction_numbers_from_params(2, 1, &[1, 3]));
    }

    dirs
}

fn sobol_direction_numbers_from_params(s: usize, a: u32, m: &[u32]) -> [u32; 32] {
    const MAX_BITS: usize = 32;
    assert_eq!((m.len()), s);
    assert!((1..MAX_BITS).contains(&s));

    let mut v = [0u32; MAX_BITS];
    for i in 0..s {
        v[i] = m[i] << (31 - i);
    }

    for i in s..MAX_BITS {
        let mut vi = v[i - s] ^ (v[i - s] >> s);
        for k in 1..s {
            let bit = (a >> (s - 1 - k)) & 1;
            if bit == 1 {
                vi ^= v[i - k];
            }
        }
        v[i] = vi;
    }

    v
}
