//! PINN-Specific Geometry Extensions
//!
//! This module extends the domain-layer geometric abstractions with PINN-specific
//! functionality for collocation point sampling, interface condition handling,
//! and adaptive refinement strategies.
//!
//! # Design Philosophy
//!
//! This module BUILDS ON TOP OF `domain::geometry` rather than reimplementing it.
//! The domain layer provides:
//! - Basic geometric primitives (RectangularDomain, SphericalDomain)
//! - Point-in-domain tests and normal computation
//! - Uniform sampling of interior and boundary points
//!
//! This module adds PINN-specific concerns:
//! - Stratified sampling for better training convergence
//! - Interface condition enforcement for multi-region domains
//! - Adaptive mesh refinement based on loss gradients
//! - Boundary normal computation for Neumann conditions
//!
//! # Mathematical Foundation
//!
//! PINNs require collocation points at three locations:
//!
//! 1. **Interior points** (x, y) ∈ Ω: Enforce PDE residual
//!    ```text
//!    L_interior = || ρ ∂²u/∂t² - ∇·σ - f ||²
//!    ```
//!
//! 2. **Boundary points** (x, y) ∈ ∂Ω: Enforce boundary conditions
//!    ```text
//!    L_boundary = || BC(u) - g ||²
//!    ```
//!
//! 3. **Interface points** (x, y) ∈ Γ: Enforce continuity (multi-region only)
//!    ```text
//!    L_interface = || u₁ - u₂ ||² + || σ₁·n - σ₂·n ||²
//!    ```
//!
//! # References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks" - JCP 378:686-707
//! - Jagtap et al. (2020): "Conservative physics-informed neural networks on
//!   discrete domains for conservation laws" - CMAME 365:113028
//! - Karniadakis et al. (2021): "Physics-informed machine learning" - Nature
//!   Reviews Physics 3:422-440

use crate::domain::geometry::{GeometricDomain, PointLocation};
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Interface condition specification for multi-region PINN training
///
/// Defines how the neural network solution should behave at internal boundaries
/// between regions with different material properties.
#[derive(Clone)]
pub enum InterfaceCondition {
    /// Continuity of displacement and traction (standard elastic interface)
    ///
    /// Enforces:
    /// - u₁ = u₂ (displacement continuity)
    /// - σ₁·n = σ₂·n (traction continuity)
    ElasticContinuity,

    /// Welded contact (same as ElasticContinuity, but named for clarity)
    WeldedContact,

    /// Sliding contact (tangential slip allowed, normal stress continuous)
    ///
    /// Enforces:
    /// - u₁·n = u₂·n (normal displacement continuity)
    /// - σ₁·n = σ₂·n (normal stress continuity)
    /// - (σ₁·n)×n = 0 (zero tangential traction)
    SlidingContact,

    /// Free boundary (special case: interface with vacuum/air)
    ///
    /// Enforces:
    /// - σ·n = 0 (stress-free condition)
    FreeBoundary,

    /// Acoustic-elastic interface (fluid-solid coupling)
    ///
    /// Enforces:
    /// - u_solid·n = v_fluid·n (normal velocity continuity)
    /// - σ_solid·n = -p_fluid·n (pressure-stress balance)
    AcousticElastic { fluid_density: f64 },

    /// Custom interface condition with user-defined residual function
    ///
    /// The function receives:
    /// - displacement on each side: u₁, u₂ [m]
    /// - stress tensors: σ₁, σ₂ [Pa]
    /// - unit normal vector: n̂ (pointing from region 1 to region 2)
    ///
    /// Returns: residual that should be zero when condition is satisfied
    Custom {
        residual_fn: Arc<
            dyn Fn(&[f64], &[f64], &[[f64; 2]; 2], &[[f64; 2]; 2], &[f64]) -> f64 + Send + Sync,
        >,
    },
}

impl std::fmt::Debug for InterfaceCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ElasticContinuity => write!(f, "ElasticContinuity"),
            Self::WeldedContact => write!(f, "WeldedContact"),
            Self::SlidingContact => write!(f, "SlidingContact"),
            Self::FreeBoundary => write!(f, "FreeBoundary"),
            Self::AcousticElastic { fluid_density } => {
                write!(f, "AcousticElastic(ρ={})", fluid_density)
            }
            Self::Custom { .. } => write!(f, "Custom"),
        }
    }
}

/// Multi-region domain for heterogeneous media
///
/// Represents a domain composed of multiple subdomains with different material
/// properties. The PINN must satisfy interface conditions at region boundaries.
pub struct MultiRegionDomain {
    /// Subdomains (each is a geometric region)
    pub regions: Vec<Box<dyn GeometricDomain>>,
    /// Material property IDs for each region (index into material database)
    pub material_ids: Vec<usize>,
    /// Interface conditions between adjacent regions
    /// interfaces[i] specifies condition between regions i and i+1
    pub interfaces: Vec<InterfaceCondition>,
}

impl std::fmt::Debug for MultiRegionDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiRegionDomain")
            .field("regions", &format!("{} regions", self.regions.len()))
            .field("material_ids", &self.material_ids)
            .field("interfaces", &self.interfaces)
            .finish()
    }
}

impl MultiRegionDomain {
    /// Create a new multi-region domain
    pub fn new(
        regions: Vec<Box<dyn GeometricDomain>>,
        material_ids: Vec<usize>,
        interfaces: Vec<InterfaceCondition>,
    ) -> Self {
        assert_eq!(
            regions.len(),
            material_ids.len(),
            "Each region must have a material ID"
        );
        assert_eq!(
            interfaces.len(),
            regions.len() - 1,
            "Need N-1 interfaces for N regions"
        );
        Self {
            regions,
            material_ids,
            interfaces,
        }
    }

    /// Find which region contains a given point
    ///
    /// Returns region index and interior/boundary classification.
    /// If point is on interface, returns the lower-index region.
    pub fn locate_point(&self, point: &[f64], tolerance: f64) -> Option<(usize, PointLocation)> {
        for (i, region) in self.regions.iter().enumerate() {
            let loc = region.classify_point(point, tolerance);
            if loc != PointLocation::Exterior {
                return Some((i, loc));
            }
        }
        None
    }

    /// Sample interface points between regions
    ///
    /// Returns points on the boundaries between adjacent regions where
    /// interface conditions must be enforced.
    pub fn sample_interface_points(
        &self,
        n_points_per_interface: usize,
        seed: Option<u64>,
    ) -> Array2<f64> {
        let mut all_points = Vec::new();

        for i in 0..self.regions.len() - 1 {
            let boundary_i = self.regions[i].sample_boundary(n_points_per_interface * 2, seed);

            let tolerance = 1e-8;
            for row_idx in 0..boundary_i.nrows() {
                let point = boundary_i.row(row_idx);
                let point_slice: Vec<f64> = point.iter().cloned().collect();

                if self.regions[i + 1].classify_point(&point_slice, tolerance)
                    == PointLocation::Boundary
                {
                    all_points.push(point_slice);
                    if all_points.len() >= n_points_per_interface {
                        break;
                    }
                }
            }
        }

        let n_found = all_points.len();
        let dim = if n_found > 0 { all_points[0].len() } else { 2 };
        let mut result = Array2::zeros((n_found, dim));
        for (i, point) in all_points.iter().enumerate() {
            for (j, &coord) in point.iter().enumerate() {
                result[[i, j]] = coord;
            }
        }

        result
    }
}

/// Sampling strategy for PINN collocation points
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
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
    /// Geometric domain
    pub domain: Box<dyn GeometricDomain>,
    /// Sampling strategy
    pub strategy: SamplingStrategy,
    /// Random seed for reproducibility
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
    /// Create a new sampler
    pub fn new(
        domain: Box<dyn GeometricDomain>,
        strategy: SamplingStrategy,
        seed: Option<u64>,
    ) -> Self {
        Self {
            domain,
            strategy,
            seed,
        }
    }

    /// Sample interior collocation points
    pub fn sample_interior(&self, n_points: usize) -> Array2<f64> {
        match self.strategy {
            SamplingStrategy::Uniform => self.domain.sample_interior(n_points, self.seed),
            SamplingStrategy::LatinHypercube => self.latin_hypercube_sample(n_points, false),
            SamplingStrategy::Sobol => self.sobol_sample(n_points, false),
            SamplingStrategy::AdaptiveRefinement => {
                self.domain.sample_interior(n_points, self.seed)
            }
        }
    }

    /// Sample boundary collocation points
    pub fn sample_boundary(&self, n_points: usize) -> Array2<f64> {
        match self.strategy {
            SamplingStrategy::Uniform => self.domain.sample_boundary(n_points, self.seed),
            SamplingStrategy::LatinHypercube => self.latin_hypercube_sample(n_points, true),
            SamplingStrategy::Sobol => self.sobol_sample(n_points, true),
            SamplingStrategy::AdaptiveRefinement => {
                self.domain.sample_boundary(n_points, self.seed)
            }
        }
    }

    /// Latin hypercube sampling (better space-filling than uniform)
    fn latin_hypercube_sample(&self, n_points: usize, boundary: bool) -> Array2<f64> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = if let Some(s) = self.seed {
            ChaCha8Rng::seed_from_u64(s)
        } else {
            ChaCha8Rng::from_entropy()
        };

        let bbox = self.domain.bounding_box();
        let dim = bbox.len() / 2;

        let mut points = Array2::zeros((n_points, dim));

        for d in 0..dim {
            let mut perm: Vec<usize> = (0..n_points).collect();
            for i in (1..n_points).rev() {
                let j = rng.gen_range(0..=i);
                perm.swap(i, j);
            }

            for (i, &p) in perm.iter().enumerate() {
                let u = (p as f64 + rng.gen::<f64>()) / n_points as f64;
                points[[i, d]] = bbox[2 * d] + u * (bbox[2 * d + 1] - bbox[2 * d]);
            }
        }

        let mut valid_points = Vec::new();
        let tolerance = 1e-8;
        for i in 0..n_points {
            let point: Vec<f64> = points.row(i).iter().cloned().collect();
            let loc = self.domain.classify_point(&point, tolerance);

            if boundary && loc == PointLocation::Boundary {
                valid_points.push(point);
            } else if !boundary && loc == PointLocation::Interior {
                valid_points.push(point);
            }
        }

        let n_valid = valid_points.len().min(n_points);
        let mut result = Array2::zeros((n_valid, dim));
        for (i, point) in valid_points.iter().take(n_valid).enumerate() {
            for (j, &coord) in point.iter().enumerate() {
                result[[i, j]] = coord;
            }
        }

        result
    }

    /// Sobol quasi-random sequence (low-discrepancy sampling)
    fn sobol_sample(&self, n_points: usize, boundary: bool) -> Array2<f64> {
        if boundary {
            return self.domain.sample_boundary(n_points, self.seed);
        }

        let bbox = self.domain.bounding_box();
        let dim = bbox.len() / 2;

        let unit_points = sobol_unit_hypercube_points(n_points, dim, self.seed);
        let mut points = Array2::zeros((n_points, dim));
        for (i, u) in unit_points.iter().enumerate() {
            for d in 0..dim {
                points[[i, d]] = bbox[2 * d] + u[d] * (bbox[2 * d + 1] - bbox[2 * d]);
            }
        }

        points
    }
}

fn sobol_unit_hypercube_points(n_points: usize, dim: usize, seed: Option<u64>) -> Vec<Vec<f64>> {
    const MAX_BITS: usize = 32;
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

    debug_assert_eq!(result.len(), n_points);
    result
}

fn sobol_direction_numbers(dim: usize) -> Vec<[u32; 32]> {
    const MAX_BITS: usize = 32;
    let mut dirs: Vec<[u32; MAX_BITS]> = Vec::with_capacity(dim);

    let mut v1 = [0u32; MAX_BITS];
    for i in 0..MAX_BITS {
        v1[i] = 1u32 << (31 - i);
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
    assert_eq!(m.len(), s);
    assert!(s >= 1 && s < MAX_BITS);

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

/// Adaptive mesh refinement for PINN training
///
/// Refines collocation point distribution based on PDE residual magnitude.
/// Regions with high residuals get more points in subsequent training epochs.
pub struct AdaptiveRefinement {
    /// Base sampler
    sampler: CollocationSampler,
    /// Current collocation points
    points: Array2<f64>,
    /// Residual values at each point (updated during training)
    residuals: Array1<f64>,
    /// Refinement threshold (points with residual > threshold are subdivided)
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
    /// Create a new adaptive refinement manager
    pub fn new(sampler: CollocationSampler, initial_points: Array2<f64>, threshold: f64) -> Self {
        let n_points = initial_points.nrows();
        Self {
            sampler,
            points: initial_points,
            residuals: Array1::zeros(n_points),
            threshold,
        }
    }

    /// Update residuals after a training step
    pub fn update_residuals(&mut self, residuals: Array1<f64>) {
        assert_eq!(
            residuals.len(),
            self.points.nrows(),
            "Residual count mismatch"
        );
        self.residuals = residuals;
    }

    /// Refine mesh by adding points near high-residual regions
    ///
    /// Returns new set of collocation points including refinements.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::geometry::RectangularDomain;

    #[test]
    fn test_collocation_sampler() {
        let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0);
        let sampler =
            CollocationSampler::new(Box::new(domain), SamplingStrategy::Uniform, Some(42));

        let interior = sampler.sample_interior(100);
        let boundary = sampler.sample_boundary(50);

        assert_eq!(interior.shape(), &[100, 2]);
        assert_eq!(boundary.shape(), &[50, 2]);
    }

    #[test]
    fn test_interface_condition_debug() {
        let ic = InterfaceCondition::ElasticContinuity;
        assert_eq!(format!("{:?}", ic), "ElasticContinuity");

        let ic2 = InterfaceCondition::AcousticElastic {
            fluid_density: 1000.0,
        };
        assert!(format!("{:?}", ic2).contains("1000"));
    }

    #[test]
    fn test_multi_region_locate() {
        let region1 =
            Box::new(RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0)) as Box<dyn GeometricDomain>;
        let region2 =
            Box::new(RectangularDomain::new_2d(1.0, 2.0, 0.0, 1.0)) as Box<dyn GeometricDomain>;

        let multi = MultiRegionDomain::new(
            vec![region1, region2],
            vec![0, 1],
            vec![InterfaceCondition::ElasticContinuity],
        );

        let loc = multi.locate_point(&[0.5, 0.5], 1e-6);
        assert!(loc.is_some());
        assert_eq!(loc.unwrap().0, 0);
    }

    #[test]
    fn test_adaptive_refinement() {
        let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0);
        let sampler =
            CollocationSampler::new(Box::new(domain), SamplingStrategy::Uniform, Some(42));

        let initial = sampler.sample_interior(10);
        let mut adaptive = AdaptiveRefinement::new(sampler, initial.clone(), 0.1);

        let mut residuals = Array1::zeros(10);
        residuals[0] = 0.5;
        residuals[5] = 0.3;

        adaptive.update_residuals(residuals);
        let refined = adaptive.refine(2.0);

        assert!(refined.nrows() > 10);
    }

    #[test]
    fn test_sobol_unit_hypercube_points() {
        let pts = sobol_unit_hypercube_points(8, 2, Some(0));
        assert_eq!(pts.len(), 8);
        for p in pts {
            assert_eq!(p.len(), 2);
            assert!(p[0] >= 0.0 && p[0] < 1.0);
            assert!(p[1] >= 0.0 && p[1] < 1.0);
        }
    }
}
