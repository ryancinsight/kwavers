/// 1D radial Smoluchowski diffusion solver for radical species.
///
/// Solves the spherically symmetric diffusion equation on a logarithmic radial
/// grid using Crank-Nicolson time integration.
///
/// | Parameter | Default | Notes |
/// |-----------|---------|-------|
/// | `n_points` | 64 | log-radial nodes |
/// | `r_max_factor` | 1000 | `r_max = R_bubble x 1000` |
#[derive(Debug, Clone)]
pub struct RadicalDiffusionSolver {
    /// Number of radial grid points.
    pub n_points: usize,
    /// Bubble radius (m), setting the inner boundary.
    pub r_bubble_m: f64,
    /// `r_max = r_bubble * r_max_factor`.
    pub r_max_factor: f64,
}

impl Default for RadicalDiffusionSolver {
    fn default() -> Self {
        Self {
            n_points: 64,
            r_bubble_m: 10e-6,
            r_max_factor: 1000.0,
        }
    }
}

impl RadicalDiffusionSolver {
    /// Create a solver with a specific bubble radius.
    #[must_use]
    pub fn new(r_bubble_m: f64) -> Self {
        Self {
            n_points: 64,
            r_bubble_m,
            r_max_factor: 1000.0,
        }
    }
}

/// Result from a single diffusion time step.
#[derive(Debug, Clone)]
pub struct DiffusionStepResult {
    /// Updated concentrations \[species\]\[radial_node\] in mol/m^3.
    pub concentrations: Vec<Vec<f64>>,
    /// Maximum concentration change across all species and nodes.
    pub max_delta: f64,
}

/// Error type for diffusion solver.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffusionError {
    /// The bubble radius is not positive.
    InvalidBubbleRadius(f64),
    /// `n_points < 3`; cannot form a tridiagonal system.
    TooFewPoints(usize),
    /// Thomas algorithm encountered a zero pivot.
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
