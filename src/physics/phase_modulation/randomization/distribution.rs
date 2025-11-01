//! Phase distribution types for randomization

/// Phase distribution types
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum PhaseDistribution {
    /// Uniform random distribution [0, 2π)
    #[default]
    Uniform,

    /// Discrete phase states (e.g., 0, π/2, π, 3π/2)
    Discrete {
        /// Number of discrete states
        states: usize,
    },

    /// Gaussian distribution centered at 0
    Gaussian {
        /// Standard deviation (radians)
        std_dev: f64,
    },

    /// Binary phase (0 or π)
    Binary,

    /// Ternary phase (-2π/3, 0, 2π/3)
    Ternary,

    /// Quadrature phases (0, π/2, π, 3π/2)
    Quadrature,
}

