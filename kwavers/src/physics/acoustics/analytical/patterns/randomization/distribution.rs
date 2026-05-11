//! Phase distribution types for randomization

/// Phase distribution types
#[derive(Debug, Clone, Copy, PartialEq, Default)]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Default variant is Uniform.
    #[test]
    fn default_is_uniform() {
        assert_eq!(PhaseDistribution::default(), PhaseDistribution::Uniform);
    }

    /// Each variant is distinct under PartialEq.
    #[test]
    fn variants_are_distinct() {
        assert_ne!(PhaseDistribution::Binary, PhaseDistribution::Ternary);
        assert_ne!(PhaseDistribution::Ternary, PhaseDistribution::Quadrature);
        assert_ne!(PhaseDistribution::Uniform, PhaseDistribution::Binary);
    }

    /// Discrete variant stores the state count.
    #[test]
    fn discrete_stores_state_count() {
        let d = PhaseDistribution::Discrete { states: 8 };
        match d {
            PhaseDistribution::Discrete { states } => assert_eq!(states, 8),
            _ => panic!("expected Discrete"),
        }
    }

    /// Gaussian variant stores std_dev.
    #[test]
    fn gaussian_stores_std_dev() {
        let g = PhaseDistribution::Gaussian { std_dev: 0.5 };
        match g {
            PhaseDistribution::Gaussian { std_dev } => {
                assert!((std_dev - 0.5).abs() < 1e-15)
            }
            _ => panic!("expected Gaussian"),
        }
    }

    /// Clone produces an equal copy.
    #[test]
    fn clone_produces_equal_copy() {
        let original = PhaseDistribution::Discrete { states: 4 };
        let cloned = original;
        assert_eq!(original, cloned);
    }
}
