//! Element cross-talk modeling for realistic phased array behavior
//!
//! Models acoustic and electrical coupling between array elements

use ndarray::{Array1, Array2};

/// Cross-talk model for element coupling
#[derive(Debug, Clone)]
pub struct CrosstalkModel {
    /// Coupling coefficient
    coefficient: f64,
    /// Number of elements
    num_elements: usize,
    /// Coupling matrix
    coupling_matrix: Array2<f64>,
}

impl CrosstalkModel {
    /// Create cross-talk model
    #[must_use]
    pub fn create(num_elements: usize, coefficient: f64) -> Self {
        let coupling_matrix = Self::build_coupling_matrix(num_elements, coefficient);

        Self {
            coefficient,
            num_elements,
            coupling_matrix,
        }
    }

    /// Build coupling matrix based on element proximity
    fn build_coupling_matrix(n: usize, coefficient: f64) -> Array2<f64> {
        let mut matrix = Array2::eye(n);

        // Nearest-neighbor coupling
        for i in 0..n {
            if i > 0 {
                matrix[[i, i - 1]] = coefficient;
            }
            if i < n - 1 {
                matrix[[i, i + 1]] = coefficient;
            }

            // Second-nearest neighbor (weaker)
            if i > 1 {
                matrix[[i, i - 2]] = coefficient * coefficient;
            }
            if i < n - 2 {
                matrix[[i, i + 2]] = coefficient * coefficient;
            }
        }

        matrix
    }

    /// Apply cross-talk to element signals
    #[must_use]
    pub fn apply(&self, signals: &Array1<f64>) -> Array1<f64> {
        self.coupling_matrix.dot(signals)
    }

    /// Calculate isolation between elements in dB
    #[must_use]
    pub fn isolation_db(&self, element1: usize, element2: usize) -> f64 {
        if element1 == element2 {
            0.0
        } else {
            let coupling = self.coupling_matrix[[element1, element2]];
            if coupling > 0.0 {
                20.0 * coupling.log10()
            } else {
                f64::NEG_INFINITY // Perfect isolation
            }
        }
    }

    /// Update coupling coefficient
    pub fn set_coefficient(&mut self, coefficient: f64) {
        self.coefficient = coefficient;
        self.coupling_matrix = Self::build_coupling_matrix(self.num_elements, coefficient);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupling_matrix_symmetry() {
        let model = CrosstalkModel::create(5, 0.1);

        // Check symmetry
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(
                    model.coupling_matrix[[i, j]],
                    model.coupling_matrix[[j, i]],
                    "Coupling matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_diagonal_unity() {
        let model = CrosstalkModel::create(5, 0.1);

        // Check diagonal elements are 1
        for i in 0..5 {
            assert_eq!(
                model.coupling_matrix[[i, i]],
                1.0,
                "Diagonal elements should be 1"
            );
        }
    }
}
