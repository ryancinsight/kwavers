//! Gradient computation for FWI
//! Based on Plessix (2006): "Adjoint-state method for gradient computation"

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use ndarray::{Array3, Zip};

/// Gradient computation methods for FWI
#[derive(Debug)]
pub struct GradientComputer {
    /// Preconditioning matrix (optional)
    preconditioner: Option<Array3<f64>>,
}

impl Default for GradientComputer {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientComputer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            preconditioner: None,
        }
    }

    /// Compute gradient using adjoint method
    /// Based on Tarantola (1984) formulation: "Inversion of seismic reflection data"
    #[must_use]
    pub fn compute_adjoint_gradient(
        &self,
        forward_wavefield: &Array3<f64>,
        adjoint_wavefield: &Array3<f64>,
        dt: f64,
    ) -> Array3<f64> {
        // Gradient = -∫ (∂²u_f/∂t²) * u_a dt
        // where u_f is forward wavefield, u_a is adjoint wavefield
        //
        // For acoustic media, the gradient expression simplifies to
        // the correlation of forward and adjoint wavefields (Virieux & Operto 2009)
        // ∂J/∂c = 2 * ∫ (1/c³) * u_f * u_a dt
        //
        // This is the correlation of forward and adjoint wavefields
        // multiplied by the proper scaling factor

        let mut gradient = Array3::zeros(forward_wavefield.dim());

        // Compute zero-lag correlation between forward and adjoint wavefields
        Zip::from(&mut gradient)
            .and(forward_wavefield)
            .and(adjoint_wavefield)
            .for_each(|grad, &u_f, &u_a| {
                // Cross-correlation at zero lag with proper scaling
                *grad = -2.0 * dt * u_f * u_a;
            });

        // Apply preconditioning if available
        if let Some(ref precond) = self.preconditioner {
            Zip::from(&mut gradient).and(precond).for_each(|grad, &p| {
                *grad *= p;
            });
        }

        gradient
    }

    /// Compute gradient using classical imaging condition (alternative method)
    /// Based on Claerbout (1985): "Imaging the Earth's Interior"  
    #[must_use]
    pub fn compute_imaging_gradient(
        &self,
        forward_wavefield: &Array3<f64>,
        adjoint_wavefield: &Array3<f64>,
        _dt: f64,
    ) -> Array3<f64> {
        // Compute gradient using imaging condition: ∇J = -∫ u_f · u_a dt
        // where u_f is forward wavefield and u_a is adjoint wavefield

        let mut gradient = Array3::zeros(forward_wavefield.dim());

        // Zero-lag cross-correlation between forward and adjoint wavefields
        Zip::from(&mut gradient)
            .and(forward_wavefield)
            .and(adjoint_wavefield)
            .for_each(|grad, &u_f, &u_a| {
                *grad = -u_f * u_a;
            });

        gradient
    }

    /// Apply preconditioning to gradient
    /// Based on Shin et al. (2001): "Amplitude preservation for elastic migration"
    pub fn apply_preconditioning(&self, gradient: &mut Array3<f64>) {
        if let Some(ref precond) = self.preconditioner {
            Zip::from(gradient).and(precond).for_each(|g, &p| {
                *g *= p;
            });
        }
    }

    /// Compute gradient with source encoding
    /// Based on Krebs et al. (2009): "Fast full-wavefield seismic inversion"
    ///
    /// ## Theorem
    /// If the per-source gradient is linear in the source residual, then the
    /// encoded gradient is the weighted sum of the individual source gradients:
    ///
    /// ```text
    /// G_enc = Σᵢ αᵢ Gᵢ
    /// ```
    ///
    /// ## Proof sketch
    /// The adjoint-state gradient is linear in the right-hand-side source term.
    /// Therefore the gradient of a linear combination of encoded sources is the
    /// corresponding linear combination of the individual gradients.
    pub fn encoded_gradient(
        &self,
        source_gradients: &[Array3<f64>],
        encoding_weights: &[f64],
    ) -> KwaversResult<Array3<f64>> {
        if source_gradients.is_empty() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "encoded_gradient requires at least one source gradient".to_string(),
                },
            ));
        }

        if source_gradients.len() != encoding_weights.len() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Encoded gradient weight count mismatch: gradients={}, weights={}",
                        source_gradients.len(),
                        encoding_weights.len()
                    ),
                },
            ));
        }

        let shape = source_gradients[0].dim();
        if source_gradients.iter().any(|g| g.dim() != shape) {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "All source gradients must share the same shape".to_string(),
                },
            ));
        }

        let mut encoded = Array3::zeros(shape);
        for (gradient, &weight) in source_gradients.iter().zip(encoding_weights.iter()) {
            Zip::from(&mut encoded).and(gradient).for_each(|acc, &g| {
                *acc += weight * g;
            });
        }

        Ok(encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_encoded_gradient_weighted_sum() {
        let computer = GradientComputer::new();
        let g1 = Array3::from_elem((2, 2, 2), 1.0);
        let g2 = Array3::from_elem((2, 2, 2), 3.0);

        let encoded = computer
            .encoded_gradient(&[g1, g2], &[0.25, 0.75])
            .expect("encoded gradient must succeed");

        assert!(encoded.iter().all(|&v| (v - 2.5).abs() < f64::EPSILON));
    }

    #[test]
    fn test_encoded_gradient_rejects_shape_mismatch() {
        let computer = GradientComputer::new();
        let g1 = Array3::from_elem((2, 2, 2), 1.0);
        let g2 = Array3::from_elem((3, 2, 2), 3.0);

        let err = computer
            .encoded_gradient(&[g1, g2], &[0.5, 0.5])
            .expect_err("shape mismatch must fail");

        assert!(format!("{err:?}").contains("same shape"));
    }
}
