//! Stiffness detection for IMEX schemes

use super::traits::StiffnessIndicator;
use crate::core::error::KwaversResult;
use ndarray::{Array3, Zip};

/// Stiffness metric information
#[derive(Debug, Clone)]
pub struct StiffnessMetric {
    /// Stiffness ratio
    pub ratio: f64,
    /// Maximum eigenvalue magnitude
    pub max_eigenvalue: f64,
    /// Minimum eigenvalue magnitude
    pub min_eigenvalue: f64,
    /// Whether the problem is considered stiff
    pub is_stiff: bool,
}

impl StiffnessMetric {
    /// Check if the problem is stiff
    #[must_use]
    pub fn is_stiff(&self) -> bool {
        self.is_stiff
    }

    /// Get the stiffness ratio
    #[must_use]
    pub fn ratio(&self) -> f64 {
        self.ratio
    }
}

/// Stiffness detector
#[derive(Debug, Clone)]
pub struct StiffnessDetector {
    /// Threshold for stiffness detection
    threshold: f64,
    /// Last computed metric
    last_metric: Option<StiffnessMetric>,
    /// Method for stiffness detection
    method: StiffnessMethod,
}

/// Methods for stiffness detection
#[derive(Debug, Clone, Copy)]
pub enum StiffnessMethod {
    /// Eigenvalue-based detection
    Eigenvalue,
    /// Power iteration method
    PowerIteration,
    /// Norm-based estimation
    NormBased,
}

impl StiffnessDetector {
    /// Create a new stiffness detector
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            last_metric: None,
            method: StiffnessMethod::NormBased,
        }
    }

    /// Set detection method
    #[must_use]
    pub fn with_method(mut self, method: StiffnessMethod) -> Self {
        self.method = method;
        self
    }

    /// Detect stiffness in the problem
    pub fn detect<F, G>(
        &mut self,
        field: &Array3<f64>,
        explicit_rhs: &F,
        implicit_rhs: &G,
    ) -> KwaversResult<StiffnessMetric>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        let metric = match self.method {
            StiffnessMethod::Eigenvalue => {
                self.eigenvalue_detection(field, explicit_rhs, implicit_rhs)?
            }
            StiffnessMethod::PowerIteration => {
                self.power_iteration_detection(field, explicit_rhs, implicit_rhs)?
            }
            StiffnessMethod::NormBased => {
                self.norm_based_detection(field, explicit_rhs, implicit_rhs)?
            }
        };

        self.last_metric = Some(metric.clone());
        Ok(metric)
    }

    /// Get last computed metric
    #[must_use]
    pub fn last_metric(&self) -> Option<StiffnessMetric> {
        self.last_metric.clone()
    }

    /// Eigenvalue-based detection (precise but computationally intensive)
    fn eigenvalue_detection<F, G>(
        &self,
        field: &Array3<f64>,
        explicit_rhs: &F,
        implicit_rhs: &G,
    ) -> KwaversResult<StiffnessMetric>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        // Approximate eigenvalues using finite differences
        let epsilon = 1e-8;
        let mut max_eigenvalue: f64 = 0.0;
        let mut min_eigenvalue = f64::INFINITY;

        // Sample a few directions
        for _ in 0..10 {
            // Create random perturbation
            let mut perturbation = Array3::zeros(field.dim());
            for p in &mut perturbation {
                *p = 2.0 * rand::random::<f64>() - 1.0;
            }

            // Normalize perturbation
            let norm: f64 = perturbation.iter().map(|&x| x * x).sum::<f64>().sqrt();
            perturbation.mapv_inplace(|x| x / norm);

            // Compute Jacobian-vector product
            let mut field_perturbed = field.clone();
            Zip::from(&mut field_perturbed)
                .and(&perturbation)
                .for_each(|fp, &p| *fp += epsilon * p);

            let f_base_explicit = explicit_rhs(field)?;
            let f_base_implicit = implicit_rhs(field)?;
            let f_pert_explicit = explicit_rhs(&field_perturbed)?;
            let f_pert_implicit = implicit_rhs(&field_perturbed)?;

            // Compute eigenvalue approximation
            let mut jv = Array3::zeros(field.dim());
            Zip::from(&mut jv)
                .and(&f_pert_explicit)
                .and(&f_base_explicit)
                .and(&f_pert_implicit)
                .and(&f_base_implicit)
                .for_each(|j, &fpe, &fbe, &fpi, &fbi| {
                    *j = ((fpe - fbe) + (fpi - fbi)) / epsilon;
                });

            // Rayleigh quotient
            let eigenvalue = jv
                .iter()
                .zip(&perturbation)
                .map(|(&j, &p)| j * p)
                .sum::<f64>();
            max_eigenvalue = max_eigenvalue.max(eigenvalue.abs());
            min_eigenvalue = min_eigenvalue.min(eigenvalue.abs());
        }

        let ratio = if min_eigenvalue > 0.0 {
            max_eigenvalue / min_eigenvalue
        } else {
            max_eigenvalue
        };

        Ok(StiffnessMetric {
            ratio,
            max_eigenvalue,
            min_eigenvalue,
            is_stiff: ratio > self.threshold,
        })
    }

    /// Power iteration method (efficient approximation)
    fn power_iteration_detection<F, G>(
        &self,
        field: &Array3<f64>,
        explicit_rhs: &F,
        implicit_rhs: &G,
    ) -> KwaversResult<StiffnessMetric>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        let epsilon = 1e-8;
        let mut v = Array3::from_elem(field.dim(), 1.0);

        // Normalize
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        v.mapv_inplace(|x| x / norm);

        let mut eigenvalue = 0.0;

        // Power iteration
        for _ in 0..20 {
            // Apply Jacobian
            let mut field_perturbed = field.clone();
            Zip::from(&mut field_perturbed)
                .and(&v)
                .for_each(|fp, &vi| *fp += epsilon * vi);

            let f_base_explicit = explicit_rhs(field)?;
            let f_base_implicit = implicit_rhs(field)?;
            let f_pert_explicit = explicit_rhs(&field_perturbed)?;
            let f_pert_implicit = implicit_rhs(&field_perturbed)?;

            let mut jv = Array3::zeros(field.dim());
            Zip::from(&mut jv)
                .and(&f_pert_explicit)
                .and(&f_base_explicit)
                .and(&f_pert_implicit)
                .and(&f_base_implicit)
                .for_each(|j, &fpe, &fbe, &fpi, &fbi| {
                    *j = ((fpe - fbe) + (fpi - fbi)) / epsilon;
                });

            // Update eigenvalue estimate
            eigenvalue = jv.iter().zip(&v).map(|(&j, &vi)| j * vi).sum::<f64>();

            // Update vector
            v = jv;
            let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                v.mapv_inplace(|x| x / norm);
            } else {
                break;
            }
        }

        Ok(StiffnessMetric {
            ratio: eigenvalue.abs(),
            max_eigenvalue: eigenvalue.abs(),
            min_eigenvalue: 1.0, // Assume minimum eigenvalue is O(1)
            is_stiff: eigenvalue.abs() > self.threshold,
        })
    }

    /// Norm-based detection (computationally efficient but approximate)
    fn norm_based_detection<F, G>(
        &self,
        field: &Array3<f64>,
        explicit_rhs: &F,
        implicit_rhs: &G,
    ) -> KwaversResult<StiffnessMetric>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        // Evaluate RHS functions
        let f_explicit = explicit_rhs(field)?;
        let f_implicit = implicit_rhs(field)?;

        // Compute norms
        let explicit_norm: f64 = f_explicit.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let implicit_norm: f64 = f_implicit.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let field_norm: f64 = field.iter().map(|&x| x * x).sum::<f64>().sqrt();

        // Estimate stiffness based on relative magnitudes
        let explicit_scale = if field_norm > 0.0 {
            explicit_norm / field_norm
        } else {
            explicit_norm
        };

        let implicit_scale = if field_norm > 0.0 {
            implicit_norm / field_norm
        } else {
            implicit_norm
        };

        let ratio = if explicit_scale > 0.0 {
            implicit_scale / explicit_scale
        } else {
            implicit_scale
        };

        Ok(StiffnessMetric {
            ratio,
            max_eigenvalue: implicit_scale,
            min_eigenvalue: explicit_scale,
            is_stiff: ratio > self.threshold,
        })
    }
}

impl StiffnessIndicator for StiffnessDetector {
    fn compute<F, G>(
        &self,
        field: &Array3<f64>,
        explicit_rhs: &F,
        implicit_rhs: &G,
    ) -> KwaversResult<f64>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        let metric = match self.method {
            StiffnessMethod::Eigenvalue => {
                self.eigenvalue_detection(field, explicit_rhs, implicit_rhs)?
            }
            StiffnessMethod::PowerIteration => {
                self.power_iteration_detection(field, explicit_rhs, implicit_rhs)?
            }
            StiffnessMethod::NormBased => {
                self.norm_based_detection(field, explicit_rhs, implicit_rhs)?
            }
        };

        Ok(metric.ratio)
    }

    fn is_stiff(&self, indicator: f64) -> bool {
        indicator > self.threshold
    }
}
