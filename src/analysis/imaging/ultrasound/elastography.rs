//! Elastography imaging analysis types

use ndarray::Array3;

/// Inversion method for elasticity reconstruction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InversionMethod {
    /// Direct algebraic inversion
    DirectInversion,
    /// Iterative optimization
    IterativeOptimization,
    /// Regularized solution
    Regularized,
}

/// Nonlinear inversion method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonlinearInversionMethod {
    /// Nonlinear algebraic method
    NonlinearAlgebraic,
    /// Constitutive model-based
    ConstitutiveModel,
    /// Finite element based
    FiniteElement,
}

/// Elasticity map from shear wave elastography
#[derive(Debug, Clone)]
pub struct ElasticityMap {
    /// Shear modulus (Pa)
    pub shear_modulus: Array3<f64>,
    /// Young's modulus (Pa)
    pub youngs_modulus: Array3<f64>,
    /// Estimation uncertainty
    pub uncertainty: Array3<f64>,
}

/// Nonlinear parameter map
#[derive(Debug, Clone)]
pub struct NonlinearParameterMap {
    /// Nonlinearity parameter
    pub nonlinearity: Array3<f64>,
    /// Estimation uncertainty
    pub uncertainty: Array3<f64>,
}
