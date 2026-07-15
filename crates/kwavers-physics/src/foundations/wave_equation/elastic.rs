//! Elastic specific trait extensions

use super::core::{AutodiffWaveEquation, WaveEquation};
use leto::Array3;

/// Elastic wave equation trait for traditional solvers (vector displacement field)
///
/// Governs propagation of elastic waves in solids:
///
/// ```text
/// ρ ∂²u/∂t² = ∇·σ + f
/// σ = C:ε
/// ε = ½(∇u + (∇u)ᵀ)
/// ```
///
/// where:
/// - u(x,t) is displacement vector (m)
/// - σ is stress tensor (Pa)
/// - ε is strain tensor (dimensionless)
/// - C is elastic modulus tensor (Pa)
/// - ρ(x) is density [kg/m³]
/// - f(x,t) is body force [N/m³]
///
/// # See Also
///
/// For autodiff-based implementations (PINN), see [`AutodiffElasticWaveEquation`].
pub trait ElasticWaveEquation: WaveEquation {
    /// Get Lamé first parameter λ(x) (Pa)
    fn lame_lambda(&self) -> Array3<f64>;

    /// Get Lamé second parameter μ(x) (shear modulus) (Pa)
    fn lame_mu(&self) -> Array3<f64>;

    /// Get density field ρ(x) [kg/m³]
    fn density(&self) -> Array3<f64>;

    /// Compute stress tensor from displacement field
    ///
    /// σᵢⱼ = λδᵢⱼ∇·u + μ(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
    fn stress_from_displacement(&self, displacement: &Array3<f64>) -> Array3<f64>;

    /// Compute strain tensor from displacement field
    ///
    /// εᵢⱼ = ½(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
    fn strain_from_displacement(&self, displacement: &Array3<f64>) -> Array3<f64>;

    /// Compute elastic energy
    ///
    /// E = ∫ (½ρ|∂u/∂t|² + ½σ:ε) dV
    fn elastic_energy(&self, displacement: &Array3<f64>, velocity: &Array3<f64>) -> f64;

    /// Get P-wave (longitudinal) speed (m/s)
    fn p_wave_speed(&self) -> Array3<f64> {
        let lambda = self.lame_lambda();
        let mu = self.lame_mu();
        let rho = self.density();
        let two_mu = mu.mapv(|x| 2.0 * x);
        let lambda_plus_2mu = lambda.zip_map(&two_mu, |a, b| a + b);
        let cp_sq = lambda_plus_2mu.zip_map(&rho, |a, b| a / b);
        cp_sq.mapv(f64::sqrt)
    }

    /// Get S-wave (shear) speed (m/s)
    fn s_wave_speed(&self) -> Array3<f64> {
        let mu = self.lame_mu();
        let rho = self.density();
        let cp_sq = mu.zip_map(&rho, |a, b| a / b);
        cp_sq.mapv(f64::sqrt)
    }
}

/// Elastic wave equation trait for autodiff-based solvers
///
/// This trait mirrors [`ElasticWaveEquation`] but extends [`AutodiffWaveEquation`]
/// instead of [`WaveEquation`], relaxing the `Sync` constraint to accommodate
/// neural network frameworks.
///
/// # Use Cases
///
/// - Physics-Informed Neural Networks (PINN)
/// - Neural operator methods (FNO, DeepONet)
/// - Hybrid neural-numerical solvers
///
/// # Mathematical Equivalence
///
/// Despite the different trait bounds, implementations must satisfy the same
/// mathematical constraints as traditional solvers:
/// - Material property bounds (ρ > 0, μ > 0, λ > -2μ/3)
/// - Wave speed relationships (cₚ > cₛ)
/// - PDE satisfaction (residual minimization)
/// - Energy conservation
///
/// The validation framework provides separate functions for each trait hierarchy
/// but enforces identical mathematical requirements.
pub trait AutodiffElasticWaveEquation: AutodiffWaveEquation {
    /// Get Lamé first parameter λ(x) (Pa)
    fn lame_lambda(&self) -> Array3<f64>;

    /// Get Lamé second parameter μ(x) (shear modulus) (Pa)
    fn lame_mu(&self) -> Array3<f64>;

    /// Get density field ρ(x) [kg/m³]
    fn density(&self) -> Array3<f64>;

    /// Compute stress tensor from displacement field
    ///
    /// σᵢⱼ = λδᵢⱼ∇·u + μ(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
    fn stress_from_displacement(&self, displacement: &Array3<f64>) -> Array3<f64>;

    /// Compute strain tensor from displacement field
    ///
    /// εᵢⱼ = ½(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
    fn strain_from_displacement(&self, displacement: &Array3<f64>) -> Array3<f64>;

    /// Compute elastic energy
    ///
    /// E = ∫ (½ρ|∂u/∂t|² + ½σ:ε) dV
    fn elastic_energy(&self, displacement: &Array3<f64>, velocity: &Array3<f64>) -> f64;

    /// Get P-wave (longitudinal) speed (m/s)
    fn p_wave_speed(&self) -> Array3<f64> {
        let lambda = self.lame_lambda();
        let mu = self.lame_mu();
        let rho = self.density();
        let two_mu = mu.mapv(|x| 2.0 * x);
        let lambda_plus_2mu = lambda.zip_map(&two_mu, |a, b| a + b);
        let cp_sq = lambda_plus_2mu.zip_map(&rho, |a, b| a / b);
        cp_sq.mapv(f64::sqrt)
    }

    /// Get S-wave (shear) speed (m/s)
    fn s_wave_speed(&self) -> Array3<f64> {
        let mu = self.lame_mu();
        let rho = self.density();
        let cp_sq = mu.zip_map(&rho, |a, b| a / b);
        cp_sq.mapv(f64::sqrt)
    }
}
