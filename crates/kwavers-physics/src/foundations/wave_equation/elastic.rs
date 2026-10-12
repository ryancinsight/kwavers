//! Elastic specific trait extensions

use super::core::{AutodiffWaveEquation, WaveEquation, WaveEquationOps};
use leto::Array3;

/// Shared elastic wave equation methods, independent of thread-safety bounds.
///
/// Covers the vector-displacement PDE:
///
/// ```text
/// ρ ∂²u/∂t² = ∇·σ + f
/// σ = C:ε
/// ε = ½(∇u + (∇u)ᵀ)
/// ```
///
/// All types implementing this trait must also implement [`WaveEquationOps`].
/// The thread-safety variants are [`ElasticWaveEquation`] (`Send + Sync`) and
/// [`AutodiffElasticWaveEquation`] (`Send` only).
pub trait ElasticOps: WaveEquationOps {
    /// Get Lamé first parameter λ(x) (Pa).
    fn lame_lambda(&self) -> Array3<f64>;

    /// Get Lamé second parameter μ(x) — shear modulus (Pa).
    fn lame_mu(&self) -> Array3<f64>;

    /// Get density field ρ(x) [kg/m³].
    fn density(&self) -> Array3<f64>;

    /// Compute stress tensor from the displacement field.
    ///
    /// σᵢⱼ = λδᵢⱼ∇·u + μ(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
    fn stress_from_displacement(&self, displacement: &Array3<f64>) -> Array3<f64>;

    /// Compute strain tensor from the displacement field.
    ///
    /// εᵢⱼ = ½(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
    fn strain_from_displacement(&self, displacement: &Array3<f64>) -> Array3<f64>;

    /// Compute total elastic energy.
    ///
    /// E = ∫ (½ρ|∂u/∂t|² + ½σ:ε) dV
    fn elastic_energy(&self, displacement: &Array3<f64>, velocity: &Array3<f64>) -> f64;

    /// P-wave (longitudinal) speed field (m/s).
    ///
    /// c_p = √((λ + 2μ) / ρ)
    fn p_wave_speed(&self) -> Array3<f64> {
        let lambda = self.lame_lambda();
        let mu = self.lame_mu();
        let rho = self.density();
        let lambda_plus_2mu = lambda.zip_map(&mu, |a, b| a + 2.0 * b);
        lambda_plus_2mu.zip_map(&rho, |a, b| (a / b).sqrt())
    }

    /// S-wave (shear) speed field (m/s).
    ///
    /// c_s = √(μ / ρ)
    fn s_wave_speed(&self) -> Array3<f64> {
        let mu = self.lame_mu();
        let rho = self.density();
        mu.zip_map(&rho, |a, b| (a / b).sqrt())
    }
}

/// Elastic wave equation for thread-safe (`Send + Sync`) traditional solvers.
///
/// Combines [`WaveEquation`] and [`ElasticOps`].  Implementations need only
/// satisfy both supertraits — no additional methods are required.
pub trait ElasticWaveEquation: WaveEquation + ElasticOps {}

/// Elastic wave equation for autodiff-based solvers that may be `!Sync`.
///
/// Combines [`AutodiffWaveEquation`] and [`ElasticOps`].  Implementations need
/// only satisfy both supertraits — no additional methods are required.
pub trait AutodiffElasticWaveEquation: AutodiffWaveEquation + ElasticOps {}
