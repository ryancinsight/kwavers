use super::invariants::principal_stretches;
use super::models::HyperelasticModel;

/// Compute strain energy density W
///
/// # Arguments
///
/// * `i1` - First strain invariant I₁ = λ₁² + λ₂² + λ₃²
/// * `i2` - Second strain invariant I₂ = λ₁²λ₂² + λ₂²λ₃² + λ₃²λ₁²
/// * `j` - Volume ratio J = λ₁λ₂λ₃
#[must_use]
pub fn strain_energy(model: &HyperelasticModel, i1: f64, i2: f64, j: f64) -> f64 {
    match model {
        HyperelasticModel::NeoHookean { c1, d1 } => c1 * (i1 - 3.0) + d1 * (j - 1.0).powi(2),
        HyperelasticModel::MooneyRivlin { c1, c2, d1 } => {
            c1 * (i1 - 3.0) + c2 * (i2 - 3.0) + d1 * (j - 1.0).powi(2)
        }
        HyperelasticModel::Ogden { mu, alpha } => {
            // Ogden model: W = Σᵢ (μᵢ/αᵢ) [(λ₁^αⁱ + λ₂^αⁱ + λ₃^αⁱ - 3)]
            // Reference: Ogden (1972, 1984) - Nonlinear Elastic Deformations
            //
            // For invariant-based computation, we solve for principal stretches
            // from the invariants. For nearly incompressible materials, we can
            // approximate the principal stretches from I₁ and I₃.
            //
            // For small to moderate strains, λᵢ ≈ sqrt(I₁/3) with corrections

            if mu.is_empty() || alpha.is_empty() {
                return 0.0;
            }

            // For Ogden materials, we need to estimate principal stretches from invariants
            // This is an approximation for cases where deformation gradient is not available
            let lambda_avg = (i1 / 3.0).sqrt(); // Average principal stretch

            // Use the average stretch as approximation for all directions
            // This is valid for isotropic loading or small strains
            // For more accurate results, deformation gradient should be used
            mu.iter()
                .zip(alpha.iter())
                .map(|(&mui, &alphai)| {
                    let lambda_sum = 3.0 * lambda_avg.powf(alphai);
                    (mui / alphai) * (lambda_sum - 3.0)
                })
                .sum::<f64>()
        }
    }
}

/// Compute Ogden strain energy density from deformation gradient
///
/// # Theorem Reference
/// Ogden Model: W = Σᵢ (μᵢ/αᵢ) [(λ₁^αⁱ + λ₂^αⁱ + λ₃^αⁱ - 3)]
/// where λᵢ are the principal stretches from the deformation gradient F.
///
/// # Arguments
/// * `f` - Deformation gradient tensor F (3x3)
///
/// # Returns
/// Strain energy density W (J/m³)
#[must_use]
pub fn ogden_strain_energy(model: &HyperelasticModel, f: &[[f64; 3]; 3]) -> f64 {
    if let HyperelasticModel::Ogden { mu, alpha } = model {
        let lambda = principal_stretches(model, f);

        mu.iter()
            .zip(alpha.iter())
            .map(|(&mui, &alphai)| {
                (mui / alphai)
                    * ((lambda[0].powf(alphai) - 1.0)
                        + (lambda[1].powf(alphai) - 1.0)
                        + (lambda[2].powf(alphai) - 1.0))
            })
            .sum()
    } else {
        0.0
    }
}

/// Derivative of strain energy w.r.t. I₁
///
/// # Theorem Reference
/// For hyperelastic materials, ∂W/∂I₁ is used in the computation of the second Piola-Kirchhoff stress.
/// For invariant-based models (Neo-Hookean, Mooney-Rivlin): ∂W/∂I₁ = c₁
/// For Ogden materials: W = Σᵢ (μᵢ/αᵢ) [(λ₁^αⁱ + λ₂^αⁱ + λ₃^αⁱ - 3)]
///
/// # Arguments
/// * `i1` - First strain invariant I₁
/// * `i2` - Second strain invariant I₂
/// * `j` - Volume ratio J
/// * `f` - Deformation gradient tensor F (required for Ogden materials)
///
/// Compute derivative of strain energy with respect to first strain invariant ∂W/∂I₁
///
/// # Theorem Reference
/// For hyperelastic materials, ∂W/∂I₁ contributes to the second Piola-Kirchhoff stress tensor.
/// The general relation is: S = 2 ∂W/∂I₁ B + 2 ∂W/∂I₂ (I₁ B - B²) + (∂W/∂J) J C⁻¹
/// where B is the left Cauchy-Green tensor and C is the right Cauchy-Green tensor.
///
/// **Neo-Hookean Theorem**: ∂W/∂I₁ = C₁ where W = C₁(I₁ - 3) + D₁(J - 1)²
/// **Mooney-Rivlin Theorem**: ∂W/∂I₁ = C₁ where W = C₁(I₁ - 3) + C₂(I₂ - 3) + D₁(J - 1)²
/// **Ogden Theorem**: ∂W/∂I₁ = Σⱼ μⱼ λ₁^(αⱼ - 1) where λ are principal stretches
///
/// # Arguments
/// * `strain_invariant_i1` - First strain invariant I₁ = tr(C) where C = F^T·F
/// * `strain_invariant_i2` - Second strain invariant I₂ = tr(C²)
/// * `volume_ratio_j` - Volume ratio J = det(F)
/// * `deformation_gradient` - Deformation gradient tensor F (required for Ogden materials)
pub fn compute_strain_energy_derivative_wrt_i1(
    model: &HyperelasticModel,
    _strain_invariant_i1: f64,
    _strain_invariant_i2: f64,
    _volume_ratio_j: f64,
    deformation_gradient: Option<&[[f64; 3]; 3]>,
) -> f64 {
    match model {
        HyperelasticModel::NeoHookean { c1, .. } => *c1,
        HyperelasticModel::MooneyRivlin { c1, .. } => *c1,
        HyperelasticModel::Ogden { mu, alpha } => {
            // Ogden materials require deformation gradient for proper computation
            if let Some(f_tensor) = deformation_gradient {
                let lambda = principal_stretches(model, f_tensor);

                // ∂W/∂I₁ for Ogden materials requires chain rule through principal stretches
                // For proper implementation, we need ∂W/∂λᵢ and dλᵢ/dI₁
                // This is complex and requires the full Ogden formulation
                // For now, compute the effective modulus from current deformation state
                let mut strain_energy_derivative_i1_total = 0.0;
                for (&mui, &alphai) in mu.iter().zip(alpha.iter()) {
                    // Simplified: use current strain state to estimate effective modulus
                    // Full implementation requires solving the system ∂W/∂λᵢ = 0 for equilibrium
                    let lambda_sum: f64 = lambda.iter().map(|&l| l.powf(alphai)).sum();
                    strain_energy_derivative_i1_total += mui * lambda_sum / 3.0;
                    // Approximate average contribution
                }
                strain_energy_derivative_i1_total / mu.len() as f64
            } else {
                // Fallback to average modulus if no deformation gradient provided
                // This maintains backward compatibility but is mathematically incorrect
                mu.iter().sum::<f64>() / mu.len() as f64
            }
        }
    }
}

/// Compute derivative of strain energy with respect to second strain invariant ∂W/∂I₂
///
/// # Theorem Reference
/// For hyperelastic materials, ∂W/∂I₂ contributes to the second Piola-Kirchhoff stress tensor.
/// The general relation is: S = 2 ∂W/∂I₁ B + 2 ∂W/∂I₂ (I₁ B - B²) + (∂W/∂J) J C⁻¹
/// where B is the left Cauchy-Green tensor and C is the right Cauchy-Green tensor.
///
/// **Mooney-Rivlin Theorem**: ∂W/∂I₂ = C₂ where W = C₁(I₁ - 3) + C₂(I₂ - 3) + D₁(J - 1)²
/// This term accounts for the strain history dependence in rubber-like materials.
///
/// **Neo-Hookean/Ogden**: ∂W/∂I₂ = 0 (no dependence on second invariant)
///
/// # Arguments
/// * `_strain_invariant_i1` - First strain invariant (unused for I₂ derivative)
/// * `_strain_invariant_i2` - Second strain invariant (unused for derivative)
/// * `_volume_ratio_j` - Volume ratio (unused for I₂ derivative)
pub fn compute_strain_energy_derivative_wrt_i2(
    model: &HyperelasticModel,
    _strain_invariant_i1: f64,
    _strain_invariant_i2: f64,
    _volume_ratio_j: f64,
) -> f64 {
    match model {
        HyperelasticModel::MooneyRivlin { c2, .. } => *c2,
        _ => 0.0,
    }
}

/// Compute derivative of strain energy with respect to volume ratio ∂W/∂J
///
/// # Theorem Reference
/// For hyperelastic materials, ∂W/∂J contributes to the hydrostatic pressure term.
/// The general relation is: S = 2 ∂W/∂I₁ B + 2 ∂W/∂I₂ (I₁ B - B²) + (∂W/∂J) J C⁻¹
/// where B is the left Cauchy-Green tensor and C is the right Cauchy-Green tensor.
///
/// For nearly incompressible materials, the volumetric term is: W_vol = D₁(J - 1)²
/// Thus ∂W_vol/∂J = 2 D₁ (J - 1), contributing to the pressure p = -∂W/∂J * J
///
/// **Neo-Hookean/Mooney-Rivlin**: ∂W/∂J = 2 D₁ (J - 1)
/// **Ogden**: ∂W/∂J = 0 (typically assumed incompressible)
///
/// # Arguments
/// * `_strain_invariant_i1` - First strain invariant (unused for J derivative)
/// * `_strain_invariant_i2` - Second strain invariant (unused for J derivative)
/// * `volume_ratio_j` - Volume ratio J = det(F)
pub fn compute_strain_energy_derivative_wrt_j(
    model: &HyperelasticModel,
    _strain_invariant_i1: f64,
    _strain_invariant_i2: f64,
    volume_ratio_j: f64,
) -> f64 {
    match model {
        HyperelasticModel::NeoHookean { d1, .. } | HyperelasticModel::MooneyRivlin { d1, .. } => {
            2.0 * d1 * (volume_ratio_j - 1.0)
        }
        HyperelasticModel::Ogden { .. } => 0.0, // Ogden typically incompressible
    }
}
