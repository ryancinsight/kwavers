use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
use coeus_autograd::{add, exp, mul, neg, scalar_add, scalar_mul, scalar_sub, Var};

#[cfg(test)]
mod tests;

/// Compute the prescribed free charge density source term ρ (C/m³).
///
/// ## Physics — Gauss's Law (Maxwell I, differential form)
///
/// The electrostatic residual enforced by the PINN is:
/// ```text
/// ∇·(ε∇φ) + ρ_free = 0
/// ```
/// where ρ_free is the **impressed** (externally prescribed) free charge density.
/// For source-free dielectric bulk media (water, tissue), ρ_free = 0 — physically
/// correct and the most common case in acoustic-electromagnetic coupling.
///
/// For problems with finite charge distributions (space-charge layers, plasma
/// regions, membrane potentials), ρ_free is specified via `domain_params`:
/// - `"charge_density"` (C/m³): spatially-uniform source charge density
/// - `"charge_x0"`, `"charge_y0"`, `"charge_sigma"` (m): Gaussian distribution centre
///   and width, combined with `"charge_density"` as peak amplitude
///
/// ## Gaussian source (Plonus 1988, §3.2)
///
/// A volume charge distribution of total charge Q₀ in a sphere of radius σ:
/// ```text
/// ρ(r) = ρ₀ · exp(−((x−x₀)² + (y−y₀)²) / (2σ²))
/// ```
/// with `ρ₀ = domain_params["charge_density"]`, `x₀ = domain_params["charge_x0"]`,
/// `y₀ = domain_params["charge_y0"]`, `σ = domain_params["charge_sigma"]`.
///
/// If none of the Gaussian parameters are present, a spatially-uniform density is used.
///
/// ## Reference
///
/// - Jackson, J.D. (1999). *Classical Electrodynamics* (3rd ed.). Wiley. §1.5.
/// - Griffiths, D.J. (2017). *Introduction to Electrodynamics* (4th ed.). §2.3.
pub fn compute_charge_density<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
    x: &Var<f32, B>,
    y: &Var<f32, B>,
    physics_params: &PinnDomainPhysicsParameters,
) -> Var<f32, B> {
    let rho_0 = physics_params
        .domain_params
        .get("charge_density")
        .copied()
        .unwrap_or(0.0) as f32;

    let backend = B::default();
    let zeros = || {
        Var::new(
            coeus_tensor::Tensor::zeros_on(x.tensor.shape(), &backend),
            false,
        )
    };

    if rho_0 == 0.0 {
        // Source-free bulk: ρ_free = 0 (physically correct for dielectrics)
        return zeros();
    }

    // Check for Gaussian distribution parameters
    let x0 = physics_params.domain_params.get("charge_x0").copied();
    let y0 = physics_params.domain_params.get("charge_y0").copied();
    let sigma = physics_params.domain_params.get("charge_sigma").copied();

    match (x0, y0, sigma) {
        (Some(x0), Some(y0), Some(sigma)) if sigma > 0.0 => {
            // Gaussian charge distribution: ρ(r) = ρ₀ · exp(−r²/(2σ²))
            // where r² = (x−x₀)² + (y−y₀)²
            let dx = scalar_sub(x, x0 as f32);
            let dy = scalar_sub(y, y0 as f32);
            let r_sq = add(&mul(&dx, &dx), &mul(&dy, &dy));
            let two_sigma_sq = (2.0 * sigma * sigma) as f32;
            scalar_mul(&exp(&neg(&scalar_mul(&r_sq, 1.0 / two_sigma_sq))), rho_0)
        }
        _ => {
            // Spatially-uniform charge density
            scalar_add(&zeros(), rho_0)
        }
    }
}

/// Compute the z-component of impressed current density J_z (A/m²).
///
/// ## Physics — Ampère's Law (Maxwell IV, 2D TM-mode)
///
/// In the magnetostatic / quasi-static 2D TM formulation (Hz, Ex, Ey fields),
/// the z-direction Ampère equation is:
/// ```text
/// ∇×H|_z = ε·∂Ez/∂t + σ·Ez + J_ext,z
/// ```
/// This function evaluates the **impressed** source current J_ext,z at the
/// query collocation points. Conduction current (σ·Ez) is handled separately
/// by the PINN residual via the model's predicted Ez field; this function
/// supplies only the externally prescribed contribution.
///
/// ## Source parameterisation
///
/// `physics_params.domain_params` keys:
/// - `"current_density_z"` (A/m²): spatially-uniform impressed current
/// - `"current_x0"`, `"current_y0"`, `"current_sigma"` (m): Gaussian source
///   centre and half-width, combined with `"current_density_z"` as peak amplitude
/// - `"conductivity"` (S/m) + `"e_z_background"` (V/m): conduction background
///   J_cond = σ·E_z,background (constant background field approximation)
///
/// ## Gaussian line-source (Balanis 2012, §3.4)
///
/// A filamentary current of linear density K₀ spread over radius σ:
/// ```text
/// J_z(r) = K₀ · exp(−r²/(2σ²)) / (2πσ²)
/// ```
///
/// ## Reference
///
/// - Jackson, J.D. (1999). *Classical Electrodynamics* (3rd ed.). Wiley. §6.7.
/// - Pozar, D.M. (2011). *Microwave Engineering* (4th ed.). Wiley. §1.3.
/// - Balanis, C.A. (2012). *Advanced Engineering Electromagnetics* (2nd ed.). §3.4.
pub fn compute_current_density_z<
    B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default,
>(
    x: &Var<f32, B>,
    y: &Var<f32, B>,
    physics_params: &PinnDomainPhysicsParameters,
) -> Var<f32, B> {
    // Conduction background: J_cond = σ·E_z,background (if both are specified)
    let sigma = physics_params
        .domain_params
        .get("conductivity")
        .copied()
        .unwrap_or(0.0);
    let e_z_bg = physics_params
        .domain_params
        .get("e_z_background")
        .copied()
        .unwrap_or(0.0);
    let j_cond = (sigma * e_z_bg) as f32;

    // Impressed source: uniform or Gaussian distribution
    let j0 = physics_params
        .domain_params
        .get("current_density_z")
        .copied()
        .unwrap_or(0.0) as f32;

    let backend = B::default();
    let zeros = || {
        Var::new(
            coeus_tensor::Tensor::zeros_on(x.tensor.shape(), &backend),
            false,
        )
    };

    let j_impressed = if j0 != 0.0 {
        let x0 = physics_params.domain_params.get("current_x0").copied();
        let y0 = physics_params.domain_params.get("current_y0").copied();
        let sig = physics_params.domain_params.get("current_sigma").copied();
        match (x0, y0, sig) {
            (Some(cx), Some(cy), Some(s)) if s > 0.0 => {
                // Gaussian impressed current: J_z(r) = J₀·exp(−r²/(2σ²))
                let dx = scalar_sub(x, cx as f32);
                let dy = scalar_sub(y, cy as f32);
                let r_sq = add(&mul(&dx, &dx), &mul(&dy, &dy));
                let two_s2 = (2.0 * s * s) as f32;
                scalar_mul(&exp(&neg(&scalar_mul(&r_sq, 1.0 / two_s2))), j0)
            }
            _ => scalar_add(&zeros(), j0),
        }
    } else {
        zeros()
    };

    // Total impressed current (conduction background + external source)
    scalar_add(&j_impressed, j_cond)
}
