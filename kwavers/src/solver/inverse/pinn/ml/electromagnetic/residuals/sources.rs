use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

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
pub fn compute_charge_density<B: AutodiffBackend>(
    x: &Tensor<B, 2>,
    y: &Tensor<B, 2>,
    physics_params: &PhysicsParameters,
) -> Tensor<B, 2> {
    let rho_0 = physics_params
        .domain_params
        .get("charge_density")
        .copied()
        .unwrap_or(0.0) as f32;

    if rho_0 == 0.0 {
        // Source-free bulk: ρ_free = 0 (physically correct for dielectrics)
        return Tensor::zeros_like(x);
    }

    // Check for Gaussian distribution parameters
    let x0 = physics_params.domain_params.get("charge_x0").copied();
    let y0 = physics_params.domain_params.get("charge_y0").copied();
    let sigma = physics_params.domain_params.get("charge_sigma").copied();

    match (x0, y0, sigma) {
        (Some(x0), Some(y0), Some(sigma)) if sigma > 0.0 => {
            // Gaussian charge distribution: ρ(r) = ρ₀ · exp(−r²/(2σ²))
            // where r² = (x−x₀)² + (y−y₀)²
            let dx = x.clone().sub_scalar(x0 as f32);
            let dy = y.clone().sub_scalar(y0 as f32);
            let r_sq = dx.clone().mul(dx).add(dy.clone().mul(dy));
            let two_sigma_sq = (2.0 * sigma * sigma) as f32;
            r_sq.div_scalar(two_sigma_sq).neg().exp().mul_scalar(rho_0)
        }
        _ => {
            // Spatially-uniform charge density
            Tensor::zeros_like(x).add_scalar(rho_0)
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
pub fn compute_current_density_z<B: AutodiffBackend>(
    x: &Tensor<B, 2>,
    y: &Tensor<B, 2>,
    physics_params: &PhysicsParameters,
) -> Tensor<B, 2> {
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

    let j_impressed = if j0 != 0.0 {
        let x0 = physics_params.domain_params.get("current_x0").copied();
        let y0 = physics_params.domain_params.get("current_y0").copied();
        let sig = physics_params.domain_params.get("current_sigma").copied();
        match (x0, y0, sig) {
            (Some(cx), Some(cy), Some(s)) if s > 0.0 => {
                // Gaussian impressed current: J_z(r) = J₀·exp(−r²/(2σ²))
                let dx = x.clone().sub_scalar(cx as f32);
                let dy = y.clone().sub_scalar(cy as f32);
                let r_sq = dx.clone().mul(dx).add(dy.clone().mul(dy));
                let two_s2 = (2.0 * s * s) as f32;
                r_sq.div_scalar(two_s2).neg().exp().mul_scalar(j0)
            }
            _ => Tensor::zeros_like(x).add_scalar(j0),
        }
    } else {
        Tensor::zeros_like(x)
    };

    // Total impressed current (conduction background + external source)
    j_impressed.add_scalar(j_cond)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
    use burn::tensor::Tensor;
    use std::collections::HashMap;

    // -----------------------------------------------------------------------
    // Tests for compute_charge_density
    // -----------------------------------------------------------------------

    /// Source-free dielectric bulk: ρ_free = 0.
    ///
    /// Proof: charge_density not set in domain_params → rho_0 = 0 → return zeros.
    #[cfg(feature = "pinn")]
    #[test]
    fn test_charge_density_zero_for_source_free_medium() {
        type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
        use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
        use burn::tensor::Tensor;
        use std::collections::HashMap;

        let params = PhysicsParameters {
            material_properties: HashMap::new(),
            boundary_values: HashMap::new(),
            initial_values: HashMap::new(),
            domain_params: HashMap::new(),
        };
        let x: Tensor<B, 2> = Tensor::zeros([4, 1], &Default::default());
        let y: Tensor<B, 2> = Tensor::zeros([4, 1], &Default::default());

        let rho = super::compute_charge_density::<B>(&x, &y, &params);
        let rho_data: Vec<f32> = rho.into_data().to_vec().unwrap();
        for v in &rho_data {
            assert!(
                v.abs() < 1e-10,
                "expected ρ=0 for source-free medium, got {}",
                v
            );
        }
    }

    /// Uniform impressed charge density: all output elements = rho_0.
    ///
    /// Proof: domain_params["charge_density"] = ρ₀ → return tensor filled with ρ₀.
    #[cfg(feature = "pinn")]
    #[test]
    fn test_charge_density_uniform_matches_param() {
        type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
        use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
        use burn::tensor::Tensor;
        use std::collections::HashMap;

        let rho_expected = 1.5e-3_f64;
        let mut domain = HashMap::new();
        domain.insert("charge_density".to_string(), rho_expected);
        let params = PhysicsParameters {
            material_properties: HashMap::new(),
            boundary_values: HashMap::new(),
            initial_values: HashMap::new(),
            domain_params: domain,
        };
        let x: Tensor<B, 2> = Tensor::zeros([3, 1], &Default::default());
        let y: Tensor<B, 2> = Tensor::zeros([3, 1], &Default::default());

        let rho = super::compute_charge_density::<B>(&x, &y, &params);
        let rho_data: Vec<f32> = rho.into_data().to_vec().unwrap();
        for v in &rho_data {
            let diff = (v - rho_expected as f32).abs();
            assert!(
                diff < 1e-5,
                "expected ρ={:.3e}, got {:.3e}",
                rho_expected,
                v
            );
        }
    }

    /// Gaussian charge density: peak at (x0,y0), decays with σ.
    ///
    /// At (x0,y0) the Gaussian equals 1, so ρ = ρ₀.
    #[cfg(feature = "pinn")]
    #[test]
    fn test_charge_density_gaussian_peak_at_centre() {
        type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
        use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
        use burn::tensor::Tensor;
        use std::collections::HashMap;

        let rho_0 = 2.0_f64;
        let mut domain = HashMap::new();
        domain.insert("charge_density".to_string(), rho_0);
        domain.insert("charge_x0".to_string(), 0.5_f64);
        domain.insert("charge_y0".to_string(), 0.5_f64);
        domain.insert("charge_sigma".to_string(), 0.1_f64);
        let params = PhysicsParameters {
            material_properties: HashMap::new(),
            boundary_values: HashMap::new(),
            initial_values: HashMap::new(),
            domain_params: domain,
        };
        // Single point exactly at the Gaussian centre → exp(0) = 1 → ρ = ρ₀
        let x: Tensor<B, 2> = Tensor::from_data([[0.5_f32]], &Default::default());
        let y: Tensor<B, 2> = Tensor::from_data([[0.5_f32]], &Default::default());

        let rho = super::compute_charge_density::<B>(&x, &y, &params);
        let rho_data: Vec<f32> = rho.into_data().to_vec().unwrap();
        let diff = (rho_data[0] - rho_0 as f32).abs();
        assert!(
            diff < 1e-4,
            "Gaussian peak should equal ρ₀={:.2}, got {:.6}",
            rho_0,
            rho_data[0]
        );
    }

    // -----------------------------------------------------------------------
    // Tests for compute_current_density_z
    // -----------------------------------------------------------------------

    /// No sources or conductivity: J_z = 0 (source-free dielectric).
    #[cfg(feature = "pinn")]
    #[test]
    fn test_current_density_zero_for_dielectric() {
        type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
        use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
        use burn::tensor::Tensor;
        use std::collections::HashMap;

        let params = PhysicsParameters {
            material_properties: HashMap::new(),
            boundary_values: HashMap::new(),
            initial_values: HashMap::new(),
            domain_params: HashMap::new(),
        };
        let x: Tensor<B, 2> = Tensor::zeros([5, 1], &Default::default());
        let y: Tensor<B, 2> = Tensor::zeros([5, 1], &Default::default());

        let jz = super::compute_current_density_z::<B>(&x, &y, &params);
        let jz_data: Vec<f32> = jz.into_data().to_vec().unwrap();
        for v in &jz_data {
            assert!(v.abs() < 1e-10, "expected J_z=0 for dielectric, got {}", v);
        }
    }

    /// Conduction current: J = σ·E_z,background.
    ///
    /// Proof: J_cond = σ·E_bg = 2.0·3.0 = 6.0 A/m².
    #[cfg(feature = "pinn")]
    #[test]
    fn test_current_density_conduction_proportional_to_sigma_and_ez() {
        type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
        use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
        use burn::tensor::Tensor;
        use std::collections::HashMap;

        let sigma = 2.0_f64;
        let e_z_bg = 3.0_f64;
        let expected = (sigma * e_z_bg) as f32;

        let mut domain = HashMap::new();
        domain.insert("conductivity".to_string(), sigma);
        domain.insert("e_z_background".to_string(), e_z_bg);
        let params = PhysicsParameters {
            material_properties: HashMap::new(),
            boundary_values: HashMap::new(),
            initial_values: HashMap::new(),
            domain_params: domain,
        };
        let x: Tensor<B, 2> = Tensor::zeros([3, 1], &Default::default());
        let y: Tensor<B, 2> = Tensor::zeros([3, 1], &Default::default());

        let jz = super::compute_current_density_z::<B>(&x, &y, &params);
        let jz_data: Vec<f32> = jz.into_data().to_vec().unwrap();
        for v in &jz_data {
            let diff = (v - expected).abs();
            assert!(diff < 1e-4, "expected J_z=σ·E_z={}, got {}", expected, v);
        }
    }

    /// Uniform impressed current: all output elements equal current_density_z.
    #[cfg(feature = "pinn")]
    #[test]
    fn test_current_density_uniform_impressed() {
        type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
        use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
        use burn::tensor::Tensor;
        use std::collections::HashMap;

        let j0 = 5.0_f64;
        let mut domain = HashMap::new();
        domain.insert("current_density_z".to_string(), j0);
        let params = PhysicsParameters {
            material_properties: HashMap::new(),
            boundary_values: HashMap::new(),
            initial_values: HashMap::new(),
            domain_params: domain,
        };
        let x: Tensor<B, 2> = Tensor::zeros([4, 1], &Default::default());
        let y: Tensor<B, 2> = Tensor::zeros([4, 1], &Default::default());

        let jz = super::compute_current_density_z::<B>(&x, &y, &params);
        let jz_data: Vec<f32> = jz.into_data().to_vec().unwrap();
        for v in &jz_data {
            let diff = (v - j0 as f32).abs();
            assert!(diff < 1e-4, "expected J_z={}, got {}", j0, v);
        }
    }
}
