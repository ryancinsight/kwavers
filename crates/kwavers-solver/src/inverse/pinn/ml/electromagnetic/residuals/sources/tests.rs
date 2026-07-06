// -----------------------------------------------------------------------
// Tests for compute_charge_density
// -----------------------------------------------------------------------

/// Source-free dielectric bulk: ρ_free = 0.
///
/// Proof: charge_density not set in domain_params → rho_0 = 0 → return zeros.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[cfg(feature = "pinn")]
#[test]
fn test_charge_density_zero_for_source_free_medium() {
    type B = coeus_core::MoiraiBackend;
    use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
    use coeus_autograd::Var;
    use std::collections::HashMap;

    let params = PinnDomainPhysicsParameters {
        material_properties: HashMap::new(),
        boundary_values: HashMap::new(),
        initial_values: HashMap::new(),
        domain_params: HashMap::new(),
    };
    let backend = B::default();
    let x: Var<f32, B> = Var::new(coeus_tensor::Tensor::zeros_on(vec![4, 1], &backend), false);
    let y: Var<f32, B> = Var::new(coeus_tensor::Tensor::zeros_on(vec![4, 1], &backend), false);

    let rho = super::compute_charge_density::<B>(&x, &y, &params);
    let rho_data = rho.tensor.as_slice();
    for v in rho_data {
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
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[cfg(feature = "pinn")]
#[test]
fn test_charge_density_uniform_matches_param() {
    type B = coeus_core::MoiraiBackend;
    use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
    use coeus_autograd::Var;
    use std::collections::HashMap;

    let rho_expected = 1.5e-3_f64;
    let mut domain = HashMap::new();
    domain.insert("charge_density".to_string(), rho_expected);
    let params = PinnDomainPhysicsParameters {
        material_properties: HashMap::new(),
        boundary_values: HashMap::new(),
        initial_values: HashMap::new(),
        domain_params: domain,
    };
    let backend = B::default();
    let x: Var<f32, B> = Var::new(coeus_tensor::Tensor::zeros_on(vec![3, 1], &backend), false);
    let y: Var<f32, B> = Var::new(coeus_tensor::Tensor::zeros_on(vec![3, 1], &backend), false);

    let rho = super::compute_charge_density::<B>(&x, &y, &params);
    let rho_data = rho.tensor.as_slice();
    for v in rho_data {
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
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[cfg(feature = "pinn")]
#[test]
fn test_charge_density_gaussian_peak_at_centre() {
    type B = coeus_core::MoiraiBackend;
    use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
    use coeus_autograd::Var;
    use std::collections::HashMap;

    let rho_0 = 2.0_f64;
    let mut domain = HashMap::new();
    domain.insert("charge_density".to_string(), rho_0);
    domain.insert("charge_x0".to_string(), 0.5_f64);
    domain.insert("charge_y0".to_string(), 0.5_f64);
    domain.insert("charge_sigma".to_string(), 0.1_f64);
    let params = PinnDomainPhysicsParameters {
        material_properties: HashMap::new(),
        boundary_values: HashMap::new(),
        initial_values: HashMap::new(),
        domain_params: domain,
    };
    // Single point exactly at the Gaussian centre → exp(0) = 1 → ρ = ρ₀
    let backend = B::default();
    let x: Var<f32, B> = Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![1, 1], &[0.5_f32], &backend),
        false,
    );
    let y: Var<f32, B> = Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![1, 1], &[0.5_f32], &backend),
        false,
    );

    let rho = super::compute_charge_density::<B>(&x, &y, &params);
    let rho_data = rho.tensor.as_slice();
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
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[cfg(feature = "pinn")]
#[test]
fn test_current_density_zero_for_dielectric() {
    type B = coeus_core::MoiraiBackend;
    use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
    use coeus_autograd::Var;
    use std::collections::HashMap;

    let params = PinnDomainPhysicsParameters {
        material_properties: HashMap::new(),
        boundary_values: HashMap::new(),
        initial_values: HashMap::new(),
        domain_params: HashMap::new(),
    };
    let backend = B::default();
    let x: Var<f32, B> = Var::new(coeus_tensor::Tensor::zeros_on(vec![5, 1], &backend), false);
    let y: Var<f32, B> = Var::new(coeus_tensor::Tensor::zeros_on(vec![5, 1], &backend), false);

    let jz = super::compute_current_density_z::<B>(&x, &y, &params);
    let jz_data = jz.tensor.as_slice();
    for v in jz_data {
        assert!(v.abs() < 1e-10, "expected J_z=0 for dielectric, got {}", v);
    }
}

/// Conduction current: J = σ·E_z,background.
///
/// Proof: J_cond = σ·E_bg = 2.0·3.0 = 6.0 A/m².
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[cfg(feature = "pinn")]
#[test]
fn test_current_density_conduction_proportional_to_sigma_and_ez() {
    type B = coeus_core::MoiraiBackend;
    use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
    use coeus_autograd::Var;
    use std::collections::HashMap;

    let sigma = 2.0_f64;
    let e_z_bg = 3.0_f64;
    let expected = (sigma * e_z_bg) as f32;

    let mut domain = HashMap::new();
    domain.insert("conductivity".to_string(), sigma);
    domain.insert("e_z_background".to_string(), e_z_bg);
    let params = PinnDomainPhysicsParameters {
        material_properties: HashMap::new(),
        boundary_values: HashMap::new(),
        initial_values: HashMap::new(),
        domain_params: domain,
    };
    let backend = B::default();
    let x: Var<f32, B> = Var::new(coeus_tensor::Tensor::zeros_on(vec![3, 1], &backend), false);
    let y: Var<f32, B> = Var::new(coeus_tensor::Tensor::zeros_on(vec![3, 1], &backend), false);

    let jz = super::compute_current_density_z::<B>(&x, &y, &params);
    let jz_data = jz.tensor.as_slice();
    for v in jz_data {
        let diff = (v - expected).abs();
        assert!(diff < 1e-4, "expected J_z=σ·E_z={}, got {}", expected, v);
    }
}

/// Uniform impressed current: all output elements equal current_density_z.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[cfg(feature = "pinn")]
#[test]
fn test_current_density_uniform_impressed() {
    type B = coeus_core::MoiraiBackend;
    use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
    use coeus_autograd::Var;
    use std::collections::HashMap;

    let j0 = 5.0_f64;
    let mut domain = HashMap::new();
    domain.insert("current_density_z".to_string(), j0);
    let params = PinnDomainPhysicsParameters {
        material_properties: HashMap::new(),
        boundary_values: HashMap::new(),
        initial_values: HashMap::new(),
        domain_params: domain,
    };
    let backend = B::default();
    let x: Var<f32, B> = Var::new(coeus_tensor::Tensor::zeros_on(vec![4, 1], &backend), false);
    let y: Var<f32, B> = Var::new(coeus_tensor::Tensor::zeros_on(vec![4, 1], &backend), false);

    let jz = super::compute_current_density_z::<B>(&x, &y, &params);
    let jz_data = jz.tensor.as_slice();
    for v in jz_data {
        let diff = (v - j0 as f32).abs();
        assert!(diff < 1e-4, "expected J_z={}, got {}", j0, v);
    }
}
