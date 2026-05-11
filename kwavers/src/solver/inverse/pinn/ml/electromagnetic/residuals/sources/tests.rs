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
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
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
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
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
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
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
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
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
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
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
