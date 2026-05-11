//! Tests for PDE residual computations (requires `pinn` feature).

#[cfg(feature = "pinn")]
use super::*;
#[cfg(feature = "pinn")]
use burn::tensor::Tensor;

#[cfg(feature = "pinn")]
#[test]
fn test_strain_computation_mathematical_properties() -> crate::core::error::KwaversResult<()> {
    use burn::backend::{Autodiff, NdArray};

    type B = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let alpha = 2.5_f32;
    let dudx = Tensor::<B, 2>::from_floats([[alpha]], &device);
    let dudy = Tensor::<B, 2>::from_floats([[0.0_f32]], &device);
    let dvdx = Tensor::<B, 2>::from_floats([[0.0_f32]], &device);
    let dvdy = Tensor::<B, 2>::from_floats([[0.0_f32]], &device);

    let (eps_xx, eps_yy, eps_xy) = compute_strain_from_gradients(dudx, dudy, dvdx, dvdy);

    let eps_xx = eps_xx.into_data().as_slice::<f32>().unwrap()[0];
    let eps_yy = eps_yy.into_data().as_slice::<f32>().unwrap()[0];
    let eps_xy = eps_xy.into_data().as_slice::<f32>().unwrap()[0];

    assert!((eps_xx - alpha).abs() < 1e-6);
    assert!(eps_yy.abs() < 1e-6);
    assert!(eps_xy.abs() < 1e-6);
    Ok(())
}

#[cfg(feature = "pinn")]
#[test]
fn test_hookes_law_isotropic() -> crate::core::error::KwaversResult<()> {
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::Tensor;

    type B = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let lambda = 5.0_f64;
    let mu = 3.0_f64;
    let gamma = 1.2_f32;

    let eps_xx = Tensor::<B, 2>::from_floats([[0.0_f32]], &device);
    let eps_yy = Tensor::<B, 2>::from_floats([[0.0_f32]], &device);
    let eps_xy = Tensor::<B, 2>::from_floats([[gamma]], &device);

    let (sigma_xx, sigma_yy, sigma_xy) =
        compute_stress_from_strain(eps_xx, eps_yy, eps_xy, lambda, mu);

    let sigma_xx = sigma_xx.into_data().as_slice::<f32>().unwrap()[0];
    let sigma_yy = sigma_yy.into_data().as_slice::<f32>().unwrap()[0];
    let sigma_xy = sigma_xy.into_data().as_slice::<f32>().unwrap()[0];

    assert!(sigma_xx.abs() < 1e-6);
    assert!(sigma_yy.abs() < 1e-6);
    assert!((sigma_xy - (2.0_f32 * mu as f32 * gamma)).abs() < 1e-6);
    Ok(())
}

#[cfg(feature = "pinn")]
#[test]
fn test_stress_divergence_equilibrium() -> crate::core::error::KwaversResult<()> {
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::Tensor;

    type B = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let x = Tensor::<B, 2>::from_floats([[0.1_f32]], &device).require_grad();
    let y = Tensor::<B, 2>::from_floats([[0.3_f32]], &device).require_grad();

    let sigma_xx = x.clone().mul_scalar(2.0);
    let sigma_yy = x.clone().mul_scalar(3.0);
    let sigma_xy = y.clone().mul_scalar(-2.0);

    let (div_x, div_y) = compute_stress_divergence(sigma_xx, sigma_xy, sigma_yy, x, y);

    let div_x_data = div_x.into_data();
    let div_x = div_x_data.as_slice::<f32>().unwrap();

    let div_y_data = div_y.into_data();
    let div_y = div_y_data.as_slice::<f32>().unwrap();

    for &v in div_x.iter() {
        assert!(v.abs() < 1e-6);
    }
    for &v in div_y.iter() {
        assert!(v.abs() < 1e-6);
    }
    Ok(())
}
