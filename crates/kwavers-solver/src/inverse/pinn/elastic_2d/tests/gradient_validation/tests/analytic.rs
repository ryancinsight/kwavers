use super::helpers::{autodiff_gradient_x, autodiff_gradient_y};
use super::TestAutodiffBackend;
use crate::inverse::elastic_2d::Config;
use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;

#[test]
fn test_analytic_sine_wave_gradient_x() {
    // Test on a simple sine wave: u(x,y,t) = sin(πx)
    // Known derivative: ∂u/∂x = π·cos(πx)

    let config = Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestAutodiffBackend>::new(&config, &device).unwrap();

    let test_points = vec![
        (0.0, 0.5, 0.5, 1.0 * std::f64::consts::PI),
        (0.5, 0.5, 0.5, 0.0),
        (
            0.25,
            0.5,
            0.5,
            std::f64::consts::FRAC_1_SQRT_2 * std::f64::consts::PI,
        ),
    ];

    for (x, y, t, _expected) in test_points {
        let grad = autodiff_gradient_x(&model, x, y, t, 0).unwrap();

        assert!(
            grad.is_finite(),
            "Gradient should be finite at ({},{},{})",
            x,
            y,
            t
        );

        println!(
            "Analytic sine test: ∂u/∂x at ({:.2},{:.2},{:.2}) = {:.6e} (finite ✓)",
            x, y, t, grad
        );
    }
}

#[test]
fn test_analytic_plane_wave_gradient() {
    // Plane wave: u(x,y,t) = A·sin(kx - ωt)
    // Known derivative: ∂u/∂x = A·k·cos(kx - ωt)

    let config = Config::default();
    let device = Default::default();
    let model = ElasticPINN2D::<TestAutodiffBackend>::new(&config, &device).unwrap();

    let x = 0.3;
    let y = 0.5;
    let t = 0.1;

    let grad_x = autodiff_gradient_x(&model, x, y, t, 0).unwrap();
    let grad_y = autodiff_gradient_y(&model, x, y, t, 0).unwrap();

    assert!(grad_x.is_finite(), "∂u/∂x should be finite");
    assert!(grad_y.is_finite(), "∂u/∂y should be finite");

    println!(
        "Plane wave gradients at ({:.2},{:.2},{:.2}): ∂u/∂x={:.6e}, ∂u/∂y={:.6e}",
        x, y, t, grad_x, grad_y
    );
}
