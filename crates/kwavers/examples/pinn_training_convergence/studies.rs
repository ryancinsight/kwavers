//! PINN training loop, h-refinement study, and gradient validation for the convergence study.

use super::analytical_data::{generate_training_data, ExperimentConfig, PlaneWaveAnalytical};
use super::AutodiffBackend;
use coeus_autograd::{mean, mul, sub, Var};
use coeus_tensor::Tensor;
use kwavers_core::error::KwaversError;
use kwavers_solver::inverse::pinn::elastic_2d::training::optimizer::PINNOptimizer;
use kwavers_solver::inverse::pinn::elastic_2d::{Config, ElasticPINN2D};
use std::error::Error;
use std::io::Write;
use std::time::Instant;

/// Training loop with convergence tracking
pub(crate) fn train_pinn(
    mut model: ElasticPINN2D<AutodiffBackend>,
    inputs: &[[f64; 3]],
    targets: &[[f64; 2]],
    config: &ExperimentConfig,
) -> Result<(ElasticPINN2D<AutodiffBackend>, Vec<f64>), Box<dyn Error>> {
    let _ = writeln!(std::io::stdout().lock(), "Starting PINN training...");
    let _ = writeln!(std::io::stdout().lock(), "Configuration: {:?}", config);
    let _ = writeln!(
        std::io::stdout().lock(),
        "Training samples: {} (points/axis: {})",
        inputs.len(),
        config.num_points
    );

    let mut loss_history = Vec::new();

    if config.hidden_layers.is_empty() {
        return Err(
            KwaversError::InvalidInput("hidden_layers must be non-empty".to_string()).into(),
        );
    }
    if config.hidden_layers.contains(&0) {
        return Err(
            KwaversError::InvalidInput("hidden layer sizes must be positive".to_string()).into(),
        );
    }
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "learning_rate must be positive and finite".to_string(),
        )
        .into());
    }
    if config.epochs == 0 {
        return Err(KwaversError::InvalidInput("epochs must be positive".to_string()).into());
    }

    let mut optimizer = PINNOptimizer::adam(&model, config.learning_rate, 0.0, 0.9, 0.999, 1e-8);

    let mut t_data: Vec<f32> = Vec::with_capacity(inputs.len());
    let mut x_data: Vec<f32> = Vec::with_capacity(inputs.len());
    let mut y_data: Vec<f32> = Vec::with_capacity(inputs.len());
    for input in inputs {
        t_data.push(input[0] as f32);
        x_data.push(input[1] as f32);
        y_data.push(input[2] as f32);
    }

    let target_data: Vec<f32> = targets
        .iter()
        .flat_map(|u| [u[0] as f32, u[1] as f32])
        .collect();

    let backend = AutodiffBackend::default();
    let t_tensor = Var::new(
        Tensor::from_slice_on(vec![inputs.len(), 1], t_data.as_slice(), &backend),
        false,
    );
    let x_tensor = Var::new(
        Tensor::from_slice_on(vec![inputs.len(), 1], x_data.as_slice(), &backend),
        false,
    );
    let y_tensor = Var::new(
        Tensor::from_slice_on(vec![inputs.len(), 1], y_data.as_slice(), &backend),
        false,
    );
    let target_tensor = Var::new(
        Tensor::from_slice_on(vec![targets.len(), 2], target_data.as_slice(), &backend),
        false,
    );

    let start_time = Instant::now();

    for epoch in 0..config.epochs {
        let predicted = model.forward(&x_tensor, &y_tensor, &t_tensor)?;
        let diff = sub(&predicted, &target_tensor);
        let sq = mul(&diff, &diff);
        let loss = mean(&sq);
        loss.backward()?;
        optimizer.step(&mut model)?;

        let loss_value = loss.tensor.as_slice()[0] as f64;
        loss_history.push(loss_value);

        if epoch % 100 == 0 {
            let _ = writeln!(
                std::io::stdout().lock(),
                "Epoch {}/{}: Loss = {:.6e}, Time = {:.2}s",
                epoch,
                config.epochs,
                loss_value,
                start_time.elapsed().as_secs_f64()
            );
        }
    }

    let _ = writeln!(
        std::io::stdout().lock(),
        "Training completed in {:.2}s",
        start_time.elapsed().as_secs_f64()
    );
    Ok((model, loss_history))
}

/// Perform h-refinement convergence study
pub(crate) fn h_refinement_study(
    solution: &PlaneWaveAnalytical,
    resolutions: &[usize],
    domain_size: f64,
    t_max: f64,
) -> Result<Vec<(usize, f64)>, Box<dyn Error>> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n=== H-Refinement Convergence Study ==="
    );

    let mut convergence_data = Vec::new();

    for &num_points in resolutions {
        let _ = writeln!(
            std::io::stdout().lock(),
            "\nResolution: {}×{}",
            num_points,
            num_points
        );

        // Generate training data
        let (inputs, targets) = generate_training_data(solution, num_points, domain_size, t_max);

        // Create model
        let pinn_config = Config {
            hidden_layers: vec![64, 64, 64, 64],
            learning_rate: 1e-3,
            n_epochs: 500,
            ..Default::default()
        };
        let model = ElasticPINN2D::<AutodiffBackend>::new(&pinn_config)?;

        // Training config
        let config = ExperimentConfig {
            num_points,
            epochs: 500, // Reduced for h-refinement study
            ..Default::default()
        };

        // Train model
        let (_model, loss_history) = train_pinn(model, &inputs, &targets, &config)?;

        // Compute final error
        let final_loss = loss_history.last().copied().unwrap_or(f64::INFINITY);

        convergence_data.push((num_points, final_loss));

        let _ = writeln!(
            std::io::stdout().lock(),
            "Final L2 error: {:.6e}",
            final_loss
        );
    }

    // Compute convergence rate
    let _ = writeln!(std::io::stdout().lock(), "\n=== Convergence Analysis ===");
    if convergence_data.len() >= 2 {
        let n = convergence_data.len();
        let (h1, e1) = convergence_data[n - 2];
        let (h2, e2) = convergence_data[n - 1];

        let rate = (e1.ln() - e2.ln()) / ((h2 as f64 / h1 as f64).ln());
        let _ = writeln!(std::io::stdout().lock(), "Convergence rate: {:.2}", rate);
        let _ = writeln!(
            std::io::stdout().lock(),
            "Expected rate: ~2.0 for second-order scheme"
        );
    }

    Ok(convergence_data)
}

/// Validate gradients: autodiff vs finite-difference
pub(crate) fn validate_gradients(
    model: &ElasticPINN2D<AutodiffBackend>,
    test_point: [f64; 3],
) -> Result<(), Box<dyn Error>> {
    let _ = writeln!(std::io::stdout().lock(), "\n=== Gradient Validation ===");

    let eps = 1e-5;
    let backend = AutodiffBackend::default();

    let t = Var::new(
        Tensor::from_slice_on(vec![1, 1], [test_point[0] as f32].as_ref(), &backend),
        false,
    );
    let x = Var::new(
        Tensor::from_slice_on(vec![1, 1], [test_point[1] as f32].as_ref(), &backend),
        true,
    );
    let y = Var::new(
        Tensor::from_slice_on(vec![1, 1], [test_point[2] as f32].as_ref(), &backend),
        false,
    );

    let output = model.forward(&x, &y, &t)?;
    let u_x = mean(&output);
    u_x.backward()?;
    let x_grad_tensor = x
        .grad()
        .ok_or_else(|| KwaversError::InvalidInput("missing gradient for x tensor".to_string()))?;
    let autodiff_grad_x = x_grad_tensor.as_slice()[0] as f64;

    // Finite-difference gradient
    let mut point_plus = test_point;
    point_plus[1] += eps;
    let t_plus = Var::new(
        Tensor::from_slice_on(vec![1, 1], [point_plus[0] as f32].as_ref(), &backend),
        false,
    );
    let x_plus = Var::new(
        Tensor::from_slice_on(vec![1, 1], [point_plus[1] as f32].as_ref(), &backend),
        false,
    );
    let y_plus = Var::new(
        Tensor::from_slice_on(vec![1, 1], [point_plus[2] as f32].as_ref(), &backend),
        false,
    );
    let output_plus = model.forward(&x_plus, &y_plus, &t_plus)?;
    let u_plus = output_plus.tensor.as_slice()[0] as f64;

    let mut point_minus = test_point;
    point_minus[1] -= eps;
    let t_minus = Var::new(
        Tensor::from_slice_on(vec![1, 1], [point_minus[0] as f32].as_ref(), &backend),
        false,
    );
    let x_minus = Var::new(
        Tensor::from_slice_on(vec![1, 1], [point_minus[1] as f32].as_ref(), &backend),
        false,
    );
    let y_minus = Var::new(
        Tensor::from_slice_on(vec![1, 1], [point_minus[2] as f32].as_ref(), &backend),
        false,
    );
    let output_minus = model.forward(&x_minus, &y_minus, &t_minus)?;
    let u_minus = output_minus.tensor.as_slice()[0] as f64;

    let fd_grad_x = (u_plus - u_minus) / (2.0 * eps);

    let _ = writeln!(std::io::stdout().lock(), "Test point: {:?}", test_point);
    let _ = writeln!(
        std::io::stdout().lock(),
        "Autodiff ∂u/∂x: {:.6e}",
        autodiff_grad_x
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "FD ∂u/∂x:       {:.6e}",
        fd_grad_x
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Relative error: {:.6e}",
        ((autodiff_grad_x - fd_grad_x) / fd_grad_x).abs()
    );

    Ok(())
}
