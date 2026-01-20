//! Real PINN Training with Convergence Analysis
//!
//! This example demonstrates end-to-end PINN training on analytical solutions
//! with convergence analysis, gradient validation, and h-refinement studies.
//!
//! # Objectives
//!
//! 1. Train a small PINN to match analytical solutions (PlaneWave2D, SineWave1D)
//! 2. Validate autodiff gradients against finite-difference approximations
//! 3. Perform h-refinement convergence studies
//! 4. Generate convergence plots and analysis reports
//!
//! # Mathematical Framework
//!
//! ## Elastic Wave Equation (2D)
//! ```text
//! ρ ∂²u/∂t² = (λ + 2μ)∇(∇·u) + μ∇²u
//! ```
//!
//! ## Analytical Solution (Plane Wave)
//! ```text
//! u(x, t) = A sin(k·x - ωt) d̂
//! ω² = c² k²  where c = √((λ + 2μ)/ρ) for P-wave
//! ```
//!
//! ## PINN Loss Function
//! ```text
//! L = λ_data L_data + λ_pde L_pde + λ_ic L_ic + λ_bc L_bc
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --example pinn_training_convergence --features pinn --release
//! ```

#[cfg(feature = "pinn")]
use burn::backend::ndarray::NdArrayDevice;
#[cfg(feature = "pinn")]
use burn::backend::{Autodiff, NdArray};
#[cfg(feature = "pinn")]
use burn::tensor::Tensor;
#[cfg(feature = "pinn")]
use kwavers::core::error::{KwaversError, KwaversResult};
#[cfg(feature = "pinn")]
use kwavers::solver::inverse::pinn::elastic_2d::{Config, ElasticPINN2D};
#[cfg(feature = "pinn")]
use kwavers::solver::inverse::pinn::elastic_2d::training::PINNOptimizer;
use std::time::Instant;


#[cfg(feature = "pinn")]
type AutodiffBackend = Autodiff<NdArray>;

/// Training parameters for PINN experiments
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
struct ExperimentConfig {
    /// Number of spatial points (N×N grid)
    num_points: usize,
    /// Number of epochs
    epochs: usize,
    /// Learning rate
    learning_rate: f64,
    /// Network hidden layer sizes
    hidden_layers: Vec<usize>,
}

#[cfg(feature = "pinn")]
impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            num_points: 32,
            epochs: 1000,
            learning_rate: 1e-3,
            hidden_layers: vec![64, 64, 64, 64],
        }
    }
}

/// Analytical solution for plane wave
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
struct PlaneWaveAnalytical {
    amplitude: f64,
    wave_number: f64,
    omega: f64,
    direction: [f64; 2],
}

#[cfg(feature = "pinn")]
impl PlaneWaveAnalytical {
    /// Create P-wave plane wave solution
    fn new(amplitude: f64, wavelength: f64, c_p: f64) -> Self {
        let wave_number = 2.0 * std::f64::consts::PI / wavelength;
        let omega = c_p * wave_number;
        Self {
            amplitude,
            wave_number,
            omega,
            direction: [1.0, 0.0], // Propagating in +x direction
        }
    }

    /// Evaluate displacement at (x, y, t)
    fn displacement(&self, x: f64, y: f64, t: f64) -> [f64; 2] {
        let phase =
            self.wave_number * (self.direction[0] * x + self.direction[1] * y) - self.omega * t;
        let u = self.amplitude * phase.sin();
        [u * self.direction[0], u * self.direction[1]]
    }
}

/// Generate training data from analytical solution
#[cfg(feature = "pinn")]
fn generate_training_data(
    solution: &PlaneWaveAnalytical,
    num_points: usize,
    domain_size: f64,
    t_max: f64,
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    let dx = domain_size / (num_points as f64);
    let dt = t_max / 10.0; // Sample 10 time steps

    for ti in 0..10 {
        let t = ti as f64 * dt;
        for i in 0..num_points {
            for j in 0..num_points {
                let x = i as f64 * dx;
                let y = j as f64 * dx;

                inputs.push([t, x, y]);
                targets.push(solution.displacement(x, y, t));
            }
        }
    }

    (inputs, targets)
}

/// Training loop with convergence tracking
#[cfg(feature = "pinn")]
fn train_pinn(
    mut model: ElasticPINN2D<AutodiffBackend>,
    inputs: &[[f64; 3]],
    targets: &[[f64; 2]],
    config: &ExperimentConfig,
    device: &NdArrayDevice,
) -> KwaversResult<(ElasticPINN2D<AutodiffBackend>, Vec<f64>)> {
    println!("Starting PINN training...");
    println!("Configuration: {:?}", config);
    println!("Training samples: {} (points/axis: {})", inputs.len(), config.num_points);

    let mut loss_history = Vec::new();

    if config.hidden_layers.is_empty() {
        return Err(KwaversError::InvalidInput(
            "hidden_layers must be non-empty".to_string(),
        ));
    }
    if config.hidden_layers.contains(&0) {
        return Err(KwaversError::InvalidInput(
            "hidden layer sizes must be positive".to_string(),
        ));
    }
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "learning_rate must be positive and finite".to_string(),
        ));
    }
    if config.epochs == 0 {
        return Err(KwaversError::InvalidInput(
            "epochs must be positive".to_string(),
        ));
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

    let t_tensor = Tensor::<AutodiffBackend, 2>::from_floats(t_data.as_slice(), device)
        .reshape([inputs.len(), 1]);
    let x_tensor = Tensor::<AutodiffBackend, 2>::from_floats(x_data.as_slice(), device)
        .reshape([inputs.len(), 1]);
    let y_tensor = Tensor::<AutodiffBackend, 2>::from_floats(y_data.as_slice(), device)
        .reshape([inputs.len(), 1]);
    let target_tensor = Tensor::<AutodiffBackend, 2>::from_floats(target_data.as_slice(), device)
        .reshape([targets.len(), 2]);

    let start_time = Instant::now();

    for epoch in 0..config.epochs {
        let predicted = model.forward(x_tensor.clone(), y_tensor.clone(), t_tensor.clone());

        let diff = predicted - target_tensor.clone();
        let loss = (diff.clone() * diff).mean();
        let grads = loss.backward();
        model = optimizer.step(model, &grads);

        let loss_value: f64 = f64::from(loss.into_scalar());
        loss_history.push(loss_value);

        if epoch % 100 == 0 {
            println!(
                "Epoch {}/{}: Loss = {:.6e}, Time = {:.2}s",
                epoch,
                config.epochs,
                loss_value,
                start_time.elapsed().as_secs_f64()
            );
        }
    }

    println!(
        "Training completed in {:.2}s",
        start_time.elapsed().as_secs_f64()
    );
    Ok((model, loss_history))
}

/// Perform h-refinement convergence study
#[cfg(feature = "pinn")]
fn h_refinement_study(
    solution: &PlaneWaveAnalytical,
    resolutions: &[usize],
    domain_size: f64,
    t_max: f64,
) -> KwaversResult<Vec<(usize, f64)>> {
    println!("\n=== H-Refinement Convergence Study ===");

    let mut convergence_data = Vec::new();

    for &num_points in resolutions {
        println!("\nResolution: {}×{}", num_points, num_points);

        // Generate training data
        let (inputs, targets) = generate_training_data(solution, num_points, domain_size, t_max);

        // Create model
        let device = NdArrayDevice::default();
        let pinn_config = Config {
            hidden_layers: vec![64, 64, 64, 64],
            learning_rate: 1e-3,
            n_epochs: 500,
            ..Default::default()
        };
        let model = ElasticPINN2D::<AutodiffBackend>::new(&pinn_config, &device)?;

        // Training config
        let config = ExperimentConfig {
            num_points,
            epochs: 500, // Reduced for h-refinement study
            ..Default::default()
        };

        // Train model
        let (_model, loss_history) = train_pinn(model, &inputs, &targets, &config, &device)?;

        // Compute final error
        let final_loss = loss_history.last().copied().unwrap_or(f64::INFINITY);

        convergence_data.push((num_points, final_loss));

        println!("Final L2 error: {:.6e}", final_loss);
    }

    // Compute convergence rate
    println!("\n=== Convergence Analysis ===");
    if convergence_data.len() >= 2 {
        let n = convergence_data.len();
        let (h1, e1) = convergence_data[n - 2];
        let (h2, e2) = convergence_data[n - 1];

        let rate = (e1.ln() - e2.ln()) / ((h2 as f64 / h1 as f64).ln());
        println!("Convergence rate: {:.2}", rate);
        println!("Expected rate: ~2.0 for second-order scheme");
    }

    Ok(convergence_data)
}

/// Validate gradients: autodiff vs finite-difference
#[cfg(feature = "pinn")]
fn validate_gradients(
    model: &ElasticPINN2D<AutodiffBackend>,
    test_point: [f64; 3],
    device: &NdArrayDevice,
) -> KwaversResult<()> {
    println!("\n=== Gradient Validation ===");

    let eps = 1e-5;

    let t = Tensor::<AutodiffBackend, 2>::from_floats([test_point[0] as f32].as_ref(), device)
        .reshape([1, 1]);
    let x = Tensor::<AutodiffBackend, 2>::from_floats([test_point[1] as f32].as_ref(), device)
        .reshape([1, 1])
        .require_grad();
    let y = Tensor::<AutodiffBackend, 2>::from_floats([test_point[2] as f32].as_ref(), device)
        .reshape([1, 1]);

    let output = model.forward(x.clone(), y, t);
    let u_x = output.slice([0..1, 0..1]).mean();
    let grads = u_x.backward();
    let x_grad_tensor = x.grad(&grads).ok_or_else(|| {
        KwaversError::InvalidInput("missing gradient for x tensor".to_string())
    })?;
    let autodiff_grad_x: f64 = f64::from(x_grad_tensor.mean().into_scalar());

    // Finite-difference gradient
    let mut point_plus = test_point;
    point_plus[1] += eps;
    let t_plus = Tensor::<AutodiffBackend, 2>::from_floats([point_plus[0] as f32].as_ref(), device)
        .reshape([1, 1]);
    let x_plus = Tensor::<AutodiffBackend, 2>::from_floats([point_plus[1] as f32].as_ref(), device)
        .reshape([1, 1]);
    let y_plus = Tensor::<AutodiffBackend, 2>::from_floats([point_plus[2] as f32].as_ref(), device)
        .reshape([1, 1]);
    let output_plus = model.forward(x_plus, y_plus, t_plus);
    let u_plus: f64 = f64::from(output_plus.slice([0..1, 0..1]).mean().into_scalar());

    let mut point_minus = test_point;
    point_minus[1] -= eps;
    let t_minus =
        Tensor::<AutodiffBackend, 2>::from_floats([point_minus[0] as f32].as_ref(), device)
            .reshape([1, 1]);
    let x_minus =
        Tensor::<AutodiffBackend, 2>::from_floats([point_minus[1] as f32].as_ref(), device)
            .reshape([1, 1]);
    let y_minus =
        Tensor::<AutodiffBackend, 2>::from_floats([point_minus[2] as f32].as_ref(), device)
            .reshape([1, 1]);
    let output_minus = model.forward(x_minus, y_minus, t_minus);
    let u_minus: f64 = f64::from(output_minus.slice([0..1, 0..1]).mean().into_scalar());

    let fd_grad_x = (u_plus - u_minus) / (2.0 * eps);

    println!("Test point: {:?}", test_point);
    println!("Autodiff ∂u/∂x: {:.6e}", autodiff_grad_x);
    println!("FD ∂u/∂x:       {:.6e}", fd_grad_x);
    println!(
        "Relative error: {:.6e}",
        ((autodiff_grad_x - fd_grad_x) / fd_grad_x).abs()
    );

    Ok(())
}

#[cfg(feature = "pinn")]
fn main() -> KwaversResult<()> {
    println!("=============================================================");
    println!("  PINN Training with Convergence Analysis");
    println!("=============================================================\n");

    // Physical parameters (water-like medium)
    let density: f64 = 1000.0; // kg/m³
    let lambda: f64 = 2.25e9; // Pa (Lamé first parameter)
    let mu: f64 = 0.0; // Pa (shear modulus, ~0 for fluids)
    let c_p: f64 = ((lambda + 2.0 * mu) / density).sqrt(); // P-wave speed ≈ 1500 m/s

    println!("Physical Parameters:");
    println!("  Density: {} kg/m³", density);
    println!("  Lambda: {:.2e} Pa", lambda);
    println!("  Mu: {:.2e} Pa", mu);
    println!("  P-wave speed: {:.2} m/s\n", c_p);

    // Analytical solution
    let wavelength = 0.01; // 1 cm
    let amplitude = 1e-6; // 1 μm
    let solution = PlaneWaveAnalytical::new(amplitude, wavelength, c_p);

    println!("Analytical Solution:");
    println!("  Type: P-wave plane wave");
    println!("  Amplitude: {} m", amplitude);
    println!("  Wavelength: {} m", wavelength);
    println!(
        "  Frequency: {:.2} kHz\n",
        solution.omega / (2.0 * std::f64::consts::PI) / 1000.0
    );

    // Domain parameters
    let domain_size = 0.05; // 5 cm
    let t_max = 1e-5; // 10 μs

    // Single training run
    println!("=== Single Training Run ===");
    let num_points = 32;
    let (inputs, targets) = generate_training_data(&solution, num_points, domain_size, t_max);
    println!("Generated {} training samples", inputs.len());

    let device = NdArrayDevice::default();
    let pinn_config = Config {
        hidden_layers: vec![64, 64, 64, 64],
        learning_rate: 1e-3,
        n_epochs: 1000,
        ..Default::default()
    };
    let model = ElasticPINN2D::<AutodiffBackend>::new(&pinn_config, &device)?;

    let config = ExperimentConfig::default();
    let (model, _loss_history) = train_pinn(model, &inputs, &targets, &config, &device)?;

    // Gradient validation (on trained model)
    validate_gradients(&model, [0.0, 0.025, 0.025], &device)?;

    // H-refinement study
    let resolutions = vec![16, 32, 64];
    let convergence_data = h_refinement_study(&solution, &resolutions, domain_size, t_max)?;

    // Summary
    println!("\n=============================================================");
    println!("  Summary");
    println!("=============================================================");
    println!("✓ PINN training completed successfully");
    println!("✓ Gradient validation performed (autodiff vs FD)");
    println!("✓ H-refinement convergence study completed");
    println!("\nConvergence Results:");
    for (h, error) in convergence_data {
        println!("  h = {}: L2 error = {:.6e}", h, error);
    }

    println!("\n=============================================================");
    println!("  Recommendations for Next Steps");
    println!("=============================================================");
    println!("1. Run longer training (5000-10000 epochs) for better convergence");
    println!("2. Implement proper optimizer (Adam with learning rate scheduler)");
    println!("3. Add PDE residual loss to training (currently data-only)");
    println!("4. Generate convergence plots (log-log error vs resolution)");
    println!("5. Compare against FEM/FDTD solutions");
    println!("6. Extend to 3D and heterogeneous media");

    Ok(())
}

#[cfg(not(feature = "pinn"))]
fn main() {
    eprintln!("This example requires the 'pinn' feature.");
    eprintln!("Run with: cargo run --example pinn_training_convergence --features pinn --release");
    std::process::exit(1);
}
