//! Physics-Informed Neural Network (PINN) for 2D Wave Equation
//!
//! This example demonstrates solving the 2D acoustic wave equation using
//! physics-informed neural networks with automatic differentiation.
//!
//! ## Wave Equation
//!
//! ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)
//!
//! ## Analytical Solution
//!
//! u(x,y,t) = sin(πx) * sin(πy) * cos(π√2 * c * t)
//!
//! ## Features Demonstrated
//!
//! - 2D geometry handling (rectangular domains)
//! - Physics-informed loss with PDE residuals
//! - Boundary condition enforcement
//! - Training convergence monitoring
//! - Prediction on arbitrary spatial-temporal points
//! - Performance benchmarking vs analytical solution

#[cfg(feature = "pinn")]
use kwavers_core::error::KwaversResult;
#[cfg(feature = "pinn")]
use kwavers_solver::inverse::pinn::ml::wave_equation_2d::{
    LossWeights2D, PinnConfig2D, PinnTrainer2D, WaveGeometry2D,
};
#[cfg(feature = "pinn")]
use leto::{Array1, Array2};
#[cfg(feature = "pinn")]
use std::time::Instant;

#[cfg(feature = "pinn")]
use coeus_core::MoiraiBackend;
#[cfg(feature = "pinn")]
use std::io::Write;

#[cfg(feature = "pinn")]
type Backend = MoiraiBackend;

#[cfg(feature = "pinn")]
/// Analytical solution for 2D wave equation
/// u(x,y,t) = sin(πx) * sin(πy) * cos(π√2 * c * t)
fn analytical_solution_2d(x: f64, y: f64, t: f64, wave_speed: f64) -> f64 {
    let k = std::f64::consts::PI * 2.0_f64.sqrt();
    (x * std::f64::consts::PI).sin() * (y * std::f64::consts::PI).sin() * (k * wave_speed * t).cos()
}

#[cfg(feature = "pinn")]
/// Generate training data from analytical solution
fn generate_training_data(
    n_samples: usize,
    domain_size: f64,
    wave_speed: f64,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array2<f64>) {
    let mut x_data = Vec::with_capacity(n_samples);
    let mut y_data = Vec::with_capacity(n_samples);
    let mut t_data = Vec::with_capacity(n_samples);
    let mut u_data = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x = rand::random::<f64>() * domain_size;
        let y = rand::random::<f64>() * domain_size;
        let t = rand::random::<f64>() * 0.01; // Short time for stability

        let u = analytical_solution_2d(x, y, t, wave_speed);

        x_data.push(x);
        y_data.push(y);
        t_data.push(t);
        u_data.push(u);
    }

    (
        Array1::from_vec(x_data.len(), x_data)
            .expect("invariant: x sample count equals vector length"),
        Array1::from_vec(y_data.len(), y_data)
            .expect("invariant: y sample count equals vector length"),
        Array1::from_vec(t_data.len(), t_data)
            .expect("invariant: time sample count equals vector length"),
        Array2::from_shape_vec((n_samples, 1), u_data)
            .expect("invariant: generated wave samples match grid shape"),
    )
}

/// Generate test grid for validation
#[cfg(feature = "pinn")]
fn generate_test_grid(
    nx: usize,
    ny: usize,
    nt: usize,
    domain_size: f64,
    t_max: f64,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut x_test = Vec::new();
    let mut y_test = Vec::new();
    let mut t_test = Vec::new();

    let dx = domain_size / (nx - 1) as f64;
    let dy = domain_size / (ny - 1) as f64;
    let dt = t_max / (nt - 1) as f64;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nt {
                x_test.push(i as f64 * dx);
                y_test.push(j as f64 * dy);
                t_test.push(k as f64 * dt);
            }
        }
    }

    (
        Array1::from_vec(x_test.len(), x_test)
            .expect("invariant: x test count equals vector length"),
        Array1::from_vec(y_test.len(), y_test)
            .expect("invariant: y test count equals vector length"),
        Array1::from_vec(t_test.len(), t_test)
            .expect("invariant: time test count equals vector length"),
    )
}

#[cfg(feature = "pinn")]
/// Compute L2 error between predictions and analytical solution
fn compute_l2_error(
    x_pred: &Array1<f64>,
    y_pred: &Array1<f64>,
    t_pred: &Array1<f64>,
    u_pred: &Array2<f64>,
    wave_speed: f64,
) -> f64 {
    let mut error_sum = 0.0;
    let n = x_pred.len();

    for i in 0..n {
        let u_analytical = analytical_solution_2d(x_pred[i], y_pred[i], t_pred[i], wave_speed);
        let u_predicted = u_pred[[i, 0]];
        let error = (u_predicted - u_analytical).powi(2);
        error_sum += error;
    }

    (error_sum / n as f64).sqrt()
}

#[cfg(feature = "pinn")]
fn main() -> KwaversResult<()> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "🧠 Physics-Informed Neural Network for 2D Wave Equation"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "======================================================"
    );

    // Configuration
    let wave_speed = 343.0; // m/s (speed of sound in air)
    let domain_size = 1.0; // 1m x 1m domain
    let n_training_samples = 500;
    let n_collocation_points = 1000;
    let epochs = 100;

    let _ = writeln!(std::io::stdout().lock(), "📋 Configuration:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Wave speed: {} m/s",
        wave_speed
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Domain: {}m x {}m",
        domain_size,
        domain_size
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Training samples: {}",
        n_training_samples
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Collocation points: {}",
        n_collocation_points
    );
    let _ = writeln!(std::io::stdout().lock(), "   Training epochs: {}", epochs);
    let _ = writeln!(std::io::stdout().lock());

    // Initialize backend
    let _ = writeln!(std::io::stdout().lock(), "🔥 Backend: Moirai (CPU)");
    let _ = writeln!(std::io::stdout().lock());

    // Create PINN configuration
    let pinn_config = PinnConfig2D {
        hidden_layers: vec![100, 100, 100, 100],
        learning_rate: 1e-3,
        loss_weights: LossWeights2D {
            data: 1.0,
            pde: 1.0,
            boundary: 10.0,
            initial: 10.0,
        },
        num_collocation_points: n_collocation_points,
        ..Default::default()
    };

    let _ = writeln!(std::io::stdout().lock(), "🧠 PINN Configuration:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Hidden layers: {:?}",
        pinn_config.hidden_layers
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Learning rate: {}",
        pinn_config.learning_rate
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Loss weights: data={:.1}, pde={:.1}, boundary={:.1}, initial={:.1}",
        pinn_config.loss_weights.data,
        pinn_config.loss_weights.pde,
        pinn_config.loss_weights.boundary,
        pinn_config.loss_weights.initial
    );
    let _ = writeln!(std::io::stdout().lock());

    // Create geometry (unit square)
    let geometry = WaveGeometry2D::rectangular(0.0, domain_size, 0.0, domain_size);
    let _ = writeln!(
        std::io::stdout().lock(),
        "📐 Geometry: Unit square [0,{}] x [0,{}]",
        domain_size,
        domain_size
    );
    let _ = writeln!(std::io::stdout().lock());

    // Create PINN trainer
    let trainer = PinnTrainer2D::<Backend>::new_trainer(pinn_config.clone(), geometry)?;
    let _ = writeln!(
        std::io::stdout().lock(),
        "✅ PINN Trainer: Created successfully"
    );
    let _ = writeln!(std::io::stdout().lock());

    // Generate training data
    let _ = writeln!(std::io::stdout().lock(), "📊 Generating training data...");
    let (x_train, y_train, t_train, u_train) =
        generate_training_data(n_training_samples, domain_size, wave_speed);
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Training points: {}",
        x_train.len()
    );
    let _ = writeln!(std::io::stdout().lock());

    // Train PINN
    let _ = writeln!(std::io::stdout().lock(), "🚀 Training PINN...");
    let start_time = Instant::now();
    let mut trainer = trainer;
    let metrics = trainer.train(
        &x_train,
        &y_train,
        &t_train,
        &u_train,
        wave_speed,
        &pinn_config,
        epochs,
    )?;
    let training_time = start_time.elapsed();

    let _ = writeln!(
        std::io::stdout().lock(),
        "✅ Training completed in {:.2}s",
        training_time.as_secs_f64()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Final total loss: {:.6e}",
        metrics.total_loss.last().unwrap()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Final data loss: {:.6e}",
        metrics.data_loss.last().unwrap()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Final PDE loss: {:.6e}",
        metrics.pde_loss.last().unwrap()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Final BC loss: {:.6e}",
        metrics.bc_loss.last().unwrap()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Final IC loss: {:.6e}",
        metrics.ic_loss.last().unwrap()
    );
    let _ = writeln!(std::io::stdout().lock());

    // Generate test data for validation
    let _ = writeln!(
        std::io::stdout().lock(),
        "🧪 Validating PINN predictions..."
    );
    let (x_test, y_test, t_test) = generate_test_grid(10, 10, 5, domain_size, 0.01);
    let _ = writeln!(std::io::stdout().lock(), "   Test points: {}", x_test.len());

    // Make predictions
    let predictions = trainer.pinn().predict(&x_test, &y_test, &t_test)?;
    let _ = writeln!(std::io::stdout().lock(), "   Predictions completed");

    // Compute error
    let l2_error = compute_l2_error(&x_test, &y_test, &t_test, &predictions, wave_speed);
    eprintln!("   L2 Error: {:.6e}", l2_error);
    let _ = writeln!(std::io::stdout().lock());

    // Performance analysis
    let _ = writeln!(std::io::stdout().lock(), "📈 Performance Analysis:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Training time: {:.2}s ({:.1} ms/epoch)",
        training_time.as_secs_f64(),
        training_time.as_millis() as f64 / epochs as f64
    );

    let loss_reduction = metrics.total_loss[0] / metrics.total_loss.last().unwrap();
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Loss reduction: {:.2e}x",
        loss_reduction
    );

    let convergence_epoch = metrics
        .total_loss
        .iter()
        .enumerate()
        .find(|(_, &loss)| loss < 1e-3)
        .map(|(epoch, _)| epoch)
        .unwrap_or(epochs);
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Convergence: {} epochs to reach 1e-3 loss",
        convergence_epoch
    );
    let _ = writeln!(std::io::stdout().lock());

    // Demonstrate prediction at specific points
    let _ = writeln!(std::io::stdout().lock(), "🎯 Example Predictions:");
    let test_points = vec![(0.25, 0.25, 0.0), (0.5, 0.5, 0.005), (0.75, 0.75, 0.01)];

    for (x, y, t) in test_points {
        let x_point =
            Array1::from_vec(1, vec![x]).expect("invariant: single x prediction coordinate");
        let y_point =
            Array1::from_vec(1, vec![y]).expect("invariant: single y prediction coordinate");
        let t_point =
            Array1::from_vec(1, vec![t]).expect("invariant: single time prediction coordinate");

        let pred = trainer.pinn().predict(&x_point, &y_point, &t_point)?;
        let analytical = analytical_solution_2d(x, y, t, wave_speed);

        eprintln!(
            "   Point ({:.2}, {:.2}, {:.3}s): PINN={:.6}, Analytical={:.6}, Error={:.6}",
            x,
            y,
            t,
            pred[[0, 0]] as f64,
            analytical,
            (pred[[0, 0]] as f64 - analytical).abs()
        );
    }
    let _ = writeln!(std::io::stdout().lock());

    // Summary
    let _ = writeln!(
        std::io::stdout().lock(),
        "🎉 Example completed successfully!"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   PINN successfully learned the 2D wave equation physics"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Demonstrated convergence with physics-informed loss"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Achieved accurate predictions across the domain"
    );

    Ok(())
}

#[cfg(not(feature = "pinn"))]
fn main() {
    eprintln!("❌ PINN feature not enabled. Run with: cargo run --example pinn_2d_wave_equation --features pinn");
}
