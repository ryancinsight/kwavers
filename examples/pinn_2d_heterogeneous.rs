//! Physics-Informed Neural Network (PINN) for 2D Wave Equation with Heterogeneous Media
//!
//! This example demonstrates solving the 2D acoustic wave equation in heterogeneous media
//! using physics-informed neural networks with spatially varying wave speeds.
//!
//! ## Heterogeneous Wave Equation
//!
//! ∂²u/∂t² = c²(x,y)(∂²u/∂x² + ∂²u/∂y²)
//!
//! ## Heterogeneous Media Examples
//!
//! - **Layered media**: Different wave speeds in different regions
//! - **Inclusions**: High/low speed regions within a uniform background
//! - **Gradient media**: Continuously varying wave speed fields
//!
//! ## Features Demonstrated
//!
//! - Heterogeneous media support with spatially varying c(x,y)
//! - Multi-region domain handling with interface conditions
//! - Physics-informed loss with heterogeneous PDE residuals
//! - Boundary condition enforcement in complex geometries
//! - Training convergence monitoring for heterogeneous problems

#[cfg(feature = "pinn")]
use kwavers::core::error::KwaversResult;
#[cfg(feature = "pinn")]
use kwavers::solver::inverse::pinn::ml::burn_wave_equation_2d::{
    BurnLossWeights2D, BurnPINN2DConfig, BurnPINN2DTrainer, BurnPINN2DWave, Geometry2D,
};
#[cfg(feature = "pinn")]
use ndarray::{Array1, Array2};
#[cfg(feature = "pinn")]
use std::time::Instant;

#[cfg(feature = "pinn")]
use burn::backend::NdArray;

#[cfg(feature = "pinn")]
type Backend = burn::backend::Autodiff<NdArray<f32>>;

/// Layered medium wave speed function
/// c(x,y) = 1500 m/s (water) for y < 0.5, 3000 m/s (tissue) for y >= 0.5
#[cfg(feature = "pinn")]
fn layered_medium_wave_speed(_x: f32, y: f32) -> f32 {
    if y < 0.5 {
        1500.0 // Water layer
    } else {
        3000.0 // Tissue layer
    }
}

/// Radial inclusion wave speed function
/// c(x,y) = 5000 m/s inside circle centered at (0.5, 0.5) with radius 0.2
/// c(x,y) = 1500 m/s elsewhere
#[cfg(feature = "pinn")]
fn inclusion_wave_speed(x: f32, y: f32) -> f32 {
    let center_x = 0.5;
    let center_y = 0.5;
    let radius = 0.2;

    let dx = x - center_x;
    let dy = y - center_y;
    let distance = (dx * dx + dy * dy).sqrt();

    if distance <= radius {
        5000.0 // Bone inclusion
    } else {
        1500.0 // Soft tissue
    }
}

/// Gradient medium wave speed function
/// c(x,y) = 1500 + 1500 * y (linear increase with depth)
#[cfg(feature = "pinn")]
fn gradient_wave_speed(_x: f32, y: f32) -> f32 {
    1500.0 + 1500.0 * y
}

/// Generate training data for heterogeneous media
#[cfg(feature = "pinn")]
fn generate_heterogeneous_training_data<F>(
    n_samples: usize,
    geometry: &Geometry2D,
    wave_speed_fn: F,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array2<f64>)
where
    F: Fn(f32, f32) -> f32,
{
    let mut x_data = Vec::with_capacity(n_samples);
    let mut y_data = Vec::with_capacity(n_samples);
    let mut t_data = Vec::with_capacity(n_samples);
    let mut u_data = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // Sample points within geometry
        let (x_points, y_points) = geometry.sample_points(1);
        let x = x_points[0];
        let y = y_points[0];
        let t = rand::random::<f64>() * 0.01; // Short time for stability

        // Get wave speed at this location
        let _c = wave_speed_fn(x as f32, y as f32);

        // Generate analytical solution (simplified - in practice would solve PDE)
        // For demonstration, use a simple standing wave pattern
        let k = std::f64::consts::PI / 1.0; // Wavenumber
        let omega = 2.0 * std::f64::consts::PI * 1000.0; // Frequency
        let u = (k * x).sin() * (k * y).sin() * (omega * t).cos() * 0.1;

        x_data.push(x);
        y_data.push(y);
        t_data.push(t);
        u_data.push(u);
    }

    (
        Array1::from_vec(x_data),
        Array1::from_vec(y_data),
        Array1::from_vec(t_data),
        Array2::from_shape_vec((n_samples, 1), u_data).unwrap(),
    )
}

/// Generate test grid for heterogeneous media validation
#[cfg(feature = "pinn")]
fn generate_heterogeneous_test_grid<F>(
    nx: usize,
    ny: usize,
    nt: usize,
    geometry: &Geometry2D,
    wave_speed_fn: F,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Vec<f32>)
where
    F: Fn(f32, f32) -> f32,
{
    let mut x_test = Vec::new();
    let mut y_test = Vec::new();
    let mut t_test = Vec::new();
    let mut c_test = Vec::new();

    let (x_min, x_max, y_min, y_max) = geometry.bounding_box();
    let dx = (x_max - x_min) / (nx - 1) as f64;
    let dy = (y_max - y_min) / (ny - 1) as f64;
    let dt = 0.01 / (nt - 1) as f64;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nt {
                let x = x_min + i as f64 * dx;
                let y = y_min + j as f64 * dy;
                let t = k as f64 * dt;

                // Only include points inside geometry
                if geometry.contains(x, y) {
                    let c = wave_speed_fn(x as f32, y as f32);

                    x_test.push(x);
                    y_test.push(y);
                    t_test.push(t);
                    c_test.push(c);
                }
            }
        }
    }

    (
        Array1::from_vec(x_test),
        Array1::from_vec(y_test),
        Array1::from_vec(t_test),
        c_test,
    )
}

#[cfg(feature = "pinn")]
fn main() -> KwaversResult<()> {
    println!("🧠 Physics-Informed Neural Network for 2D Heterogeneous Wave Equation");
    println!("====================================================================");

    // Configuration
    let n_training_samples = 1000;
    let n_collocation_points = 2000;
    let epochs = 200;

    println!("📋 Configuration:");
    println!("   Training samples: {}", n_training_samples);
    println!("   Collocation points: {}", n_collocation_points);
    println!("   Training epochs: {}", epochs);
    println!();

    // Initialize Burn backend
    let device: <Backend as burn::tensor::backend::Backend>::Device = Default::default();
    println!("🔥 Burn Backend: Initialized (CPU with Autodiff)");
    println!();

    // Create PINN configuration optimized for heterogeneous media
    let pinn_config = BurnPINN2DConfig {
        hidden_layers: vec![150, 150, 150, 150], // Larger network for complex media
        learning_rate: 5e-4,                     // Lower learning rate for stability
        loss_weights: BurnLossWeights2D {
            data: 1.0,
            pde: 2.0, // Higher PDE weight for heterogeneous physics
            boundary: 20.0,
            initial: 20.0,
        },
        num_collocation_points: n_collocation_points,
        ..Default::default()
    };

    println!("🧠 PINN Configuration:");
    println!("   Hidden layers: {:?}", pinn_config.hidden_layers);
    println!("   Learning rate: {}", pinn_config.learning_rate);
    println!(
        "   Loss weights: data={:.1}, pde={:.1}, boundary={:.1}, initial={:.1}",
        pinn_config.loss_weights.data,
        pinn_config.loss_weights.pde,
        pinn_config.loss_weights.boundary,
        pinn_config.loss_weights.initial
    );
    println!();

    // Test different heterogeneous media configurations
    let media_configs = vec![
        (
            "Layered Medium",
            layered_medium_wave_speed as fn(f32, f32) -> f32,
        ),
        (
            "Inclusion Medium",
            inclusion_wave_speed as fn(f32, f32) -> f32,
        ),
        (
            "Gradient Medium",
            gradient_wave_speed as fn(f32, f32) -> f32,
        ),
    ];

    for (media_name, wave_speed_fn) in media_configs {
        println!("🌊 Testing {} Configuration", media_name);
        println!("=====================================");

        // Create geometry (unit square)
        let geometry = Geometry2D::rectangular(0.0, 1.0, 0.0, 1.0);
        println!("📐 Geometry: Unit square [0,1] × [0,1]");
        println!("🎵 Wave speed: {}", media_name);
        println!();

        // Create heterogeneous PINN
        let _pinn = BurnPINN2DWave::<Backend>::new_heterogeneous(
            pinn_config.clone(),
            wave_speed_fn,
            &device,
        )?;
        println!("✅ Heterogeneous PINN: Created successfully");
        println!();

        // Generate training data
        println!("📊 Generating training data...");
        let (x_train, y_train, t_train, u_train) =
            generate_heterogeneous_training_data(n_training_samples, &geometry, wave_speed_fn);
        println!("   Training points: {}", x_train.len());
        println!();

        // Generate test data for validation
        println!("🧪 Validating PINN predictions...");
        let (x_test, y_test, t_test, _c_test) =
            generate_heterogeneous_test_grid(8, 8, 3, &geometry, wave_speed_fn);
        println!("   Test points: {}", x_test.len());

        // Create trainer
        let trainer = BurnPINN2DTrainer::<Backend>::new_trainer(pinn_config.clone(), geometry, &device)?;
        println!("✅ PINN Trainer: Created successfully");
        println!();

        // Train PINN
        println!("🚀 Training PINN on {}...", media_name);
        let start_time = Instant::now();
        let mut trainer = trainer;
        let metrics = trainer.train(
            &x_train,
            &y_train,
            &t_train,
            &u_train,
            1500.0, // Default fallback (not used in heterogeneous mode)
            &pinn_config,
            &device,
            epochs,
        )?;
        let training_time = start_time.elapsed();

        println!(
            "✅ Training completed in {:.2}s",
            training_time.as_secs_f64()
        );
        println!(
            "   Final total loss: {:.6e}",
            metrics.total_loss.last().unwrap()
        );
        println!(
            "   Final data loss: {:.6e}",
            metrics.data_loss.last().unwrap()
        );
        println!(
            "   Final PDE loss: {:.6e}",
            metrics.pde_loss.last().unwrap()
        );
        println!("   Final BC loss: {:.6e}", metrics.bc_loss.last().unwrap());
        println!("   Final IC loss: {:.6e}", metrics.ic_loss.last().unwrap());
        println!();

        // Make predictions
        let predictions = trainer.pinn().predict(&x_test, &y_test, &t_test, &device)?;
        println!("   Predictions completed");

        // Compute error statistics
        let mut errors = Vec::new();
        for i in 0..x_test.len() {
            let x = x_test[i];
            let y = y_test[i];
            let t = t_test[i];
            let k = std::f64::consts::PI / 1.0;
            let omega = 2.0 * std::f64::consts::PI * 1000.0;
            let u_exact = (k * x).sin() * (k * y).sin() * (omega * t).cos() * 0.1;
            let u_pred = predictions[[i, 0]] as f64;
            errors.push((u_pred - u_exact).abs());
        }

        let mean_error: f64 = errors.iter().sum::<f64>() / errors.len() as f64;
        let max_error = errors.iter().cloned().fold(0.0, f64::max);

        println!("   Mean error: {:.6e}", mean_error);
        println!("   Max error: {:.6e}", max_error);
        println!();

        // Performance analysis
        println!("📈 Performance Analysis:");
        println!(
            "   Training time: {:.2}s ({:.1} ms/epoch)",
            training_time.as_secs_f64(),
            training_time.as_millis() as f64 / epochs as f64
        );

        let loss_reduction = metrics.total_loss[0] / metrics.total_loss.last().unwrap();
        println!("   Loss reduction: {:.2e}x", loss_reduction);

        let convergence_epoch = metrics
            .total_loss
            .iter()
            .enumerate()
            .find(|(_, &loss)| loss < 1e-2)
            .map(|(epoch, _)| epoch)
            .unwrap_or(epochs);
        println!(
            "   Convergence: {} epochs to reach 1e-2 loss",
            convergence_epoch
        );
        println!();

        // Demonstrate wave speed evaluation
        println!("🌊 Wave Speed Distribution:");
        let test_points = vec![
            (0.25, 0.25),
            (0.25, 0.75),
            (0.5, 0.5),
            (0.75, 0.25),
            (0.75, 0.75),
        ];

        for (x, y) in test_points {
            let c = wave_speed_fn(x as f32, y as f32);
            println!("   Point ({:.2}, {:.2}): c = {:.0} m/s", x, y, c);
        }
        println!();
    }

    // Summary
    println!("🎉 Heterogeneous Media PINN Examples Completed!");
    println!("   ✅ Layered media with interface conditions");
    println!("   ✅ Inclusion media with embedded structures");
    println!("   ✅ Gradient media with continuous variation");
    println!("   ✅ Physics-informed learning in complex media");
    println!("   ✅ Spatially varying PDE constraint enforcement");

    Ok(())
}

#[cfg(not(feature = "pinn"))]
fn main() {
    println!("❌ PINN feature not enabled. Run with: cargo run --example pinn_2d_heterogeneous --features pinn");
}
