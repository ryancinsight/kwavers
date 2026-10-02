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
use coeus_core::MoiraiBackend;
#[cfg(feature = "pinn")]
use kwavers_solver::inverse::pinn::elastic_2d::{Config, ElasticPINN2D};
#[cfg(feature = "pinn")]
use std::error::Error;
#[cfg(feature = "pinn")]
use std::io::Write;

#[cfg(feature = "pinn")]
type AutodiffBackend = MoiraiBackend;

#[cfg(feature = "pinn")]
#[path = "pinn_training_convergence/analytical_data.rs"]
mod analytical_data;
#[cfg(feature = "pinn")]
#[path = "pinn_training_convergence/studies.rs"]
mod studies;
#[cfg(feature = "pinn")]
use analytical_data::{generate_training_data, ExperimentConfig, PlaneWaveAnalytical};
#[cfg(feature = "pinn")]
use studies::{h_refinement_study, train_pinn, validate_gradients};

#[cfg(feature = "pinn")]
fn main() -> Result<(), Box<dyn Error>> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "============================================================="
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  PINN Training with Convergence Analysis"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "=============================================================\n"
    );

    // Physical parameters (water-like medium)
    let density: f64 = 1000.0; // kg/m³
    let lambda: f64 = 2.25e9; // Pa (Lamé first parameter)
    let mu: f64 = 0.0; // Pa (shear modulus, ~0 for fluids)
    let c_p: f64 = ((lambda + 2.0 * mu) / density).sqrt(); // P-wave speed ≈ 1500 m/s

    let _ = writeln!(std::io::stdout().lock(), "Physical Parameters:");
    let _ = writeln!(std::io::stdout().lock(), "  Density: {} kg/m³", density);
    let _ = writeln!(std::io::stdout().lock(), "  Lambda: {:.2e} Pa", lambda);
    let _ = writeln!(std::io::stdout().lock(), "  Mu: {:.2e} Pa", mu);
    let _ = writeln!(std::io::stdout().lock(), "  P-wave speed: {:.2} m/s\n", c_p);

    // Analytical solution
    let wavelength = 0.01; // 1 cm
    let amplitude = 1e-6; // 1 μm
    let solution = PlaneWaveAnalytical::new(amplitude, wavelength, c_p);

    let _ = writeln!(std::io::stdout().lock(), "Analytical Solution:");
    let _ = writeln!(std::io::stdout().lock(), "  Type: P-wave plane wave");
    let _ = writeln!(std::io::stdout().lock(), "  Amplitude: {} m", amplitude);
    let _ = writeln!(std::io::stdout().lock(), "  Wavelength: {} m", wavelength);
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Frequency: {:.2} kHz\n",
        solution.omega / (2.0 * std::f64::consts::PI) / 1000.0
    );

    // Domain parameters
    let domain_size = 0.05; // 5 cm
    let t_max = 1e-5; // 10 μs

    // Single training run
    let _ = writeln!(std::io::stdout().lock(), "=== Single Training Run ===");
    let num_points = 32;
    let (inputs, targets) = generate_training_data(&solution, num_points, domain_size, t_max);
    let _ = writeln!(
        std::io::stdout().lock(),
        "Generated {} training samples",
        inputs.len()
    );

    let pinn_config = Config {
        hidden_layers: vec![64, 64, 64, 64],
        learning_rate: 1e-3,
        n_epochs: 1000,
        ..Default::default()
    };
    let model = ElasticPINN2D::<AutodiffBackend>::new(&pinn_config)?;

    let config = ExperimentConfig::default();
    let (model, _loss_history) = train_pinn(model, &inputs, &targets, &config)?;

    // Gradient validation (on trained model)
    validate_gradients(&model, [0.0, 0.025, 0.025])?;

    // H-refinement study
    let resolutions = vec![16, 32, 64];
    let convergence_data = h_refinement_study(&solution, &resolutions, domain_size, t_max)?;

    // Summary
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n============================================================="
    );
    let _ = writeln!(std::io::stdout().lock(), "  Summary");
    let _ = writeln!(
        std::io::stdout().lock(),
        "============================================================="
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "✓ PINN training completed successfully"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "✓ Gradient validation performed (autodiff vs FD)"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "✓ H-refinement convergence study completed"
    );
    let _ = writeln!(std::io::stdout().lock(), "\nConvergence Results:");
    for (h, error) in convergence_data {
        let _ = writeln!(
            std::io::stdout().lock(),
            "  h = {}: L2 error = {:.6e}",
            h,
            error
        );
    }

    let _ = writeln!(
        std::io::stdout().lock(),
        "\n============================================================="
    );
    let _ = writeln!(std::io::stdout().lock(), "  Recommendations for Next Steps");
    let _ = writeln!(
        std::io::stdout().lock(),
        "============================================================="
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "1. Run longer training (5000-10000 epochs) for better convergence"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "2. Implement proper optimizer (Adam with learning rate scheduler)"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "3. Add PDE residual loss to training (currently data-only)"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "4. Generate convergence plots (log-log error vs resolution)"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "5. Compare against FEM/FDTD solutions"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "6. Extend to 3D and heterogeneous media"
    );

    Ok(())
}

#[cfg(not(feature = "pinn"))]
fn main() {
    eprintln!("This example requires the 'pinn' feature.");
    eprintln!("Run with: cargo run --example pinn_training_convergence --features pinn --release");
    std::process::exit(1);
}
