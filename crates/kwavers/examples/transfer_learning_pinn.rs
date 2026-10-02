//! Transfer Learning for PINNs Example
//!
//! This example demonstrates the transfer learning capabilities of the PINN framework,
//! showing how pre-trained models can be efficiently adapted to new physics scenarios.

#[cfg(feature = "pinn")]
use coeus_core::MoiraiBackend;
#[cfg(feature = "pinn")]
use kwavers_solver::inverse::pinn::ml::transfer_learning::{
    FreezeStrategy, TransferLearner, TransferLearningConfig,
};
#[cfg(feature = "pinn")]
use kwavers_solver::inverse::pinn::ml::{
    BoundaryCondition2D, PinnConfig2D, PinnWave2D, WaveGeometry2D,
};
use std::io::Write;

#[cfg(feature = "pinn")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    type Backend = MoiraiBackend;

    let source_model = PinnWave2D::<Backend>::new(PinnConfig2D::default())?;

    let transfer_config = TransferLearningConfig {
        fine_tune_lr: 1e-4,
        fine_tune_epochs: 10,
        freeze_strategy: FreezeStrategy::FreezeAllButLast,
        patience: 3,
        wave_speed: 1500.0,
    };

    let mut learner = TransferLearner::new(source_model, transfer_config);

    let target_geometry = WaveGeometry2D::rectangular(0.0, 1.0, 0.0, 1.0);
    let target_conditions: Vec<BoundaryCondition2D> = Vec::new();

    let (_target_model, metrics) =
        learner.transfer_to_geometry(&target_geometry, &target_conditions)?;

    let _ = writeln!(std::io::stdout().lock(), "Transfer completed.");
    let _ = writeln!(
        std::io::stdout().lock(),
        "Initial accuracy: {:.3}",
        metrics.initial_accuracy
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Final accuracy: {:.3}",
        metrics.final_accuracy
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Convergence epochs: {}",
        metrics.convergence_epochs
    );

    Ok(())
}

#[cfg(not(feature = "pinn"))]
fn main() {
    eprintln!("🚫 PINN feature not enabled!");
    let _ = writeln!(
        std::io::stdout().lock(),
        "   This example requires the 'pinn' feature to be enabled."
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Run with: cargo run --example transfer_learning_pinn --features pinn"
    );
    std::process::exit(1);
}
