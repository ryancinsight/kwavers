//! Bounded multi-domain PINN training for the cavitation-to-light pathway.
//!
//! The example runs the real universal solver over its registered cavitation,
//! sonoluminescence, and electromagnetic domains. It reports the returned
//! domain statistics; it does not substitute fabricated luminosity or
//! conservation values for a physical field solve.

use coeus_core::MoiraiBackend;
use kwavers_core::error::KwaversResult;
use kwavers_solver::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
use kwavers_solver::inverse::pinn::ml::universal_solver::{
    UniversalPINNSolver, UniversalTrainingConfig,
};

type Backend = MoiraiBackend;

#[expect(
    clippy::field_reassign_with_default,
    reason = "the bounded demo varies workload controls while retaining canonical solver defaults"
)]
fn demo_config() -> UniversalTrainingConfig {
    let mut config = UniversalTrainingConfig::default();
    config.epochs = 2;
    config.collocation_points = 32;
    config.boundary_points = 8;
    config.initial_points = 8;
    config.batch_size = 8;
    config.adaptive_sampling = false;
    config
}

fn main() -> KwaversResult<()> {
    let mut solver = UniversalPINNSolver::<Backend>::with_cavitation_sonoluminescence_coupling()?;
    let config = demo_config();
    let physics = PinnDomainPhysicsParameters {
        material_properties: [
            ("ambient_pressure".to_owned(), 101_325.0),
            ("liquid_density".to_owned(), 1_000.0),
            ("speed_of_sound".to_owned(), 1_500.0),
            ("surface_tension".to_owned(), 0.072),
            ("viscosity".to_owned(), 0.001),
        ]
        .into(),
        boundary_values: [
            ("pressure_amplitude".to_owned(), vec![1.0e5]),
            ("frequency".to_owned(), vec![1.0e6]),
        ]
        .into(),
        initial_values: [
            ("initial_bubble_radius".to_owned(), vec![1.0e-6]),
            ("equilibrium_radius".to_owned(), vec![5.0e-6]),
        ]
        .into(),
        domain_params: [
            ("bubble_concentration".to_owned(), 1.0e8),
            ("temperature".to_owned(), 293.15),
            ("dissolved_gas".to_owned(), 0.02),
        ]
        .into(),
    };

    let result = solver.train_all_domains(&config, &physics)?;
    println!("trained {} domains", result.domain_stats.len());
    println!("total final loss: {:.6e}", result.total_loss);
    println!("training time: {:.3}s", result.training_time.as_secs_f64());

    let mut domain_names: Vec<_> = result.domain_stats.keys().collect();
    domain_names.sort_unstable();
    for domain_name in domain_names {
        let stats = &result.domain_stats[domain_name];
        println!(
            "{domain_name}: final_loss={:.6e}, converged={}, epochs={}",
            stats.final_loss, stats.convergence_info.converged, stats.convergence_info.final_epoch,
        );
    }

    Ok(())
}
