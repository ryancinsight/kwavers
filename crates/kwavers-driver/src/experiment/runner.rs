//! Experiment orchestrator — the public entry point for a full driver→transducer simulation run
//! (Phase 5).
//!
//! [`run_experiment`] wires the slice together:
//!
//! 1. Validate the manifest + budget into a typed [`crate::validate::KwaversBeamStep`].
//! 2. Simulate the acoustic field via the injected [`AcousticSimulator`].
//! 3. Propagate device dissipation to a [`ThermalState`] via the package θ_jc.
//! 4. Build the 4-check [`PhysicsReport`] from the simulated scalars.
//! 5. Assemble and return an [`ExperimentReport`].
//!
//! All dependency injection is by trait bound (`S: AcousticSimulator`); the orchestrator has no
//! concrete imports from `kwavers-transducer` and is not gated on the `kwavers` feature.

use crate::error::validate::Validate;
use crate::manifest::{DriverManifest, EnergyBudgetReport};
use crate::validate::manifest_to_kwavers_beam_step;

use super::acoustic::AcousticSimulator;
use super::metrics::{build_beam_report, ExperimentMetrics};
use super::recorder::ExperimentRecord;
use super::thermal::{propagate_thermal, ThermalState};

/// Full output of one [`run_experiment`] call.
#[derive(Debug, Clone)]
pub struct ExperimentReport {
    /// The deterministic output bundle (pre-step, metrics, beam report).
    pub record: ExperimentRecord,
    /// Per-tile thermal state (rises, peak, headroom).
    pub thermal: ThermalState,
}

/// Run a complete driver→transducer experiment.
///
/// # Parameters
///
/// * `manifest` — full-stack v2 manifest (96 lanes, 4 tile profiles, no legacy stim block).
/// * `budget` — pre-computed energy budget from [`DriverManifest::validate_v2_energy_budget`].
/// * `simulator` — an [`AcousticSimulator`] impl (inject [`super::acoustic::InCrateAcousticSim`]
///   for the in-crate model, or [`super::acoustic::KwaversSim`] when the `kwavers` feature is on).
/// * `theta_jc_k_per_w` — package junction-to-case thermal resistance (K/W).
/// * `dt_max_k` — design thermal budget (K); headroom is `dt_max_k − peak_rise_k`.
///
/// # Errors
///
/// * [`crate::error::Validate::KwaversBeamStepContract`] — if the manifest is not full-stack v2
///   or the geometry is non-physical (from [`manifest_to_kwavers_beam_step`]).
/// * Any error surfaced by the [`AcousticSimulator`] impl (e.g. non-finite pressure, DIP seam).
pub fn run_experiment<S: AcousticSimulator>(
    manifest: &DriverManifest,
    budget: &EnergyBudgetReport,
    simulator: &S,
    theta_jc_k_per_w: f64,
    dt_max_k: f64,
) -> crate::error::Result<ExperimentReport> {
    // 1. Validate manifest + budget into the typed pre-step.
    let step = manifest_to_kwavers_beam_step(manifest, budget)
        .map_err(Validate::KwaversBeamStepContract)?;

    // 2. Acoustic simulation via the injected simulator.
    let pressure_map = simulator.simulate(&step, budget)?;

    // 3. Electro-thermal propagation.
    let thermal = propagate_thermal(budget, theta_jc_k_per_w, dt_max_k);

    // 4. Build the 4-check beam report from the simulated scalars (not from the analytical
    //    estimate in validate_against_budget). The per-tile min resistor margin is sourced
    //    from the budget (already gated at validate_v2_energy_budget time).
    let min_resistor_margin_w = step
        .resistor_margin_w
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let min_resistor_margin_w = if min_resistor_margin_w.is_finite() {
        min_resistor_margin_w
    } else {
        0.0
    };
    let beam_report = build_beam_report(&pressure_map, &step, min_resistor_margin_w);

    // 5. Assemble.
    let metrics = ExperimentMetrics::from_parts(&pressure_map, &thermal);
    let record = ExperimentRecord::new(step, metrics, beam_report);
    Ok(ExperimentReport { record, thermal })
}
