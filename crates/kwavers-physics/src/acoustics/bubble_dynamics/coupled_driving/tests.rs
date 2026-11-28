use super::coupling::history_row;
use super::*;
use crate::acoustics::bubble_dynamics::bubble_state::BubbleParameters;
use crate::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;
use std::f64::consts::PI;

/// WS-B medium (km_coupled.py constants): ρ = 998, σ = 0.0725,
/// μ = 1e-3, pv = 2.34e3, c_KM = 1481, p0 = 1.013e5, γ = 1.4
/// (air), thermal/mass-transfer off (the Python closure).
fn imaging_params(r0: f64) -> BubbleParameters {
    let mut p = BubbleParameters::default();
    p.r0 = r0;
    p.p0 = 1.013e5;
    p.rho_liquid = 998.0;
    p.c_liquid = 1481.0;
    p.mu_liquid = 1.0e-3;
    p.sigma = 0.0725;
    p.pv = 2.34e3;
    p.use_thermal_effects = false;
    p.use_mass_transfer = false;
    p.driving_frequency = COUPLED_VALIDATED_FREQUENCY_HZ;
    p
}

fn tone_history(n: usize, dt: f64, amp: f64) -> Vec<f64> {
    let omega = 2.0 * PI * COUPLED_VALIDATED_FREQUENCY_HZ;
    (0..=n)
        .map(|k| amp * (omega * k as f64 * dt).sin())
        .collect()
}

fn test_config() -> CoupledCloudConfig {
    CoupledCloudConfig {
        steps_per_cycle: 500,
        ..CoupledCloudConfig::default()
    }
}

fn cloud_band_power_db(sol: &CoupledSolution, config: &CoupledCloudConfig) -> f64 {
    let dt = 1.0 / (COUPLED_VALIDATED_FREQUENCY_HZ * config.steps_per_cycle as f64);
    let seg = config.segment_start_cycle * config.steps_per_cycle;
    let [nb, _] = sol.volume_accel.shape();
    let total: f64 = (0..nb)
        .map(|i| {
            segment_band_power(
                history_row(&sol.volume_accel, i),
                dt,
                seg,
                config.density_kg_m3,
            )
        })
        .sum();
    10.0 * total.log10()
}

#[test]
fn pair_geometry_weights_and_delays() {
    let pos = [[0.0, 0.0, 0.0], [1.0e-3, 0.0, 0.0]];
    let (w, d) = pair_geometry(&pos, 998.0, 1540.0);
    assert_eq!(w[[0, 0]], 0.0);
    assert_eq!(d[[1, 1]], 0.0);
    let expected_w = 998.0 / (4.0 * PI * 1.0e-3);
    assert!((w[[0, 1]] - expected_w).abs() / expected_w < 1e-12);
    assert!((d[[0, 1]] - 1.0e-3 / 1540.0).abs() < 1e-15);
    assert_eq!(w[[0, 1]], w[[1, 0]]);
}

/// WS-B dilute-limit validation, self-consistent form: at
/// 5× the cloud radius (density ×0.008 of the bracket scale)
/// the coupled solve must recover independent driving with
/// the 1.5f₀ band power within ≈ 0.2 dB (Python: −0.11 dB).
#[test]
fn dilute_limit_recovers_independent_driving() {
    let config = test_config();
    let n = config.n_cycles * config.steps_per_cycle;
    let dt = 1.0 / (COUPLED_VALIDATED_FREQUENCY_HZ * config.steps_per_cycle as f64);
    let incident = tone_history(n, dt, 0.268e6);
    // Three bubbles ≥ 7 mm apart (cf. WS-B dilute radius ×5).
    let bubbles = vec![
        CoupledBubble {
            radius0_m: 2.5e-6,
            position_m: [0.0, 0.0, 0.0],
        },
        CoupledBubble {
            radius0_m: 2.5e-6,
            position_m: [7.0e-3, 1.0e-3, 0.0],
        },
        CoupledBubble {
            radius0_m: 2.5e-6,
            position_m: [1.0e-3, 7.0e-3, 2.0e-3],
        },
    ];
    let models: Vec<KellerMiksisModel> = bubbles
        .iter()
        .map(|b| KellerMiksisModel::new(imaging_params(b.radius0_m)))
        .collect();
    let independent = solve_coupled_cloud(&models, &bubbles, &incident, &config, 0).unwrap();
    let coupled = solve_coupled_cloud(&models, &bubbles, &incident, &config, 3).unwrap();
    let delta_db =
        cloud_band_power_db(&coupled, &config) - cloud_band_power_db(&independent, &config);
    assert!(
        delta_db.abs() < 0.2,
        "dilute coupled−independent band power {delta_db} dB (WS-B: −0.11 dB)"
    );
}

/// Two identical bubbles at mirror positions under a spatially
/// uniform incident tone must follow identical trajectories.
#[test]
fn two_identical_bubbles_are_symmetric() {
    let config = test_config();
    let n = config.n_cycles * config.steps_per_cycle;
    let dt = 1.0 / (COUPLED_VALIDATED_FREQUENCY_HZ * config.steps_per_cycle as f64);
    let incident = tone_history(n, dt, 0.268e6);
    let bubbles = vec![
        CoupledBubble {
            radius0_m: 2.5e-6,
            position_m: [-25.0e-6, 0.0, 0.0],
        },
        CoupledBubble {
            radius0_m: 2.5e-6,
            position_m: [25.0e-6, 0.0, 0.0],
        },
    ];
    let models: Vec<KellerMiksisModel> = bubbles
        .iter()
        .map(|b| KellerMiksisModel::new(imaging_params(b.radius0_m)))
        .collect();
    let sol = solve_coupled_cloud(&models, &bubbles, &incident, &config, 2).unwrap();
    let max_diff = history_row(&sol.radius, 0)
        .iter()
        .zip(history_row(&sol.radius, 1))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff < 1e-12, "symmetry broken: max ΔR {max_diff} m");
    // And the coupling must actually be active at 50 µm:
    // coupled trajectories differ from independent ones.
    let independent = solve_coupled_cloud(&models, &bubbles, &incident, &config, 0).unwrap();
    let effect = history_row(&sol.radius, 0)
        .iter()
        .zip(history_row(&independent.radius, 0))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(effect > 1e-9, "coupling inactive at 50 µm: ΔR {effect} m");
}

/// Damped Picard converges to a fixed point on a close pair
/// (the configuration where plain Picard oscillates, WS-B):
/// successive band-power changes shrink and the last is small.
#[test]
fn damped_picard_converges_to_fixed_point() {
    let config = test_config();
    let n = config.n_cycles * config.steps_per_cycle;
    let dt = 1.0 / (COUPLED_VALIDATED_FREQUENCY_HZ * config.steps_per_cycle as f64);
    let incident = tone_history(n, dt, 0.123e6);
    let bubbles = vec![
        CoupledBubble {
            radius0_m: 2.5e-6,
            position_m: [0.0, 0.0, 0.0],
        },
        CoupledBubble {
            radius0_m: 2.5e-6,
            position_m: [50.0e-6, 0.0, 0.0],
        },
    ];
    let models: Vec<KellerMiksisModel> = bubbles
        .iter()
        .map(|b| KellerMiksisModel::new(imaging_params(b.radius0_m)))
        .collect();
    let sol = solve_coupled_cloud(&models, &bubbles, &incident, &config, 5).unwrap();
    let changes = &sol.pass_band_power_change_db;
    assert_eq!(changes.len(), 5);
    let first = changes[0].abs();
    let last = changes[changes.len() - 1].abs();
    assert!(
        last < first && last < 0.5,
        "Picard not converging: changes {changes:?} dB"
    );
}

#[test]
fn regime_guard_flags_trigger_pressures() {
    assert!(coupled_regime_reasons(2.45e6, 0.26e6, 32.0).is_empty());
    assert!(coupled_regime_reasons(2.45e6, 1.27e6, 128.0).is_empty());
    // Trigger pressures: outside, by construction.
    assert!(!coupled_regime_reasons(2.45e6, 20.0e6, 32.0).is_empty());
    assert!(!coupled_regime_reasons(1.0e6, 0.26e6, 32.0).is_empty());
    assert!(!coupled_regime_reasons(2.45e6, 0.26e6, 500.0).is_empty());
}
