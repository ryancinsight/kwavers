//! Unit tests for the standing-wave suppression pipeline.
//!
//! Analytical contract:
//!
//! 1. SWI must be positive for DAS steering (reflected field creates λ/2 fringes).
//! 2. After `n_opt_iter` iterations SWI must be strictly lower than the initial value.
//! 3. Focal pressure history must have `n_opt_iter + 1` entries.
//! 4. Snapshot fields must have shape `(n_snapshots, nx, ny)` with non-zero energy.

use crate::therapy::theranostic_guidance::{
    run_standing_wave_suppression, StandingWaveOptConfig,
};
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE};

/// Minimal fast configuration for testing: small grid, few elements, few iterations.
fn small_config() -> StandingWaveOptConfig {
    StandingWaveOptConfig {
        nx: 64,
        ny: 32,
        dx_m: 1.5e-3,
        pml_cells: 8,
        c_ref_m_s: SOUND_SPEED_TISSUE,
        rho_ref_kg_m3: DENSITY_WATER_NOMINAL,
        c_layer_m_s: 2000.0,
        rho_layer_kg_m3: 1500.0,
        layer_x_start: 44,
        layer_x_end: 48,
        frequency_hz: 250_000.0,
        cfl: 0.25,
        source_x: 9,
        focus_x: 32,
        focus_y: 16,
        n_elements: 6,
        element_y_min: 6,
        element_y_max: 26,
        focal_radius_cells: 3,
        burst_cycles: 3.0,
        accum_skip_cycles: 1.5,
        swi_axis_half_width: 2,
        n_opt_iter: 5,
        swi_weight: 0.70,
        focal_weight: 0.30,
        grad_delta_rad: 0.05,
        armijo_c1: 0.01,
        line_search_alpha0: 1.0,
        line_search_beta: 0.5,
        line_search_max: 8,
        n_snapshots: 3,
    }
}

#[test]
fn swi_decreases_after_optimization() {
    let config = small_config();
    let result = run_standing_wave_suppression(&config);

    let swi_init = result.swi_history[0];
    let swi_final = *result.swi_history.last().unwrap();

    // History length: n_opt_iter + 1 (iter 0 is the DAS baseline).
    assert_eq!(
        result.swi_history.len(),
        config.n_opt_iter + 1,
        "SWI history must have n_opt_iter+1 entries"
    );

    // SWI must be non-negative and bounded by 1.
    assert!(
        swi_init >= 0.0 && swi_init <= 1.0,
        "initial SWI {swi_init} not in [0,1]"
    );

    // Optimization must reduce SWI (reflective layer creates measurable fringes).
    assert!(
        swi_final < swi_init,
        "SWI did not decrease: {swi_init:.4} → {swi_final:.4}"
    );
}

#[test]
fn history_lengths_match() {
    let config = small_config();
    let result = run_standing_wave_suppression(&config);
    let expected = config.n_opt_iter + 1;

    assert_eq!(result.swi_history.len(), expected);
    assert_eq!(result.focal_pressure_history.len(), expected);
    assert_eq!(result.objective_history.len(), expected);
}

#[test]
fn snapshot_shape_and_energy() {
    let config = small_config();
    let result = run_standing_wave_suppression(&config);

    let n_snap = result.snapshot_iterations.len();
    assert!(n_snap >= 1, "at least one snapshot expected");
    let (sn, sx, sy) = result.snapshot_fields_re.dim();
    assert_eq!((sn, sx, sy), (n_snap, config.nx, config.ny));

    // Each snapshot slice must carry non-zero acoustic energy.
    for k in 0..n_snap {
        let energy: f32 = result
            .snapshot_fields_re
            .slice(ndarray::s![k, .., ..])
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            + result
                .snapshot_fields_im
                .slice(ndarray::s![k, .., ..])
                .iter()
                .map(|&v| v * v)
                .sum::<f32>();
        assert!(energy > 0.0, "snapshot {k} has zero energy");
    }
}

#[test]
fn initial_and_final_fields_differ() {
    let config = small_config();
    let result = run_standing_wave_suppression(&config);

    let diff: f32 = result
        .initial_field_re
        .iter()
        .zip(result.final_field_re.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.0,
        "initial and final fields are identical — phases were not updated"
    );
}

#[test]
fn geometry_scalars_consistent() {
    let config = small_config();
    let result = run_standing_wave_suppression(&config);

    assert_eq!(result.nx, config.nx);
    assert_eq!(result.ny, config.ny);
    assert_eq!((result.dx_m - config.dx_m).abs() < 1e-15, true);
    assert_eq!(result.n_elements, config.n_elements);
    assert_eq!(result.element_ys.len(), config.n_elements);
    assert_eq!(result.reflector_x_start, config.layer_x_start);
    assert_eq!(result.reflector_x_end, config.layer_x_end);
    assert_eq!(result.sound_speed_map.dim(), (config.nx, config.ny));
}
