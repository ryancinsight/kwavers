use crate::clinical::therapy::theranostic_guidance::nonlinear3d::types::{
    GridIndex, Nonlinear3dAperture, SourceDomain,
};
use crate::clinical::therapy::theranostic_guidance::Point3;

use super::super::TimeSchedule;
use super::{
    build_source_plan, inject_sources, plan::DriveContext, travel::source_focus_travel_time_s,
};
use crate::clinical::therapy::theranostic_guidance::nonlinear3d::{
    encoding::SourceEncoding, types::flat_index,
};

#[test]
fn source_plan_preserves_per_element_drive_weights() {
    let n = 8;
    let cells = n * n * n;
    let aperture = Nonlinear3dAperture {
        sources: vec![
            GridIndex { x: 1, y: 1, z: 1 },
            GridIndex { x: 6, y: 1, z: 1 },
            GridIndex { x: 1, y: 6, z: 1 },
            GridIndex { x: 6, y: 6, z: 1 },
        ],
        receivers: Vec::new(),
        therapy_points_m: vec![
            Point3 {
                x_m: 0.0,
                y_m: 0.0,
                z_m: 0.0,
            };
            4
        ],
        receiver_points_m: Vec::new(),
        model_name: "source_plan_test".to_owned(),
        source_domain: SourceDomain::TissueBoundary,
        focus: GridIndex { x: 4, y: 4, z: 4 },
    };
    let speed = vec![1500.0; cells];

    let plan = build_source_plan(
        &speed,
        n,
        1.0e-3,
        &aperture,
        SourceEncoding { index: 0, count: 1 },
        None,
    );

    assert_eq!(plan.encoding_weights, vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(plan.source_stencils.len(), 4);
    for stencil in &plan.source_stencils {
        let max_weight = stencil
            .iter()
            .map(|(_, weight)| *weight)
            .fold(0.0, f64::max);
        assert!((max_weight - 1.0).abs() < 1.0e-12);
        assert!(stencil
            .iter()
            .all(|(_, weight)| *weight > 0.0 && *weight <= 1.0));
    }
}

#[test]
fn source_plan_steers_with_straight_ray_slowness() {
    let n = 9;
    let focus = GridIndex { x: 6, y: 6, z: 4 };
    let far_fast = GridIndex { x: 2, y: 2, z: 4 };
    let near_slow = GridIndex { x: 6, y: 4, z: 4 };
    let aperture = Nonlinear3dAperture {
        sources: vec![far_fast, near_slow],
        receivers: Vec::new(),
        therapy_points_m: vec![
            Point3 {
                x_m: 0.0,
                y_m: 0.0,
                z_m: 0.0,
            };
            2
        ],
        receiver_points_m: Vec::new(),
        model_name: "slowness_steering_test".to_owned(),
        source_domain: SourceDomain::TissueBoundary,
        focus,
    };
    let mut speed = vec![15_000.0; n * n * n];
    for y in 5..=6 {
        speed[flat_index(GridIndex { x: 6, y, z: 4 }, n)] = 343.0;
    }

    let far_time = source_focus_travel_time_s(&speed, n, 1.0e-3, far_fast, focus);
    let near_time = source_focus_travel_time_s(&speed, n, 1.0e-3, near_slow, focus);
    let plan = build_source_plan(
        &speed,
        n,
        1.0e-3,
        &aperture,
        SourceEncoding { index: 0, count: 1 },
        None,
    );

    assert!(
        far_time < near_time,
        "heterogeneous slowness must be able to dominate geometric distance"
    );
    assert!((plan.focused_delays_s[0] - (near_time - far_time)).abs() < 1.0e-12);
    assert!(plan.focused_delays_s[1].abs() < 1.0e-12);
}

#[test]
fn focused_delays_align_emission_phase_at_focus() {
    let n = 11;
    let focus = GridIndex { x: 6, y: 5, z: 5 };
    let sources = vec![
        GridIndex { x: 1, y: 5, z: 5 },
        GridIndex { x: 6, y: 2, z: 5 },
        GridIndex { x: 8, y: 9, z: 5 },
    ];
    let aperture = Nonlinear3dAperture {
        sources: sources.clone(),
        receivers: Vec::new(),
        therapy_points_m: vec![
            Point3 {
                x_m: 0.0,
                y_m: 0.0,
                z_m: 0.0,
            };
            sources.len()
        ],
        receiver_points_m: Vec::new(),
        model_name: "phase_alignment_test".to_owned(),
        source_domain: SourceDomain::TissueBoundary,
        focus,
    };
    let speed = vec![1500.0; n * n * n];
    let plan = build_source_plan(
        &speed,
        n,
        1.0e-3,
        &aperture,
        SourceEncoding { index: 0, count: 1 },
        None,
    );
    let arrivals = sources
        .iter()
        .zip(plan.focused_delays_s.iter())
        .map(|(source, delay)| {
            source_focus_travel_time_s(&speed, n, 1.0e-3, *source, focus) + delay
        })
        .collect::<Vec<_>>();
    let reference = arrivals[0];

    for (idx, arrival) in arrivals.iter().enumerate() {
        assert!(
            (*arrival - reference).abs() <= 1.0e-15,
            "source {idx} arrival {arrival:e} must match reference {reference:e}"
        );
    }
}

#[test]
fn focused_delays_apply_fast_skull_path_phase_correction() {
    let n = 11;
    let focus = GridIndex { x: 5, y: 5, z: 5 };
    let fast_skull_path = GridIndex { x: 1, y: 5, z: 5 };
    let soft_tissue_path = GridIndex { x: 9, y: 5, z: 5 };
    let sources = vec![fast_skull_path, soft_tissue_path];
    let aperture = Nonlinear3dAperture {
        sources: sources.clone(),
        receivers: Vec::new(),
        therapy_points_m: vec![
            Point3 {
                x_m: 0.0,
                y_m: 0.0,
                z_m: 0.0,
            };
            sources.len()
        ],
        receiver_points_m: Vec::new(),
        model_name: "skull_phase_test".to_owned(),
        source_domain: SourceDomain::TissueBoundary,
        focus,
    };
    let mut speed = vec![1500.0; n * n * n];
    for x in 2..=4 {
        speed[flat_index(GridIndex { x, y: 5, z: 5 }, n)] = 3000.0;
    }
    let fast_time = source_focus_travel_time_s(&speed, n, 1.0e-3, fast_skull_path, focus);
    let soft_time = source_focus_travel_time_s(&speed, n, 1.0e-3, soft_tissue_path, focus);
    let plan = build_source_plan(
        &speed,
        n,
        1.0e-3,
        &aperture,
        SourceEncoding { index: 0, count: 1 },
        None,
    );

    assert!(
        fast_time < soft_time,
        "high-speed skull segment must reduce the straight-ray travel time"
    );
    assert!((plan.focused_delays_s[0] - (soft_time - fast_time)).abs() <= 1.0e-15);
    assert!(plan.focused_delays_s[1].abs() <= 1.0e-15);
    assert!(
        (plan.focused_delays_s[0] + fast_time - soft_time).abs() <= 1.0e-15,
        "delayed fast path must arrive in phase with the slower path"
    );
}

#[test]
fn source_injection_imposes_bounded_pressure_without_accumulating_drive() {
    use crate::clinical::therapy::theranostic_guidance::nonlinear3d::types::Nonlinear3dConfig;
    use crate::clinical::therapy::theranostic_guidance::AnatomyKind;
    let n = 8;
    let mut field = vec![0.0; n * n * n];
    let source = GridIndex { x: 4, y: 4, z: 4 };
    let aperture = Nonlinear3dAperture {
        sources: vec![source],
        receivers: Vec::new(),
        therapy_points_m: vec![Point3 {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 0.0,
        }],
        receiver_points_m: Vec::new(),
        model_name: "bounded_source_test".to_owned(),
        source_domain: SourceDomain::TissueBoundary,
        focus: source,
    };
    let speed = vec![1500.0; field.len()];
    let plan = build_source_plan(
        &speed,
        n,
        1.0e-3,
        &aperture,
        SourceEncoding { index: 0, count: 1 },
        None,
    );
    let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
    config.frequency_hz = 1.0e6;
    config.source_pressure_pa = 28.0e6;
    config.cycles = 2.0;
    let schedule = TimeSchedule {
        dt_s: 0.25 / config.frequency_hz,
        time_steps: 8,
    };
    let drive = DriveContext {
        config: &config,
        schedule,
        source_scale: 1.0,
    };
    for step in 0..32 {
        inject_sources(&mut field, &plan, &drive, step);
    }
    let peak = field.iter().copied().map(f64::abs).fold(0.0, f64::max);
    assert!(
        peak <= config.source_pressure_pa,
        "bounded pressure source must not accumulate above configured drive; peak={peak:e}"
    );
}

#[test]
fn exterior_coupling_source_stencil_excludes_body_cells() {
    use super::plan::source_plan_metrics;
    let n = 7;
    let cells = n * n * n;
    let source = GridIndex { x: 3, y: 3, z: 2 };
    let body_neighbor = GridIndex { x: 3, y: 3, z: 3 };
    let mut body = vec![false; cells];
    body[flat_index(body_neighbor, n)] = true;
    let aperture = Nonlinear3dAperture {
        sources: vec![source],
        receivers: Vec::new(),
        therapy_points_m: vec![Point3 {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 0.0,
        }],
        receiver_points_m: Vec::new(),
        model_name: "coupling_stencil_test".to_owned(),
        source_domain: SourceDomain::ExteriorCoupling,
        focus: GridIndex { x: 3, y: 3, z: 4 },
    };
    let speed = vec![1500.0; cells];

    let plan = build_source_plan(
        &speed,
        n,
        1.0e-3,
        &aperture,
        SourceEncoding { index: 0, count: 1 },
        Some(&body),
    );

    let stencil = &plan.source_stencils[0];
    assert!(
        stencil.len() > 1,
        "finite-area exterior source must distribute over a coupling patch"
    );
    let max_weight = stencil
        .iter()
        .map(|(_, weight)| *weight)
        .fold(0.0, f64::max);
    assert!(
        (max_weight - 1.0).abs() < 1.0e-12,
        "finite-area exterior pressure source must impose full drive at its apodization peak"
    );
    assert!(
        stencil.iter().all(|(cell, _)| !body[*cell]),
        "exterior coupling source stencil must not inject directly into body cells"
    );
    assert!(
        !stencil
            .iter()
            .any(|(cell, _)| *cell == flat_index(body_neighbor, n)),
        "body-facing source neighbor must be excluded from finite source support"
    );
    let weight_sum = stencil.iter().map(|(_, weight)| *weight).sum::<f64>();
    assert!(
        weight_sum > 1.0,
        "pressure-boundary finite area must not be point-source sum-normalized"
    );

    let metrics = source_plan_metrics(
        &speed,
        n,
        1.0e-3,
        &aperture,
        SourceEncoding { index: 0, count: 1 },
        Some(&body),
    );
    assert!(metrics.source_support_min > 1);
    assert!(metrics.source_support_mean > 1.0);
    assert!(metrics.source_support_max >= metrics.source_support_min);
    assert!(metrics.focused_delay_span_s >= 0.0);
}
