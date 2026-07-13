//! Guidance-free skull-template alignment with the exact-adjoint engine (MOFI).
//!
//! Companion to the *Inverse Problems* (§2.5, §10.1) and *Transcranial UST*
//! (§26.4.1) book chapters. It builds an asymmetric 2-D sound-speed template,
//! applies a known rigid misalignment, generates the acoustic data with the
//! **self-adjoint exact-gradient engine** (`FwiEngine::SecondOrderSelfAdjoint`,
//! ADR 016), and recovers the pose by manifold optimisation of the *acoustic*
//! misfit alone — no guidance image. It prints figure-ready metrics
//! (initial/final misfit, recovered vs true pose, errors) and emits a CSV.
//!
//! Run: `cargo run -p kwavers-solver --example mofi_exact_adjoint_demo`

use kwavers_grid::Grid;
use kwavers_solver::inverse::fwi::time_domain::{
    mofi_align, mofi_transform, FwiEngine, FwiGeometry, FwiProcessor, MofiConfig, RigidTransform,
};
use kwavers_solver::inverse::seismic::parameters::FwiParameters;
use kwavers_source::{GridSource, SourceMode};
use leto::{Array2, Array3};

fn main() {
    let (nx, ny) = (32usize, 32);
    let dx = 1e-3;
    let grid = Grid::new(nx, ny, 1, dx, dx, dx).expect("grid");
    let c0 = 1500.0;

    // Asymmetric two-blob template (asymmetry makes rotation identifiable).
    let mut template = Array3::from_elem((nx, ny, 1), c0);
    for j in 0..ny {
        for i in 0..nx {
            let r1 = (i as f64 - 11.0).powi(2) + (j as f64 - 13.0).powi(2);
            let r2 = (i as f64 - 21.0).powi(2) + (j as f64 - 19.0).powi(2);
            template[[i, j, 0]] += 500.0 * (-r1 / 8.0).exp() + 300.0 * (-r2 / 8.0).exp();
        }
    }

    // Ring-array acquisition: two orthogonal sources, inset receiver frame.
    let nt = 200usize;
    let mut p_mask = Array3::zeros((nx, ny, 1));
    p_mask[[2, 16, 0]] = 1.0;
    p_mask[[16, 2, 0]] = 1.0;
    let mut p_signal = Array2::zeros((1, nt));
    for t in 0..24 {
        let phase = (t as f64) * 0.35;
        p_signal[[0, t]] = (-phase * phase / 9.0).exp() * (2.0 * phase).sin();
    }
    let source = GridSource {
        p0: None,
        u0: None,
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Additive,
        u_mask: None,
        u_signal: None,
        u_mode: SourceMode::default(),
    };
    let mut sensor_mask = Array3::from_elem((nx, ny, 1), false);
    for i in 3..nx - 3 {
        sensor_mask[[i, 3, 0]] = true;
        sensor_mask[[i, ny - 4, 0]] = true;
    }
    for j in 3..ny - 3 {
        sensor_mask[[3, j, 0]] = true;
        sensor_mask[[nx - 4, j, 0]] = true;
    }
    let geometry = FwiGeometry::new(source, sensor_mask);

    let processor = FwiProcessor::new(FwiParameters {
        nt,
        dt: 2e-7,
        frequency: 5e5,
        ..FwiParameters::default()
    })
    .with_engine(FwiEngine::SecondOrderSelfAdjoint);

    // Known misalignment → observed data from the transformed template.
    let phi_true = RigidTransform {
        theta_rad: 6.0_f64.to_radians(),
        delta_x_m: 2.0e-3,
        delta_y_m: -1.5e-3,
    };
    let true_model = mofi_transform(&template, &phi_true, &grid, c0);
    let observed = processor
        .generate_synthetic_data(&true_model, &geometry, &grid)
        .expect("observed data");

    let config = MofiConfig {
        max_iterations: 100,
        initial_step_m: 4e-3,
        background_c: c0,
        tolerance: 1e-8,
        ..MofiConfig::default()
    };
    let result = mofi_align(&processor, &template, &observed, &geometry, &grid, &config)
        .expect("MOFI alignment");

    let d_theta = (result.transform.theta_rad - phi_true.theta_rad)
        .to_degrees()
        .abs();
    let d_trans = (((result.transform.delta_x_m - phi_true.delta_x_m).powi(2)
        + (result.transform.delta_y_m - phi_true.delta_y_m).powi(2))
    .sqrt())
        * 1e3;

    println!("MOFI exact-adjoint guidance-free alignment");
    println!("  iterations         : {}", result.iterations);
    println!(
        "  misfit  initial→final: {:.3e} → {:.3e}  ({:.1}% reduction)",
        result.initial_misfit,
        result.final_misfit,
        100.0 * (1.0 - result.final_misfit / result.initial_misfit)
    );
    println!(
        "  rotation  true/rec   : {:.3}° / {:.3}°   (|Δθ| = {:.3}°)",
        phi_true.theta_rad.to_degrees(),
        result.transform.theta_rad.to_degrees(),
        d_theta
    );
    println!(
        "  translation true/rec : ({:.2},{:.2}) / ({:.2},{:.2}) mm   (|Δδ| = {:.3} mm)",
        phi_true.delta_x_m * 1e3,
        phi_true.delta_y_m * 1e3,
        result.transform.delta_x_m * 1e3,
        result.transform.delta_y_m * 1e3,
        d_trans
    );

    // Figure-ready CSV (one summary row; consumable by a plotting script).
    let csv = format!(
        "metric,value\ninitial_misfit,{:e}\nfinal_misfit,{:e}\nd_theta_deg,{:.6}\nd_trans_mm,{:.6}\niterations,{}\n",
        result.initial_misfit, result.final_misfit, d_theta, d_trans, result.iterations
    );
    let path = std::env::temp_dir().join("mofi_exact_adjoint_demo.csv");
    if std::fs::write(&path, csv).is_ok() {
        println!("  wrote summary CSV    : {}", path.display());
    }

    assert!(
        d_theta < 1.0 && d_trans < 1.0,
        "alignment must reach <1°/<1mm"
    );
}
