//! Generate transcranial-UST reconstruction figure data: a 2-D head phantom
//! (skull annulus + brain structures), a known rigid misalignment of the CT
//! template, and the MOFI guidance-free alignment recovered from the **acoustic
//! data alone** with the exact-adjoint engine (ADR 016/017; Ch18 §10.1, Ch26
//! §26.4.1).
//!
//! Writes CSV grids (patient / misaligned-template / MOFI-aligned / error) to
//! `target/book_data/transcranial/`, consumed by
//! `crates/kwavers-python/examples/book/ch26_transcranial_reconstruction.py`.
//!
//! Run: `cargo run -p kwavers-solver --example transcranial_ust_reconstruction`

use kwavers_grid::Grid;
use kwavers_solver::inverse::fwi::time_domain::{
    mofi_align_homotopy, mofi_default_homotopy, mofi_transform, FwiEngine, FwiGeometry,
    FwiProcessor, MofiConfig, RigidTransform,
};
use kwavers_solver::inverse::seismic::parameters::FwiParameters;
use kwavers_source::{GridSource, SourceMode};
use ndarray::{Array2, Array3};
use std::io::Write;

const NX: usize = 64;
const NY: usize = 64;
const DX: f64 = 2e-3; // 2 mm → 12.8 cm field of view
const C_WATER: f64 = 1500.0;
const C_BRAIN: f64 = 1540.0;
const C_CSF: f64 = 1500.0;
const C_LESION: f64 = 1600.0;
const C_SKULL: f64 = 2600.0;

/// Build a 2-D axial head slice: water coupling, an elliptical skull annulus,
/// brain soft tissue with two ventricles, a midline falx, and an off-centre
/// lesion. The geometry is left–right asymmetric so rotation is identifiable.
fn head_phantom() -> Array3<f64> {
    let mut m = Array3::from_elem((NX, NY, 1), C_WATER);
    let (cx, cy) = (32.0_f64, 32.0_f64);
    let ellipse = |i: usize, j: usize, a: f64, b: f64| -> f64 {
        ((i as f64 - cx) / a).powi(2) + ((j as f64 - cy) / b).powi(2)
    };
    for j in 0..NY {
        for i in 0..NX {
            let outer = ellipse(i, j, 27.0, 23.0);
            let inner = ellipse(i, j, 23.0, 19.0);
            if outer <= 1.0 && inner > 1.0 {
                m[[i, j, 0]] = C_SKULL; // skull (bone) annulus
            } else if inner <= 1.0 {
                m[[i, j, 0]] = C_BRAIN; // brain parenchyma
            }
        }
    }
    // Two lateral ventricles (CSF) — placed asymmetrically.
    for j in 0..NY {
        for i in 0..NX {
            if ellipse(i, j, 23.0, 19.0) <= 1.0 {
                let v_l = ((i as f64 - 26.0) / 4.0).powi(2) + ((j as f64 - 34.0) / 6.0).powi(2);
                let v_r = ((i as f64 - 38.0) / 3.5).powi(2) + ((j as f64 - 31.0) / 5.0).powi(2);
                if v_l <= 1.0 || v_r <= 1.0 {
                    m[[i, j, 0]] = C_CSF;
                }
                // Falx midline (thin vertical structure, slightly slow).
                if (i as f64 - cx).abs() < 0.9 && (j as f64 - cy).abs() < 16.0 {
                    m[[i, j, 0]] = C_CSF;
                }
                // Off-centre lesion (target).
                let les = ((i as f64 - 40.0) / 3.0).powi(2) + ((j as f64 - 38.0) / 3.0).powi(2);
                if les <= 1.0 {
                    m[[i, j, 0]] = C_LESION;
                }
            }
        }
    }
    m
}

fn dump_slice(dir: &std::path::Path, name: &str, field: &Array3<f64>) {
    let path = dir.join(format!("{name}.csv"));
    let mut f = std::fs::File::create(&path).expect("create csv");
    for j in 0..NY {
        let row: Vec<String> = (0..NX)
            .map(|i| format!("{:.2}", field[[i, j, 0]]))
            .collect();
        writeln!(f, "{}", row.join(",")).expect("write row");
    }
}

fn main() {
    let grid = Grid::new(NX, NY, 1, DX, DX, DX).expect("grid");
    let template = head_phantom();

    // Ring transducer cap: two orthogonal sources, receiver frame just inside.
    let nt = 320usize;
    let dt = 2.5e-7; // 2-D CFL c_skull·dt/dx ≈ 0.33
    let mut p_mask = Array3::zeros((NX, NY, 1));
    p_mask[[6, 32, 0]] = 1.0;
    p_mask[[32, 6, 0]] = 1.0;
    let mut p_signal = Array2::zeros((1, nt));
    for t in 0..28 {
        let phase = (t as f64) * 0.3;
        p_signal[[0, t]] = (-phase * phase / 12.0).exp() * (2.0 * phase).sin();
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
    let mut sensor_mask = Array3::from_elem((NX, NY, 1), false);
    for i in 4..NX - 4 {
        sensor_mask[[i, 4, 0]] = true;
        sensor_mask[[i, NY - 5, 0]] = true;
    }
    for j in 4..NY - 4 {
        sensor_mask[[4, j, 0]] = true;
        sensor_mask[[NX - 5, j, 0]] = true;
    }
    let geometry = FwiGeometry::new(source, sensor_mask);

    let processor = FwiProcessor::new(FwiParameters {
        nt,
        dt,
        frequency: 5e5,
        ..FwiParameters::default()
    })
    .with_engine(FwiEngine::SecondOrderSelfAdjoint);

    // The patient = CT template at an unknown rigid pose. Observed = its data.
    let phi_true = RigidTransform {
        theta_rad: 8.0_f64.to_radians(),
        delta_x_m: 4.0e-3,
        delta_y_m: -3.0e-3,
    };
    let patient = mofi_transform(&template, &phi_true, &grid, C_WATER);
    let observed = processor
        .generate_synthetic_data(&patient, &geometry, &grid)
        .expect("observed");

    // Recover the pose from the acoustic data alone (no guidance image). A misfit
    // homotopy (Wasserstein → envelope → L2) widens the capture basin so the large
    // 8° skull rotation is recovered where single-misfit L2 would cycle-skip.
    let config = MofiConfig {
        max_iterations: 60,
        initial_step_m: 6e-3,
        background_c: C_WATER,
        tolerance: 1e-8,
        ..MofiConfig::default()
    };
    let result = mofi_align_homotopy(
        &processor,
        &template,
        &observed,
        &geometry,
        &grid,
        &mofi_default_homotopy(config),
    )
    .expect("MOFI homotopy");
    let aligned = mofi_transform(&template, &result.transform, &grid, C_WATER);
    let mut error = aligned.clone();
    for (e, &p) in error.iter_mut().zip(patient.iter()) {
        *e = (*e - p).abs();
    }

    let d_theta = (result.transform.theta_rad - phi_true.theta_rad)
        .to_degrees()
        .abs();
    let d_trans = (((result.transform.delta_x_m - phi_true.delta_x_m).powi(2)
        + (result.transform.delta_y_m - phi_true.delta_y_m).powi(2))
    .sqrt())
        * 1e3;

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/book_data/transcranial");
    std::fs::create_dir_all(&dir).expect("mkdir");
    dump_slice(&dir, "patient", &patient);
    dump_slice(&dir, "template_initial", &template);
    dump_slice(&dir, "mofi_aligned", &aligned);
    dump_slice(&dir, "error", &error);
    let meta = format!(
        "dx_mm,{}\nc_water,{}\nc_brain,{}\nc_skull,{}\n\
         theta_true_deg,{:.4}\ntheta_rec_deg,{:.4}\ndtheta_deg,{:.4}\n\
         dx_true_mm,{:.3}\ndy_true_mm,{:.3}\ndtrans_mm,{:.4}\n\
         initial_misfit,{:e}\nfinal_misfit,{:e}\niterations,{}\n",
        DX * 1e3,
        C_WATER,
        C_BRAIN,
        C_SKULL,
        phi_true.theta_rad.to_degrees(),
        result.transform.theta_rad.to_degrees(),
        d_theta,
        phi_true.delta_x_m * 1e3,
        phi_true.delta_y_m * 1e3,
        d_trans,
        result.initial_misfit,
        result.final_misfit,
        result.iterations,
    );
    std::fs::write(dir.join("meta.csv"), meta).expect("meta");

    println!("transcranial UST reconstruction data → {}", dir.display());
    println!(
        "  recovered pose: θ={:.3}° (true 8.000°, |Δ|={:.3}°), |Δδ|={:.3} mm; misfit {:.2e}→{:.2e}",
        result.transform.theta_rad.to_degrees(),
        d_theta,
        d_trans,
        result.initial_misfit,
        result.final_misfit
    );
    assert!(
        d_theta < 1.5 && d_trans < 1.5,
        "alignment should be sub-degree/sub-mm"
    );
}
