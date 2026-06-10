//! Genuine (non-inverse-crime-idealised) transcranial brain FWI reconstruction.
//!
//! Unlike the MOFI *registration* demo (`transcranial_ust_reconstruction`, Fig
//! 26.8), here the brain anomaly is **unknown to the starting model** and is
//! recovered pixel-wise from **noisy** multi-shot data with the skull frozen to
//! its known value (`invert_multi_source_masked`; Guasch 2020). The result is
//! deliberately *imperfect* — that is the honest behaviour of an ill-posed
//! limited-aperture inversion.
//!
//! Note on realism: the data are still synthesised on the same grid/solver as the
//! inversion (a same-grid synthetic), but additive measurement noise + the
//! genuinely unknown anomaly + the inversion's ill-posedness make the recovery
//! realistic (the misfit does not collapse to round-off). Cross-grid data
//! generation is the remaining step toward a true crime-free benchmark.
//!
//! Run: `cargo run -p kwavers-solver --example transcranial_brain_fwi`

use kwavers_grid::Grid;
use kwavers_math::inverse_problems::tv_denoise_chambolle;
use kwavers_solver::inverse::fwi::time_domain::{FwiEngine, FwiGeometry, FwiProcessor};
use kwavers_solver::inverse::seismic::parameters::{FwiParameters, RegularizationParameters};
use kwavers_source::{GridSource, SourceMode};
use ndarray::{Array2, Array3};
use std::io::Write;

const NX: usize = 64;
const NY: usize = 64;
const DX: f64 = 2.2e-3;
const C_WATER: f64 = 1500.0;
const C_SKULL: f64 = 1520.0; // thin penetrable skull/coupling ring (frozen/known)
const C_BRAIN: f64 = 1540.0;
const C_ANOMALY: f64 = 1700.0; // the UNKNOWN lesion to be recovered (+160 m/s)
const NOISE_PCT: f64 = 5.0; // measurement noise (so there are real artefacts for PnP to clean)

fn ellipse(i: usize, j: usize, cx: f64, cy: f64, a: f64, b: f64) -> f64 {
    ((i as f64 - cx) / a).powi(2) + ((j as f64 - cy) / b).powi(2)
}

/// Skull annulus + homogeneous brain (the *known* reference / starting model).
fn reference_head() -> (Array3<f64>, Array3<bool>) {
    let mut m = Array3::from_elem((NX, NY, 1), C_WATER);
    let mut skull = Array3::from_elem((NX, NY, 1), false);
    for j in 0..NY {
        for i in 0..NX {
            let outer = ellipse(i, j, 32.0, 32.0, 27.0, 24.0);
            let inner = ellipse(i, j, 32.0, 32.0, 23.0, 20.0);
            if outer <= 1.0 && inner > 1.0 {
                m[[i, j, 0]] = C_SKULL;
                skull[[i, j, 0]] = true;
            } else if inner <= 1.0 {
                m[[i, j, 0]] = C_BRAIN;
            }
        }
    }
    (m, skull)
}

fn dump(dir: &std::path::Path, name: &str, f: &Array3<f64>) {
    let mut file = std::fs::File::create(dir.join(format!("{name}.csv"))).expect("csv");
    for j in 0..NY {
        let row: Vec<String> = (0..NX).map(|i| format!("{:.2}", f[[i, j, 0]])).collect();
        writeln!(file, "{}", row.join(",")).expect("row");
    }
}

fn main() {
    let grid = Grid::new(NX, NY, 1, DX, DX, DX).expect("grid");
    let (reference, skull_mask) = reference_head();

    // Ground truth = reference + an unknown off-centre brain anomaly.
    let mut truth = reference.clone();
    for j in 0..NY {
        for i in 0..NX {
            if !skull_mask[[i, j, 0]] && ellipse(i, j, 40.0, 36.0, 6.0, 6.0) <= 1.0 {
                truth[[i, j, 0]] = C_ANOMALY;
            }
        }
    }

    let nt = 400usize;
    let dt = 3e-7; // SA 2-D leapfrog stable (c·dt/dx ≈ 0.27 ≪ 1/√2)
    let wavelet = |n: usize| {
        let mut s = Array2::zeros((1, nt));
        for t in 0..30 {
            let phase = (t as f64) * 0.28;
            s[[0, t]] = (-phase * phase / 14.0).exp() * (2.0 * phase).sin();
        }
        let _ = n;
        s
    };
    // Receiver frame just inside the domain (shared by all shots).
    let mut sensor_mask = Array3::from_elem((NX, NY, 1), false);
    for i in 5..NX - 5 {
        sensor_mask[[i, 5, 0]] = true;
        sensor_mask[[i, NY - 6, 0]] = true;
    }
    for j in 5..NY - 5 {
        sensor_mask[[5, j, 0]] = true;
        sensor_mask[[NX - 6, j, 0]] = true;
    }
    // Eight shots around the cap for angular coverage.
    let src_positions = [
        (6usize, 32usize),
        (57, 32),
        (32, 6),
        (32, 57),
        (14, 14),
        (49, 49),
        (14, 49),
        (49, 14),
    ];

    let make_processor = |max_iter: usize| {
        FwiProcessor::new(FwiParameters {
            nt,
            dt,
            frequency: 5e5,
            max_iterations: max_iter,
            step_size: 35.0,
            tolerance: 1e-7,
            source_mute_radius: 3,
            regularization: RegularizationParameters {
                tikhonov_weight: 0.0,
                tv_weight: 0.0,
                smoothness_weight: 0.0,
            },
            ..FwiParameters::default()
        })
        .with_engine(FwiEngine::SecondOrderSelfAdjoint)
    };
    let processor = make_processor(28);

    // Build shots: observed = forward(truth) + 2% additive Gaussian-ish noise
    // (deterministic LCG so the figure is reproducible).
    let mut lcg: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next_noise = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Box–Muller-ish: map two uniforms to an approximately normal sample.
        let u = ((lcg >> 11) as f64) / ((1u64 << 53) as f64);
        2.0 * u - 1.0
    };
    let mut shots: Vec<(FwiGeometry, Array2<f64>)> = Vec::new();
    for &(si, sj) in &src_positions {
        let mut p_mask = Array3::zeros((NX, NY, 1));
        p_mask[[si, sj, 0]] = 1.0;
        let source = GridSource {
            p0: None,
            u0: None,
            p_mask: Some(p_mask),
            p_signal: Some(wavelet(0)),
            p_mode: SourceMode::Additive,
            u_mask: None,
            u_signal: None,
            u_mode: SourceMode::default(),
        };
        let geometry = FwiGeometry::new(source, sensor_mask.clone());
        let mut data = processor
            .generate_synthetic_data(&truth, &geometry, &grid)
            .expect("forward");
        let rms = (data.iter().map(|v| v * v).sum::<f64>() / data.len() as f64).sqrt();
        let sigma = (NOISE_PCT / 100.0) * rms;
        data.mapv_inplace(|v| v + sigma * next_noise());
        shots.push((geometry, data));
    }

    // Start from the homogeneous brain (no anomaly); skull frozen to known value.
    let start = reference.clone();
    let recon = processor
        .invert_multi_source_masked(
            &shots,
            &start,
            &reference,
            &skull_mask,
            C_BRAIN - 20.0,
            C_ANOMALY + 40.0,
            &grid,
        )
        .expect("masked FWI");

    // Plug-and-Play (CS-MRI / CT-MBIR style): alternate short masked-FWI bursts
    // with an edge-preserving TV denoiser on the brain (skull frozen). The prior
    // suppresses streak artefacts between data-fidelity steps while preserving the
    // lesion boundary (Venkatakrishnan 2013; Lustig 2007).
    let pnp_processor = make_processor(4);
    let mut recon_pnp = start.clone();
    for _ in 0..7 {
        recon_pnp = pnp_processor
            .invert_multi_source_masked(
                &shots,
                &recon_pnp,
                &reference,
                &skull_mask,
                C_BRAIN - 20.0,
                C_ANOMALY + 40.0,
                &grid,
            )
            .expect("PnP masked FWI burst");
        recon_pnp = tv_denoise_chambolle(&recon_pnp, 0.4, 60, Some(&skull_mask));
    }

    // Brain-region accuracy (skull excluded): RMS error, lesion-peak recovery,
    // and normalised model error ‖rec−true‖/‖true−homogeneous‖.
    let brain_metrics = |model: &Array3<f64>| -> (f64, f64, f64) {
        let (mut sq_err, mut sq_sig, mut n) = (0.0, 0.0, 0usize);
        let (mut peak_true, mut peak_rec) = (C_BRAIN, C_BRAIN);
        for j in 0..NY {
            for i in 0..NX {
                if !skull_mask[[i, j, 0]] && ellipse(i, j, 32.0, 32.0, 23.0, 20.0) <= 1.0 {
                    let t = truth[[i, j, 0]];
                    let r = model[[i, j, 0]];
                    sq_err += (r - t).powi(2);
                    sq_sig += (t - C_BRAIN).powi(2);
                    n += 1;
                    if ellipse(i, j, 40.0, 36.0, 6.0, 6.0) <= 1.0 {
                        peak_true = peak_true.max(t);
                        peak_rec = peak_rec.max(r);
                    }
                }
            }
        }
        let rms = (sq_err / n as f64).sqrt();
        let frac = (peak_rec - C_BRAIN) / (peak_true - C_BRAIN);
        let rel = (sq_err / sq_sig.max(1e-9)).sqrt();
        (rms, frac, rel)
    };
    let (rms_err, recovered_frac, model_rel_err) = brain_metrics(&recon);
    let (rms_pnp, frac_pnp, rel_pnp) = brain_metrics(&recon_pnp);

    let abs_err = |m: &Array3<f64>| {
        let mut e = m.clone();
        for (v, &t) in e.iter_mut().zip(truth.iter()) {
            *v = (*v - t).abs();
        }
        e
    };
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/book_data/brain_fwi");
    std::fs::create_dir_all(&dir).expect("mkdir");
    dump(&dir, "truth", &truth);
    dump(&dir, "start", &start);
    dump(&dir, "recon", &recon);
    dump(&dir, "error", &abs_err(&recon));
    dump(&dir, "recon_pnp", &recon_pnp);
    dump(&dir, "error_pnp", &abs_err(&recon_pnp));
    let meta = format!(
        "dx_mm,{}\nc_water,{}\nc_skull,{}\nc_brain,{}\nc_anomaly,{}\nnoise_pct,{}\nshots,{}\n\
         iterations,28\nbrain_rms_err_m_s,{:.3}\nlesion_peak_recovered_frac,{:.3}\nmodel_rel_err,{:.4}\n\
         pnp_rms_err_m_s,{:.3}\npnp_lesion_peak_recovered_frac,{:.3}\npnp_model_rel_err,{:.4}\n",
        DX * 1e3, C_WATER, C_SKULL, C_BRAIN, C_ANOMALY, NOISE_PCT, src_positions.len(),
        rms_err, recovered_frac, model_rel_err, rms_pnp, frac_pnp, rel_pnp,
    );
    std::fs::write(dir.join("meta.csv"), meta).expect("meta");

    println!("genuine brain FWI reconstruction → {}", dir.display());
    println!(
        "  plain FWI : RMS {:.1} m/s, lesion {:.0}%, rel.err {:.2}",
        rms_err,
        100.0 * recovered_frac,
        model_rel_err
    );
    println!(
        "  PnP+TV    : RMS {:.1} m/s, lesion {:.0}%, rel.err {:.2}",
        rms_pnp,
        100.0 * frac_pnp,
        rel_pnp
    );
}
