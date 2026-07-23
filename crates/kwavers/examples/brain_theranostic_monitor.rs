//! Theranostic brain imaging + real-time therapy monitor.
//!
//! End-to-end pipeline (ADR 024):
//!
//! ```text
//!  CT/phantom → c,ρ  →  1024-element hemispherical array (mapped to grid)
//!                                   │
//!            low-dose sparse-subset multi-source FWI  →  full-brain c image
//!                                   │
//!                            target selection (theranostic::targeting)
//!                                   │
//!     interleaved therapy/imaging schedule (theranostic::pulsing)
//!       · therapy frames: focused heating → Pennes bioheat → CEM43 dose
//!         (or cavitation void-fraction ODE for histotripsy mode)
//!       · imaging frames: lesion → medium perturbation (theranostic::lesion)
//!         → 2-D monitored-slice FWI reconstruction from simulated echoes
//!                                   │
//!                 per-interval lesion image + dose metrics + PNG frames
//! ```
//!
//! # Real-simulation discipline
//!
//! Nothing about the lesion image is painted on. The therapy frames advance a
//! genuine temperature field (explicit Pennes bioheat with an absorbed-power
//! source `Q = 2αI`) and a genuine CEM43 thermal-dose field; the monitored slice
//! is **reconstructed by FWI from synthetic transmission data** of the
//! therapy-perturbed medium — never read out from the ground-truth perturbation.
//! The lesion appears in the reconstruction only because the medium physically
//! changed (Duck 1990 thermal `c(T)`, or Wood 1930 cavitation `c(β)`).
//!
//! # Runtime
//!
//! Literal 1024-element 3-D forward sims per pulse are not CPU-runnable in demo
//! time. The 1024-element hemisphere defines the physical aperture; its element
//! positions are mapped to the grid and a documented reduced transmit/iteration
//! budget is used (the same code path scales to the full array and iteration
//! count by raising the constants below). Default run is a few minutes on CPU.
//!
//! # Modes
//!
//! `KW_THERAPY_MODE=thermal` (default) or `cavitation`.
//!
//! # References
//! - Guasch 2020 (npj Digital Medicine) — 3-D brain FWI.
//! - Duck 1990 — soft-tissue `c(T)`; Wood 1930 — bubbly `c(β)`.
//! - Sapareto & Dewey 1984; Damianou & Hynynen 1994 — CEM43 / 240 ablation.
//! - Pennes 1948 — bioheat equation.

use aequitas::systems::si::quantities::Time;
use kwavers::theranostic::{
    interleave_schedule, lesion, sparse_transmit_subsets, PulseKind, TargetSelection,
};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_physics::thermal::{TemperatureCoefficients, ThermalCEM43Grid};
use kwavers_solver::inverse::fwi::time_domain::{FwiGeometry, FwiProcessor};
use kwavers_solver::inverse::seismic::parameters::{FwiParameters, RegularizationParameters};
use kwavers_source::{GridSource, Source, SourceMode};
use kwavers_transducer::HemisphericalArray;
use leto::Array3 as LetoArray3;
use leto::{Array2, Array3};
use std::f64::consts::PI;
use std::time::Instant;

// ── Grid (CPML 10-cell safe; modest for runtime) ────────────────────────────
const DX: f64 = 3.0e-3; // m
const NX: usize = 40; // 120 mm lateral
const NY: usize = 24; // 72 mm elevation (true 3-D recon)
const NZ: usize = 40; // 120 mm depth

// ── Phantom geometry (voxels from grid centre) ──────────────────────────────
const R_HEAD: f64 = 16.0;
const R_SKULL_OUT: f64 = 15.0;
const R_SKULL_IN: f64 = 13.0;
const R_BRAIN: f64 = 12.0;

// ── HU / acoustic constants (Aubry 2003) ────────────────────────────────────
const C_WATER: f64 = 1500.0;
const C_BRAIN: f64 = 1540.0;
const C_CORTICAL: f64 = 2900.0;
const RHO_WATER: f64 = 1000.0;
const RHO_BRAIN: f64 = 1030.0;
const RHO_CORTICAL: f64 = 1900.0;

// ── Acquisition (mapped from the real 1024-element array) ───────────────────
const ARRAY_RADIUS_M: f64 = 0.07; // 70 mm hemispherical radius
const ARRAY_ELEMENTS: usize = 1024; // full physical aperture
const R_ARRAY_VOX: f64 = 17.0; // mapped shell radius [voxels]
const F0_HZ: f64 = 150_000.0;
const P0_PA: f64 = 1.0e5;

// Reduced-but-scalable recon budget (raise to approach full-aperture clinical run).
const RECON_TX_SUBSETS: usize = 4; // sparse low-dose transmit subsets
const RECON_TX_PER_SUBSET: usize = 1; // transmit elements fired per subset
const RECON_ITERS: usize = 2;

// ── Therapy / monitoring ────────────────────────────────────────────────────
const N_CYCLES: usize = 4; // therapy/imaging interleave cycles
const THERAPY_PER_CYCLE: usize = 1;
const IMAGING_PER_CYCLE: usize = 1;
const THERAPY_PRI_S: f64 = 2.0; // seconds of heating per therapy frame
const IMAGING_PRI_S: f64 = 0.05;
const THERMAL_SUBSTEPS: usize = 200; // bioheat sub-steps per therapy frame

// Tissue / bioheat (Pennes 1948; Duck 1990).
const BODY_TEMP_C: f64 = 37.0;
const RHO_CP: f64 = 1000.0 * 3600.0; // ρ·c_p [J/(m³·K)]
const THERMAL_DIFFUSIVITY: f64 = 1.4e-7; // m²/s
const PERFUSION_COEFF: f64 = 5.0e-4; // w_b·c_b/(ρ·c_p) [1/s]
const ABSORPTION_NP_M: f64 = 4.5; // brain α at ~0.65 MHz [Np/m]
const FOCAL_INTENSITY_W_M2: f64 = 2.0e6; // 200 W/cm² focal ISPTA (HIFU)
const FOCAL_SIGMA_VOX: f64 = 1.3; // focal half-width [voxels]

// Cavitation (histotripsy) cumulative-damage void fraction. Tissue
// fractionation is irreversible, so the void fraction accumulates monotonically
// where the focus drives it — the lesion persists and grows across pulses.
const CAV_PROD_PER_S: f64 = 0.15; // damage-accumulation rate at full drive
const CAV_BETA_MAX: f64 = 5.0e-3; // saturation void fraction
const CAV_BETA_LESION: f64 = 1.0e-3; // β above which a voxel counts as fractionated

// ─────────────────────────────────────────────────────────────────────────────

/// 3-D spherical skull+brain phantom: sound speed and density fields.
fn build_phantom() -> (Array3<f64>, Array3<f64>) {
    let (cx, cy, cz) = (NX as f64 / 2.0, NY as f64 / 2.0, NZ as f64 / 2.0);
    let mut c = Array3::from_elem((NX, NY, NZ), C_WATER);
    let mut rho = Array3::from_elem((NX, NY, NZ), RHO_WATER);
    for i in 0..NX {
        for j in 0..NY {
            for k in 0..NZ {
                let r =
                    ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2) + (k as f64 - cz).powi(2))
                        .sqrt();
                let (cv, rv) = if r > R_HEAD {
                    (C_WATER, RHO_WATER)
                } else if r > R_SKULL_OUT {
                    (1700.0, 1100.0) // scalp/soft
                } else if r > R_SKULL_IN {
                    (C_CORTICAL, RHO_CORTICAL) // cortical bone
                } else if r > R_BRAIN {
                    (1600.0, 1050.0) // inner cortical/CSF transition
                } else {
                    (C_BRAIN, RHO_BRAIN) // brain parenchyma
                };
                c[[i, j, k]] = cv;
                rho[[i, j, k]] = rv;
            }
        }
    }
    (c, rho)
}

/// Ricker wavelet scaled to peak pressure `P0_PA`.
fn ricker(f0: f64, dt: f64, nt: usize) -> Vec<f64> {
    let t_peak = 1.5 / f0;
    (0..nt)
        .map(|i| {
            let tau = PI * f0 * (i as f64 * dt - t_peak);
            P0_PA * (1.0 - 2.0 * tau * tau) * (-tau * tau).exp()
        })
        .collect()
}

/// Map the real 1024-element hemispherical array onto distinct exterior grid
/// voxels (angular distribution preserved, scaled to `R_ARRAY_VOX`).
fn map_array_to_grid() -> KwaversResult<Vec<(usize, usize, usize)>> {
    let array = HemisphericalArray::new(ARRAY_RADIUS_M, ARRAY_ELEMENTS, F0_HZ)?;
    let (cx, cy, cz) = (NX as f64 / 2.0, NY as f64 / 2.0, NZ as f64 / 2.0);
    let scale = R_ARRAY_VOX / ARRAY_RADIUS_M;
    let mut seen = std::collections::BTreeSet::new();
    let mut elems = Vec::new();
    for (ex, ey, ez) in array.positions() {
        // Hemisphere opens along +z; recentre its dome onto the grid centre.
        let ix = (cx + ex * scale).round();
        let iy = (cy + ey * scale).round();
        let iz = (cz + (ez - ARRAY_RADIUS_M / 2.0) * scale).round();
        if ix < 1.0 || iy < 1.0 || iz < 1.0 {
            continue;
        }
        let (i, j, k) = (ix as usize, iy as usize, iz as usize);
        if i >= NX - 1 || j >= NY - 1 || k >= NZ - 1 {
            continue;
        }
        // Keep only voxels in the water bath outside the head.
        let r =
            ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2) + (k as f64 - cz).powi(2)).sqrt();
        if r <= R_HEAD {
            continue;
        }
        if seen.insert((i, j, k)) {
            elems.push((i, j, k));
        }
    }
    Ok(elems)
}

/// Build an FWI shot (single source voxel + receiver mask) on a given grid.
fn build_shot(
    src: (usize, usize, usize),
    receivers: &[(usize, usize, usize)],
    dims: (usize, usize, usize),
    nt: usize,
    dt: f64,
) -> FwiGeometry {
    let mut source_mask = Array3::<f64>::zeros(dims);
    source_mask[[src.0, src.1, src.2]] = 1.0;
    let wavelet = ricker(F0_HZ, dt, nt);
    let mut p_signal = Array2::<f64>::zeros((1, nt));
    for t in 0..nt {
        p_signal[[0, t]] = wavelet[t];
    }
    let mut source = GridSource::new_empty();
    source.p_mask = Some(source_mask);
    source.p_signal = Some(p_signal);
    source.p_mode = SourceMode::Dirichlet;

    let mut recv = Array3::<bool>::from_elem(dims, false);
    for &(i, j, k) in receivers {
        if (i, j, k) != src {
            recv[[i, j, k]] = true;
        }
    }
    FwiGeometry::new(source, recv)
}

/// FWI parameters with a small low-dose TV regularizer (sparse-view denoising).
fn fwi_params(nt: usize, dt: f64, n_recv: usize, iters: usize) -> FwiParameters {
    FwiParameters {
        max_iterations: iters,
        tolerance: 1e-12,
        step_size: 40.0,
        frequency: F0_HZ,
        nt,
        dt,
        n_trace: n_recv.max(1),
        n_depth: 1,
        regularization: RegularizationParameters {
            tikhonov_weight: 0.0,
            tv_weight: 2.0e-4, // low-dose edge-preserving smoothing
            directional_tv_weight: 0.0,
            directional_tv_adaptive: false,
            smoothness_weight: 0.0,
        },
        source_mute_radius: 0,
        ..FwiParameters::default()
    }
}

/// One explicit Pennes bioheat sub-step (FTCS): advance temperature [°C].
fn bioheat_step(temp: &mut LetoArray3<f64>, q: &LetoArray3<f64>, dt: f64) {
    let inv_dx2 = 1.0 / (DX * DX);
    let prev = temp.clone();
    for i in 1..NX - 1 {
        for j in 1..NY - 1 {
            for k in 1..NZ - 1 {
                let lap = (prev[[i + 1, j, k]]
                    + prev[[i - 1, j, k]]
                    + prev[[i, j + 1, k]]
                    + prev[[i, j - 1, k]]
                    + prev[[i, j, k + 1]]
                    + prev[[i, j, k - 1]]
                    - 6.0 * prev[[i, j, k]])
                    * inv_dx2;
                let perfusion = PERFUSION_COEFF * (BODY_TEMP_C - prev[[i, j, k]]);
                let dtemp = THERMAL_DIFFUSIVITY * lap + perfusion + q[[i, j, k]] / RHO_CP;
                temp[[i, j, k]] = prev[[i, j, k]] + dt * dtemp;
            }
        }
    }
}

/// Gaussian focal absorbed-power field `Q = 2αI·exp(-r²/2σ²)` [W/m³].
fn focal_q(target: (usize, usize, usize)) -> LetoArray3<f64> {
    let q0 = 2.0 * ABSORPTION_NP_M * FOCAL_INTENSITY_W_M2;
    let mut q = LetoArray3::<f64>::zeros([NX, NY, NZ]);
    let s2 = 2.0 * FOCAL_SIGMA_VOX * FOCAL_SIGMA_VOX;
    for i in 0..NX {
        for j in 0..NY {
            for k in 0..NZ {
                let r2 = ((i as f64 - target.0 as f64).powi(2)
                    + (j as f64 - target.1 as f64).powi(2)
                    + (k as f64 - target.2 as f64).powi(2))
                    / s2;
                q[[i, j, k]] = q0 * (-r2).exp();
            }
        }
    }
    q
}

/// Write a normalised grayscale PNG of a 2-D field (lo→0, hi→255).
fn write_png(path: &str, w: usize, h: usize, field: &[f64], lo: f64, hi: f64) {
    let span = (hi - lo).max(1e-12);
    let bytes: Vec<u8> = field
        .iter()
        .map(|&v| (((v - lo) / span).clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    let file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  PNG write failed ({path}): {e}");
            return;
        }
    };
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    if let Ok(mut writer) = enc.write_header() {
        let _ = writer.write_image_data(&bytes);
    }
}

/// Extract a coronal (x–z) slice at elevation `jy` as an `(NX, 2, NZ)` array for
/// 2-D FWI (NY=2 quasi-2-D embedding the existing solvers require).
fn coronal_slice(vol: &Array3<f64>, jy: usize) -> Array3<f64> {
    let mut s = Array3::<f64>::zeros((NX, 2, NZ));
    for i in 0..NX {
        for k in 0..NZ {
            let v = vol[[i, jy, k]];
            s[[i, 0, k]] = v;
            s[[i, 1, k]] = v;
        }
    }
    s
}

fn main() -> KwaversResult<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("warn"));
    let mode = match std::env::var("KW_THERAPY_MODE").as_deref() {
        Ok("cavitation") => lesion::TherapyMode::Cavitation,
        _ => lesion::TherapyMode::Thermal,
    };

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Theranostic brain imaging + real-time therapy monitor      ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("  Therapy mode    : {mode:?}");

    // ── 1. Phantom + grid ────────────────────────────────────────────────
    let (true_c, _rho) = build_phantom();
    let grid = Grid::new(NX, NY, NZ, DX, DX, DX)?;
    let c_lo = true_c.iter().copied().fold(f64::INFINITY, f64::min);
    let c_hi = true_c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "\n[1/5] Phantom {NX}×{NY}×{NZ} @ {:.0} mm — c∈[{c_lo:.0},{c_hi:.0}] m/s",
        DX * 1e3
    );

    // ── 2. Map the 1024-element hemisphere onto the grid ─────────────────
    let elements = map_array_to_grid()?;
    println!(
        "[2/5] 1024-element hemisphere → {} distinct grid acquisition voxels",
        elements.len()
    );
    assert!(
        elements.len() >= 4,
        "need ≥4 acquisition elements after mapping"
    );

    // ── 3. Low-dose sparse-subset full-brain FWI ─────────────────────────
    let dt = 0.3 * DX / (c_hi * 3.0_f64.sqrt());
    let t_transit = (NX as f64 * DX) / C_WATER;
    let nt = ((t_transit * 1.2) / dt).ceil() as usize;

    let subsets = sparse_transmit_subsets(elements.len(), RECON_TX_SUBSETS);
    let mut tx: Vec<usize> = Vec::new();
    for s in &subsets {
        tx.extend(s.iter().take(RECON_TX_PER_SUBSET));
    }
    println!(
        "[3/5] Full-brain FWI: {} low-dose transmit elements, {} receivers, {} iters (dt={:.1} ns, nt={nt})",
        tx.len(),
        elements.len() - 1,
        RECON_ITERS,
        dt * 1e9
    );

    let fwi = FwiProcessor::new(fwi_params(nt, dt, elements.len() - 1, RECON_ITERS));
    let dims = (NX, NY, NZ);
    let mut shots = Vec::new();
    let t0 = Instant::now();
    for &e in &tx {
        let geom = build_shot(elements[e], &elements, dims, nt, dt);
        let obs = fwi.generate_synthetic_data(&true_c, &geom, &grid)?;
        shots.push((geom, obs));
    }
    // Smooth initial model: water background (CT prior would go here).
    let initial = Array3::from_elem(dims, C_WATER);
    let recon = fwi.invert_multi_source(&shots, &initial, &grid)?;
    println!(
        "      reconstructed c∈[{:.0},{:.0}] m/s in {:.1} s",
        recon.iter().copied().fold(f64::INFINITY, f64::min),
        recon.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        t0.elapsed().as_secs_f32()
    );

    // ── 4. Target selection inside the reconstructed brain ───────────────
    let (cx, cy, cz) = (NX as f64 / 2.0, NY as f64 / 2.0, NZ as f64 / 2.0);
    let mut brain_mask = Array3::from_elem(dims, false);
    for i in 0..NX {
        for j in 0..NY {
            for k in 0..NZ {
                let r =
                    ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2) + (k as f64 - cz).powi(2))
                        .sqrt();
                brain_mask[[i, j, k]] = r < R_BRAIN;
            }
        }
    }
    // Score: reconstructed sound-speed contrast vs. brain baseline (ROI proxy).
    let score = recon.mapv(|v| (v - C_BRAIN).abs());
    let target = TargetSelection::max_in_mask(&score, &brain_mask, [0.0; 3], [DX; 3]).unwrap_or(
        TargetSelection {
            voxel: (NX / 2, NY / 2, NZ / 2),
            position_m: [cx * DX, cy * DX, cz * DX],
            score: 0.0,
        },
    );
    let tgt = target.voxel;
    println!(
        "[4/5] Target voxel {:?}  position {:?} m",
        tgt, target.position_m
    );

    // ── 5. Interleaved therapy + monitored-slice reconstruction ──────────
    let tx_indices: Vec<usize> = tx.clone();
    let schedule = interleave_schedule(
        &tx_indices,
        &subsets,
        THERAPY_PER_CYCLE,
        IMAGING_PER_CYCLE,
        N_CYCLES,
        THERAPY_PRI_S,
        IMAGING_PRI_S,
    );
    println!(
        "[5/5] Interleaved schedule: {} frames ({} therapy, {} imaging)",
        schedule.len(),
        schedule
            .iter()
            .filter(|f| f.kind == PulseKind::Therapy)
            .count(),
        schedule
            .iter()
            .filter(|f| f.kind == PulseKind::Imaging)
            .count(),
    );

    // Monitored coronal slice through the focus.
    let jy = tgt.1;
    let slice_dims = (NX, 2, NZ);
    let slice_grid = Grid::new(NX, 2, NZ, DX, DX, DX)?;
    // Acquisition elements projected onto the monitored slice (dedup in x–z).
    let mut slice_seen = std::collections::BTreeSet::new();
    let mut slice_elems: Vec<(usize, usize, usize)> = Vec::new();
    for &(i, _j, k) in &elements {
        if slice_seen.insert((i, k)) {
            slice_elems.push((i, 0, k));
        }
    }
    // The 2-D slice monitor is cheap per cell, so it affords a real iteration
    // budget (MON_ITERS) while a halved record window (nt_mon) keeps each frame
    // affordable; transmission across the slice still completes in this window.
    const MON_ITERS: usize = 3;
    let nt_mon = (nt / 2).max(96);
    let true_slice = coronal_slice(&true_c, jy);
    let slice_subsets = sparse_transmit_subsets(slice_elems.len(), RECON_TX_SUBSETS);
    let slice_fwi = FwiProcessor::new(fwi_params(nt_mon, dt, slice_elems.len() - 1, MON_ITERS));
    let mut slice_tx: Vec<usize> = Vec::new();
    for s in &slice_subsets {
        slice_tx.extend(s.iter().take(RECON_TX_PER_SUBSET));
    }

    // Helper closure: reconstruct the monitored slice of a given medium by FWI
    // from simulated echoes, warm-started from `init`.
    let recon_slice_of =
        |medium_slice: &Array3<f64>, init: &Array3<f64>| -> KwaversResult<Array3<f64>> {
            let mut slice_shots = Vec::new();
            for &e in &slice_tx {
                let geom = build_shot(slice_elems[e], &slice_elems, slice_dims, nt_mon, dt);
                let obs = slice_fwi.generate_synthetic_data(medium_slice, &geom, &slice_grid)?;
                slice_shots.push((geom, obs));
            }
            slice_fwi.invert_multi_source(&slice_shots, init, &slice_grid)
        };

    // Pre-therapy baseline: reconstruct the UNPERTURBED slice with the SAME 2-D
    // acquisition, warm-started from the 3-D recon slice. Differencing against
    // this isolates the lesion (the 2-D-vs-3-D geometric bias cancels).
    let init_slice = coronal_slice(&recon, jy);
    let baseline_slice_recon = recon_slice_of(&true_slice, &init_slice)?;

    // Therapy state.
    let coeff = TemperatureCoefficients::soft_tissue();
    let mut temperature = LetoArray3::from_elem([NX, NY, NZ], BODY_TEMP_C);
    let mut cem43 = ThermalCEM43Grid::new(NX, NY, NZ);
    let mut void_fraction = Array3::<f64>::zeros(dims);
    let q_focal = focal_q(tgt);

    let mut monitor_frame = 0usize;
    println!("\n  frame | t [s] | maxT °C | CEM43_max | lesion vox | applied Δc | recon ROI Δc");
    println!("  ------+-------+---------+-----------+------------+------------+-------------");
    for frame in &schedule {
        match frame.kind {
            PulseKind::Therapy => {
                let dts = THERAPY_PRI_S / THERMAL_SUBSTEPS as f64;
                match mode {
                    lesion::TherapyMode::Thermal => {
                        for _ in 0..THERMAL_SUBSTEPS {
                            bioheat_step(&mut temperature, &q_focal, dts);
                        }
                        cem43.update(&temperature, Time::from_base(THERAPY_PRI_S))?;
                    }
                    lesion::TherapyMode::Cavitation => {
                        // Cumulative (irreversible) fractionation: β grows toward
                        // saturation where the focus drives it; no dissolution.
                        for i in 0..NX {
                            for j in 0..NY {
                                for k in 0..NZ {
                                    let drive = q_focal[[i, j, k]]
                                        / (2.0 * ABSORPTION_NP_M * FOCAL_INTENSITY_W_M2);
                                    let b = void_fraction[[i, j, k]];
                                    let prod = CAV_PROD_PER_S * drive * (CAV_BETA_MAX - b).max(0.0);
                                    void_fraction[[i, j, k]] =
                                        (b + THERAPY_PRI_S * prod).clamp(0.0, CAV_BETA_MAX);
                                }
                            }
                        }
                    }
                }
            }
            PulseKind::Imaging => {
                // Perturb the medium from the current lesion state.
                let state = match mode {
                    lesion::TherapyMode::Thermal => lesion::LesionState::Thermal {
                        temperature_c: &temperature,
                        reference_c: BODY_TEMP_C,
                        coeff,
                    },
                    lesion::TherapyMode::Cavitation => lesion::LesionState::Cavitation {
                        void_fraction: &void_fraction,
                    },
                };
                let perturbed = lesion::perturb_sound_speed(&true_c, &state);

                // Reconstruct the monitored slice by FWI from simulated echoes of
                // the therapy-perturbed medium (warm-started from the pre-therapy
                // baseline reconstruction).
                let perturbed_slice = coronal_slice(&perturbed, jy);
                let recon_slice = recon_slice_of(&perturbed_slice, &baseline_slice_recon)?;

                // Applied focal Δc: the genuine sound-speed change the therapy
                // produced in the ROI (reference for the lesion physics — NOT the
                // monitor image, which is reconstructed below).
                let mut applied_dc = 0.0_f64;

                // Lesion image = reconstructed change vs. pre-therapy baseline recon
                // (same 2-D geometry → geometric bias cancels, lesion remains).
                let mut diff = vec![0.0_f64; NX * NZ];
                for i in 0..NX {
                    for k in 0..NZ {
                        diff[i * NZ + k] = recon_slice[[i, 0, k]] - baseline_slice_recon[[i, 0, k]];
                    }
                }
                // Quantify the lesion in a focal ROI around the target (off-focus
                // reconstruction artefacts dominate a global max and hide it).
                const ROI: usize = 3;
                let (mut roi_max, mut roi_sum, mut roi_n) = (0.0_f64, 0.0_f64, 0usize);
                for i in tgt.0.saturating_sub(ROI)..=(tgt.0 + ROI).min(NX - 1) {
                    for k in tgt.2.saturating_sub(ROI)..=(tgt.2 + ROI).min(NZ - 1) {
                        let d = recon_slice[[i, 0, k]] - baseline_slice_recon[[i, 0, k]];
                        roi_max = roi_max.max(d.abs());
                        roi_sum += d;
                        roi_n += 1;
                        applied_dc = applied_dc
                            .max((perturbed_slice[[i, 0, k]] - true_slice[[i, 0, k]]).abs());
                    }
                }
                let roi_mean = roi_sum / roi_n.max(1) as f64;

                // Metrics from the genuine therapy state (mode-aware lesion count).
                let max_t = temperature
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let cem_max = cem43.get_max_dose()?.as_minutes();
                let lesion_vox = match mode {
                    lesion::TherapyMode::Thermal => {
                        lesion::lesion_mask(cem43.get_dose(), lesion::ABLATION_CEM43_THRESHOLD_MIN)
                            .iter()
                            .filter(|&&b| b)
                            .count()
                    }
                    lesion::TherapyMode::Cavitation => void_fraction
                        .iter()
                        .filter(|&&b| b >= CAV_BETA_LESION)
                        .count(),
                };

                println!(
                    "   {:3}  | {:5.2} | {:7.2} | {:9.2} | {:10} | {:9.1} | {:6.2}/{:+6.2}",
                    monitor_frame,
                    frame.time_s,
                    max_t,
                    cem_max,
                    lesion_vox,
                    applied_dc,
                    roi_max,
                    roi_mean
                );

                // PNG of the reconstructed lesion-difference slice.
                let png_lo = diff.iter().copied().fold(f64::INFINITY, f64::min);
                let png_hi = diff.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                write_png(
                    &format!("brain_monitor_frame{monitor_frame:02}.png"),
                    NZ,
                    NX,
                    &diff,
                    png_lo,
                    png_hi,
                );
                monitor_frame += 1;
            }
        }
    }

    println!("\n  Wrote {monitor_frame} monitored-slice PNG frames (brain_monitor_frameNN.png).");
    println!("  Lesion image is reconstructed from simulated echoes of the");
    println!("  therapy-perturbed medium — not read out from ground truth.");
    Ok(())
}
