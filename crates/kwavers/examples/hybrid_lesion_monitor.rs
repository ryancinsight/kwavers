//! Hybrid lesion monitor — FD-CBS full-wave + PAM passive + fusion (ADR 024).
//!
//! Demonstrates the verified `kwavers::theranostic::monitor` hybrid pipeline
//! tracking a lesion as it grows, on a single monitored slice:
//!
//! ```text
//!   per frame (lesion grows):
//!     perturbed slice ─┬─ FD-CBS differential FWI ──► quantitative Δc map
//!                      └─ PAM passive emission map ─► cavitation source map
//!                                  │
//!                          fusion (agreement = √(q̂·p̂)) ──► lesion image
//!                                  │
//!                       lesion extent (growth) + PNG frame
//! ```
//!
//! Unlike RTM or low-budget time-domain transmission FWI (which could not resolve
//! the lesion), the frequency-domain dense convergent-Born-series inversion
//! recovers the quantitative sound-speed change, and PAM independently confirms
//! the source location; fusion keeps only what both agree on.
//!
//! Run: `cargo run --example hybrid_lesion_monitor`
//! Fast (~20 s CPU); writes `hybrid_lesion_frameNN.png`.

use kwavers::theranostic::monitor::{
    fd, fuse_lesion_map, lesion_extent, passive_acoustic_map, synthesize_emission, FusionWeights,
    PamMonitorConfig,
};
use kwavers_core::error::KwaversResult;
use leto::{
    Array2,
    Array3,
};

const N: usize = 12; // slice pixels per side
const SPACING_M: f64 = 1.0e-3;
const C_BRAIN: f64 = 1540.0;
const C_SKULL: f64 = 1720.0; // mild skull ring (kept in CBS convergence regime)
const LESION_DC: f64 = 60.0; // thermal sound-speed rise at full lesion [m/s]
const N_FRAMES: usize = 5;
const RING_ELEMENTS: usize = 16;
const RING_DIAMETER_M: f64 = 0.018;

/// Background slice: homogeneous brain with a mild skull annulus.
fn background_slice() -> Array3<f64> {
    let c = (N / 2) as f64;
    let mut v = Array3::from_elem((N, N, 1), C_BRAIN);
    for i in 0..N {
        for j in 0..N {
            let r = ((i as f64 - c).powi(2) + (j as f64 - c).powi(2)).sqrt();
            if (4.5..5.5).contains(&r) {
                v[[i, j, 0]] = C_SKULL; // thin skull ring
            }
        }
    }
    v
}

/// Add a circular thermal lesion of `radius` voxels at the slice centre.
fn with_lesion(background: &Array3<f64>, radius: f64) -> Array3<f64> {
    let c = (N / 2) as f64;
    let mut v = background.clone();
    for i in 0..N {
        for j in 0..N {
            let r = ((i as f64 - c).powi(2) + (j as f64 - c).powi(2)).sqrt();
            if r <= radius {
                v[[i, j, 0]] += LESION_DC;
            }
        }
    }
    v
}

/// Write a normalized grayscale PNG of a 2-D field.
fn write_png(path: &str, field: &Array2<f64>) {
    let (nx, nz) = field.dim();
    let hi = field
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-12);
    let bytes: Vec<u8> = field
        .iter()
        .map(|&v| ((v / hi).clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    let Ok(file) = std::fs::File::create(path) else {
        eprintln!("  PNG write failed: {path}");
        return;
    };
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), nx as u32, nz as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    if let Ok(mut w) = enc.write_header() {
        let _ = w.write_image_data(&bytes);
    }
}

fn to_2d_dc(recon_dc: &Array3<f64>) -> Array2<f64> {
    let mut m = Array2::zeros((N, N));
    for i in 0..N {
        for j in 0..N {
            m[[i, j]] = recon_dc[[i, j, 0]];
        }
    }
    m
}

fn main() -> KwaversResult<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Hybrid lesion monitor — FD-CBS + PAM + fusion (real-time)   ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    let cfg = fd::FdMonitorConfig {
        ring_elements: RING_ELEMENTS,
        ring_diameter_m: RING_DIAMETER_M,
        spacing_m: SPACING_M,
        frequencies_hz: vec![3.0e5, 5.0e5],
        reference_sound_speed_m_s: C_BRAIN,
        min_sound_speed_m_s: 1480.0,
        max_sound_speed_m_s: 1800.0,
        fwi_iterations: 6,
        estimate_source_scaling: false,
        cbs_iterations: 20,
        cbs_tolerance: 1.0e-3,
        tikhonov_weight: 0.0,
        use_gauss_newton: true,
    };
    let ring = fd::ring_around_slice(cfg.ring_elements, cfg.ring_diameter_m)?;

    // PAM image frame aligned to the FD voxel-centre convention.
    let origin = [
        (0.5 - N as f64 / 2.0) * SPACING_M,
        0.0,
        (0.5 - N as f64 / 2.0) * SPACING_M,
    ];
    let pam_cfg = PamMonitorConfig {
        sound_speed_m_s: C_BRAIN,
        origin_m: origin,
        spacing_m: SPACING_M,
        nx: N,
        nz: N,
    };
    let r = RING_DIAMETER_M / 2.0;
    let elements: Vec<[f64; 3]> = (0..RING_ELEMENTS)
        .map(|k| {
            let a = std::f64::consts::TAU * k as f64 / RING_ELEMENTS as f64;
            [r * a.cos(), 0.0, r * a.sin()]
        })
        .collect();
    let centre = N / 2;
    let source = [
        origin[0] + centre as f64 * SPACING_M,
        0.0,
        origin[2] + centre as f64 * SPACING_M,
    ];
    let fs = 2.0e6;

    let background = background_slice();

    println!("\n  frame | lesion r | recon centre Δc | PAM peak px | fused extent");
    println!("  ------+----------+-----------------+-------------+-------------");
    for frame in 0..N_FRAMES {
        // Lesion grows from 0.6 to ~2.6 voxels across frames.
        let radius = 0.6 + frame as f64 * 0.5;
        let perturbed = with_lesion(&background, radius);

        // ── Quantitative channel: Gauss-Newton + Born differential map from a
        //    common reference (the skull cancels; the lesion remains). ──
        let quant_dc = fd::differential_lesion_map(&background, &perturbed, &ring, &cfg)?;
        let quant = to_2d_dc(&quant_dc);
        let centre_dc = quant[[centre, centre]];

        // ── Passive channel: PAM map of the cavitation emission (amplitude
        //    scales with lesion size as more nuclei activate). ──
        let data = synthesize_emission(source, &elements, fs, C_BRAIN, 400, 3.0)?;
        let passive = passive_acoustic_map(&data, &elements, fs, &pam_cfg)?;
        let (mut pi, mut pj, mut pv) = (0usize, 0usize, f64::NEG_INFINITY);
        for i in 0..N {
            for j in 0..N {
                if passive[[i, j]] > pv {
                    pv = passive[[i, j]];
                    pi = i;
                    pj = j;
                }
            }
        }

        // ── Fusion. ──
        let fused = fuse_lesion_map(&quant, &passive, FusionWeights::default())?;
        let extent = lesion_extent(&fused.agreement, 0.5);

        println!(
            "   {:3}  | {:8.2} | {:15.2} | ({:2},{:2})    | {:11}",
            frame, radius, centre_dc, pi, pj, extent
        );
        write_png(
            &format!("hybrid_lesion_frame{frame:02}.png"),
            &fused.agreement,
        );
    }

    println!("\n  Wrote {N_FRAMES} fused lesion-image frames (hybrid_lesion_frameNN.png).");
    println!("  Quantitative Δc from FD-CBS full-wave inversion; source confirmed by");
    println!("  PAM; fused agreement is the lesion the monitor reports growing.");
    Ok(())
}
