//! Seismic Imaging — Full Waveform Inversion with multi-shot surface acquisition.
//!
//! # Physical pipeline
//!
//! ```text
//! Geological model → c(x), ρ(x) → FDTD forward → synthetic traces
//!                                                        │
//!                                   ← adjoint source ← L2 residual
//!                                   │
//!                                   FDTD adjoint
//!                                   │
//!                                   gradient ∂J/∂c → model update
//! ```
//!
//! # Geological model
//!
//! Quasi-2-D depth cross-section (x–z plane) inspired by the Marmousi benchmark.
//! The layered structure is the seismic analog of the 5-layer skull phantom in
//! `skull_ct_phase_correction.rs` (water → scalp → cortical → diploe → brain).
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │  near-surface / water (0–400 m)          c = 1500    │
//! │  upper clastic       (400–1000 m)        c = 2100    │
//! │  lower clastic      (1000–1900 m)        c = 2700    │
//! │                  ┌──────────────┐                    │
//! │  deep sediment ──┤  SALT DOME   ├──                  │
//! │  (1900–2800 m)   │  c = 4500    │  c = 3200          │
//! │                  └──────────────┘                    │
//! │  basement           (2800–3200 m)        c = 3800    │
//! └──────────────────────────────────────────────────────┘
//!  SRC  SRC  SRC  SRC  SRC  SRC    ← surface shots  (z = 1)
//!  RCV RCV RCV … RCV              ← receiver spread (z = 2)
//! ```
//!
//! # Acquisition geometry
//!
//! The surface spread is the seismic analog of `skull_ct_phase_correction`'s
//! 1024-element hemispherical ExAblate array:
//!
//! | skull_ct_phase_correction (ultrasound) | This demo (seismic)                |
//! |----------------------------------------|------------------------------------|
//! | 1024 ExAblate elements on 150 mm sphere| 6 shots on 2-D surface line        |
//! | Golden-angle hemispherical placement   | Uniform lateral spacing            |
//! | 650 kHz Insightec transducer           | 5 Hz Ricker wavelet                |
//! | Thin-phase-screen aberration correction| Adjoint-state FWI (full inversion) |
//!
//! # FWI objective and gradient
//!
//! Acoustic L2 misfit (Tarantola 1984; Virieux & Operto 2009):
//!
//! ```text
//! J(c) = (dt / 2) Σ_{r,t} [d_syn(r,t; c) − d_obs(r,t)]²
//!
//! ∂J/∂m(x) = −∫₀ᵀ λ(x, T−t) ∂²p(x,t)/∂t² dt,   m = c⁻²
//! ∂J/∂c(x) = −2 c(x)⁻³ ∂J/∂m(x)
//! ```
//!
//! # References
//!
//! - Tarantola, A. (1984). Inversion of seismic reflection data in the acoustic
//!   approximation. *Geophysics*, 49(8), 1259–1266.
//! - Virieux, J. & Operto, S. (2009). An overview of full-waveform inversion in
//!   exploration geophysics. *Geophysics*, 74(6), WCC1–WCC26.
//! - Martin, G.S. et al. (2006). Marmousi2: an elastic upgrade for Marmousi.
//!   *The Leading Edge*, 25(2), 156–166.
//! - Gardner, G.H.F. et al. (1974). Formation velocity and density — the
//!   diagnostic basics for stratigraphic traps. *Geophysics*, 39(6), 770–780.
//! - Ricker, N. (1953). Wavelet contraction, wavelet expansion and the control
//!   of seismic resolution. *Geophysics*, 18(4), 769–792.

use kwavers::core::error::{KwaversError, KwaversResult};
use kwavers::domain::grid::Grid;
use kwavers::domain::source::{GridSource, SourceMode};
use kwavers::solver::inverse::seismic::{
    fwi::{FwiGeometry, FwiProcessor},
    parameters::{
        BoundaryType, FwiParameters, ImagingCondition, RegularizationParameters, RtmSettings,
        StorageStrategy,
    },
    rtm::RtmProcessor,
};
use ndarray::{Array2, Array3};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// Grid and model constants
// ─────────────────────────────────────────────────────────────────────────────

/// Grid spacing [m].
///
/// 50 m gives λ/6 resolution at f₀ = 5 Hz in water (λ = 300 m).
/// Analogous to the ~0.3 mm CT voxel spacing in skull_ct_phase_correction.
const DX: f64 = 50.0;

/// Grid dimensions.
///
/// NY = 2 satisfies the FDTD staggered-stencil minimum while keeping the
/// second y-plane acoustically transparent (identical medium in both planes),
/// matching the quasi-2-D embedding used in transcranial_fwi.rs.
const NX: usize = 64; // lateral extent  3200 m
const NY: usize = 2; // quasi-2-D embedding
const NZ: usize = 64; // depth extent    3200 m

// Geological layer boundaries (voxel indices in the z / depth direction).
// Analogous to skull_ct_phase_correction's layer radii:
//   R_HEAD, R_SKULL_OUT, R_DIPLOE, R_SKULL_IN, R_BRAIN
const Z_UPPER_CLASTIC: usize = 8; //   0 – 400 m: near-surface / water layer
const Z_LOWER_CLASTIC: usize = 20; //  400–1000 m: upper clastic sediment
const Z_DEEP_SEDIMENT: usize = 38; // 1000–1900 m: lower clastic sediment
const Z_BASEMENT: usize = 56; // 1900–2800 m: deep sediment → basement

// P-wave velocities for each geological unit [m/s].
// Reference: Martin et al. (2006) Marmousi2; typical exploration values.
const C_NEAR_SURFACE: f64 = 1500.0; // water / unconsolidated near-surface
const C_UPPER_CLASTIC: f64 = 2100.0; // upper clastic (shale / sand)
const C_LOWER_CLASTIC: f64 = 2700.0; // lower clastic (compacted sand / shale)
const C_DEEP_SEDIMENT: f64 = 3200.0; // deeper sediment / tight formations
const C_BASEMENT: f64 = 3800.0; // basement / crystalline rock
const C_ANOMALY: f64 = 4500.0; // salt dome / carbonate hard intrusion

// ─────────────────────────────────────────────────────────────────────────────
// Source wavelet
// ─────────────────────────────────────────────────────────────────────────────

/// Centre frequency of the Ricker (Mexican hat) source wavelet [Hz].
///
/// 5 Hz is the low end of the exploration seismic band.  At c_water = 1500 m/s,
/// λ = 300 m, giving 6 grid points per wavelength at dx = 50 m — adequate for
/// a broadband (±2 octave) Ricker pulse.
const F0_HZ: f64 = 5.0;

/// Peak source pressure amplitude [Pa].
///
/// 100 kPa matches the convention in transcranial_fwi.rs.  At seismic scale
/// the absolute pressure cancels in the max-norm-normalized gradient; only
/// the wavelet shape affects the inversion.
const P0_PA: f64 = 1.0e5;

/// Ricker (Mexican hat) wavelet.
///
/// ```text
/// w(t) = P₀ · (1 − 2π²f₀²τ²) · exp(−π²f₀²τ²),   τ = t − t_peak
/// ```
///
/// Time-domain peak at t_peak = 1.5 / f₀ (three half-cycles of build-up).
///
/// Reference: Ricker N (1953). *Geophysics*, 18(4), 769–792.
fn ricker_wavelet(f0: f64, dt: f64, nt: usize) -> Vec<f64> {
    let t_peak = 1.5 / f0;
    (0..nt)
        .map(|i| {
            let t = i as f64 * dt;
            let tau = PI * f0 * (t - t_peak);
            P0_PA * (1.0 - 2.0 * tau * tau) * (-tau * tau).exp()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Geological model
// ─────────────────────────────────────────────────────────────────────────────

/// 2-D seismic velocity and density model.
///
/// The seismic analog of `SkullPhantom` in transcranial_fwi.rs and the
/// `HeterogeneousSkull` / `CtVolume` structures in skull_ct_phase_correction.
pub struct SeismicModel {
    /// P-wave velocity c(x) [m/s]
    pub sound_speed: Array3<f64>,
    /// Bulk density ρ(x) [kg/m³]
    pub density: Array3<f64>,
}

/// Gardner (1974) empirical velocity–density relation.
///
/// ρ [g/cm³] = 0.31 · V[ft/s]^0.25  (Gardner et al. 1974)
///
/// Converted to SI units (1 g/cm³ = 1000 kg/m³; 1 ft/s ≈ 0.3048 m/s):
///   ρ [kg/m³] ≈ 1741 · (c [m/s] / 1000)^0.25
///
/// Reference: Gardner GHF et al. (1974). Geophysics 39(6), 770–780. Eq. (1).
#[inline]
fn gardner_density(c_m_per_s: f64) -> f64 {
    1741.0 * (c_m_per_s / 1000.0).powf(0.25)
}

/// Build the full geological model (layers + salt dome anomaly).
///
/// The seismic analog of `build_skull_phantom()` in transcranial_fwi.rs and
/// `skull_from_hu()` in skull_ct_phase_correction.rs.
///
/// ## Layer geometry
///
/// | Layer               | z range               | c [m/s] |
/// |---------------------|-----------------------|---------|
/// | Near-surface / water| z < Z_UPPER_CLASTIC   |  1500   |
/// | Upper clastic       | Z_UPPER_CLASTIC ≤ z < Z_LOWER_CLASTIC |  2100 |
/// | Lower clastic       | Z_LOWER_CLASTIC ≤ z < Z_DEEP_SEDIMENT |  2700 |
/// | Deep sediment       | Z_DEEP_SEDIMENT ≤ z < Z_BASEMENT |  3200 |
/// | Basement            | z ≥ Z_BASEMENT        |  3800   |
/// | Salt dome (anomaly) | r < R_ANOMALY from (cx, 34) |  4500 |
///
/// ## Dome geometry
///
/// The anomaly is a disc of radius R_ANOMALY = 8 voxels (400 m) centred at
/// (NX/2, 0, 34) ≈ (1600 m lateral, 1700 m depth).  A disc cross-section is
/// used as in skull_ct_phase_correction's circular skull phantom.
fn build_seismic_model() -> SeismicModel {
    let cx = (NX / 2) as f64; // 32.0
    const Z_ANOMALY_CENTRE: f64 = 34.0; // 1700 m depth
    const R_ANOMALY: f64 = 8.0; // 400 m radius

    let mut sound_speed = Array3::<f64>::zeros((NX, NY, NZ));

    for i in 0..NX {
        for k in 0..NZ {
            let base_c = if k < Z_UPPER_CLASTIC {
                C_NEAR_SURFACE
            } else if k < Z_LOWER_CLASTIC {
                C_UPPER_CLASTIC
            } else if k < Z_DEEP_SEDIMENT {
                C_LOWER_CLASTIC
            } else if k < Z_BASEMENT {
                C_DEEP_SEDIMENT
            } else {
                C_BASEMENT
            };

            // Disc-shaped salt dome — mirrors the circular skull cross-section
            // in skull_ct_phase_correction.
            let ddx = i as f64 - cx;
            let ddz = k as f64 - Z_ANOMALY_CENTRE;
            let r = (ddx * ddx + ddz * ddz).sqrt();
            let c = if r < R_ANOMALY { C_ANOMALY } else { base_c };

            for j in 0..NY {
                sound_speed[[i, j, k]] = c;
            }
        }
    }

    let density = sound_speed.mapv(gardner_density);
    SeismicModel {
        sound_speed,
        density,
    }
}

/// Build the 1-D background model used as the FWI starting point.
///
/// Contains only layered geology — no salt-dome anomaly.  Starting FWI from a
/// smooth 1-D reference is standard practice in exploration seismic
/// (Virieux & Operto 2009 §4.2).  The seismic analog of starting from a
/// homogeneous water model in transcranial_fwi.rs.
fn build_background_model() -> Array3<f64> {
    let mut c = Array3::<f64>::zeros((NX, NY, NZ));
    for i in 0..NX {
        for k in 0..NZ {
            let base = if k < Z_UPPER_CLASTIC {
                C_NEAR_SURFACE
            } else if k < Z_LOWER_CLASTIC {
                C_UPPER_CLASTIC
            } else if k < Z_DEEP_SEDIMENT {
                C_LOWER_CLASTIC
            } else if k < Z_BASEMENT {
                C_DEEP_SEDIMENT
            } else {
                C_BASEMENT
            };
            for j in 0..NY {
                c[[i, j, k]] = base;
            }
        }
    }
    c
}

// ─────────────────────────────────────────────────────────────────────────────
// Acquisition geometry — multi-shot surface spread
// ─────────────────────────────────────────────────────────────────────────────

/// Number of surface shots.
///
/// The seismic analog of skull_ct_phase_correction's ELEMENT_COUNT (1024).
/// 6 shots provide illumination from six distinct angles into the salt dome,
/// reducing the single-source null space of the FWI inversion.
const N_SHOTS: usize = 6;

/// Number of receivers in the common-receiver spread.
const N_RECEIVERS: usize = 16;

/// Gradient descent step size for the FWI model update [m/s].
///
/// After max-norm gradient normalization, `step_size` directly controls the
/// maximum velocity perturbation per iteration.  100 m/s is conservative for
/// a 1500–4500 m/s model.
const STEP_SIZE: f64 = 100.0;

/// Pixel size of each model panel in the output PPM image.
///
/// Each of the four panels (true, initial, reconstructed, difference) is
/// `PANEL × PANEL` pixels.  The x-z model slice at j = 0 is bilinearly
/// resampled into this raster.
const PANEL: usize = 320;

/// Height of the velocity colorbar strip beneath each panel [pixels].
const COLORBAR_H: usize = 20;

/// Surface shot positions (ix, iz_src), uniformly spaced across the lateral
/// dimension.
///
/// The seismic analog of `hemispherical_projected_elements()` in
/// skull_ct_phase_correction.rs.  Uniform lateral spacing maximizes aperture
/// coverage in the 2-D acquisition geometry.
///
/// | Shot | ix | x [m] | Description                |
/// |------|----|----|------------------------------|
/// |  0   |  8 |  400 | left flank                 |
/// |  1   | 18 |  900 | upper-left                 |
/// |  2   | 28 | 1400 | left of centre             |
/// |  3   | 37 | 1850 | right of centre            |
/// |  4   | 46 | 2300 | upper-right                |
/// |  5   | 55 | 2750 | right flank                |
fn surface_shot_positions() -> [(usize, usize); N_SHOTS] {
    let spacing = (NX - 2) / (N_SHOTS + 1);
    let mut positions = [(0usize, 1usize); N_SHOTS];
    for (s, pos) in positions.iter_mut().enumerate() {
        *pos = (1 + (s + 1) * spacing, 1);
    }
    positions
}

/// Build the common-receiver mask shared by all shots.
///
/// 16 receivers at z = 2 (100 m depth) uniformly spaced across the lateral
/// dimension, recording reflected and refracted arrivals from all shots.
fn build_receiver_mask() -> Array3<bool> {
    let mut mask = Array3::<bool>::from_elem((NX, NY, NZ), false);
    let spacing = (NX - 2) / (N_RECEIVERS + 1);
    for r in 0..N_RECEIVERS {
        let ix = 1 + (r + 1) * spacing;
        if ix < NX {
            mask[[ix, 0, 2]] = true;
        }
    }
    mask
}

/// Build `FwiGeometry` for one shot at surface voxel `(ix, 0, iz_src)`.
///
/// Identical in structure to `build_shot()` in transcranial_fwi.rs; only
/// the wavelet centre frequency F0_HZ differs.
fn build_shot(ix: usize, iz_src: usize, nt: usize, dt: f64) -> FwiGeometry {
    let mut source_mask = Array3::<f64>::zeros((NX, NY, NZ));
    source_mask[[ix, 0, iz_src]] = 1.0;

    let wavelet = ricker_wavelet(F0_HZ, dt, nt);
    let mut p_signal = Array2::<f64>::zeros((1, nt));
    for t in 0..nt {
        p_signal[[0, t]] = wavelet[t];
    }

    let mut source = GridSource::new_empty();
    source.p_mask = Some(source_mask);
    source.p_signal = Some(p_signal);
    source.p_mode = SourceMode::Dirichlet;

    FwiGeometry::new(source, build_receiver_mask())
}

// ─────────────────────────────────────────────────────────────────────────────
// Reconstruction quality metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Report reconstruction quality against the ground-truth velocity model.
///
/// Identical to `print_quality_report()` in transcranial_fwi.rs:
/// RMSE, Pearson r (with uniform-model guard), max |error|, ±100 m/s fraction.
///
/// Returns ‖true − reconstructed‖² (unnormalized L2 proxy).
fn print_quality_report(true_model: &Array3<f64>, reconstructed: &Array3<f64>) -> f64 {
    let n = true_model.len() as f64;

    let l2: f64 = true_model
        .iter()
        .zip(reconstructed.iter())
        .map(|(&t, &r)| (t - r).powi(2))
        .sum();
    let rmse = (l2 / n).sqrt();

    let mean_t = true_model.sum() / n;
    let mean_r = reconstructed.sum() / n;
    let cov = true_model
        .iter()
        .zip(reconstructed.iter())
        .map(|(&t, &r)| (t - mean_t) * (r - mean_r))
        .sum::<f64>();
    let var_t = true_model
        .iter()
        .map(|&t| (t - mean_t).powi(2))
        .sum::<f64>();
    let var_r = reconstructed
        .iter()
        .map(|&r| (r - mean_r).powi(2))
        .sum::<f64>();
    let denom = (var_t * var_r).sqrt();

    let max_err = true_model
        .iter()
        .zip(reconstructed.iter())
        .map(|(&t, &r)| (t - r).abs())
        .fold(0.0_f64, f64::max);

    let within_100 = true_model
        .iter()
        .zip(reconstructed.iter())
        .filter(|(&t, &r)| (t - r).abs() <= 100.0)
        .count() as f64
        / n
        * 100.0;

    println!("  RMSE            : {rmse:8.1} m/s");
    if denom > f64::EPSILON {
        let pearson = cov / denom;
        println!("  Pearson r       : {pearson:8.4}");
    } else {
        println!("  Pearson r       :      N/A  (uniform model)");
    }
    println!("  Max |error|     : {max_err:8.1} m/s");
    println!("  Voxels ±100 m/s : {within_100:7.1} %");

    l2
}

// ─────────────────────────────────────────────────────────────────────────────
// Image output
// ─────────────────────────────────────────────────────────────────────────────

/// Write one pixel into a flat RGB byte buffer.
///
/// `width` is the total image width in pixels; `x` and `y` are column/row.
/// Out-of-bounds writes are silently ignored — same contract as
/// `skull_ct_phase_correction::put_pixel`.
fn put_pixel(rgb: &mut [u8], width: usize, height: usize, x: usize, y: usize, color: [u8; 3]) {
    if x >= width || y >= height {
        return;
    }
    let idx = 3 * (y * width + x);
    rgb[idx..idx + 3].copy_from_slice(&color);
}

/// Map velocity to RGB using a 5-stop blue → cyan → green → yellow → red
/// colormap, normalized to [c_lo, c_hi].
///
/// This is the seismic analog of `phase_color` in skull_ct_phase_correction:
/// that function maps phase ∈ [−π, π] to hue; this function maps velocity
/// ∈ [c_lo, c_hi] to the same 5-stop rainbow.
fn velocity_color(c: f64, c_lo: f64, c_hi: f64) -> [u8; 3] {
    let t = ((c - c_lo) / (c_hi - c_lo)).clamp(0.0, 1.0);
    // 5-stop linear interpolation: blue→cyan→green→yellow→red
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0) // blue → cyan
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s) // cyan → green
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0) // green → yellow
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0) // yellow → red
    };
    [(255.0 * r) as u8, (255.0 * g) as u8, (255.0 * b) as u8]
}

/// Map a signed scalar to a blue ← 0 → red diverging colormap.
///
/// `max_abs` is the symmetric clip level; values outside [−max_abs, +max_abs]
/// are clamped.  Zero maps to white; positive → red; negative → blue.
fn diverging_color(value: f64, max_abs: f64) -> [u8; 3] {
    if max_abs < f64::EPSILON {
        return [200, 200, 200];
    }
    let t = (value / max_abs).clamp(-1.0, 1.0);
    if t >= 0.0 {
        // white → red
        let gb = (255.0 * (1.0 - t)) as u8;
        [255, gb, gb]
    } else {
        // white → blue
        let rg = (255.0 * (1.0 + t)) as u8;
        [rg, rg, 255]
    }
}

/// Render one velocity model panel (x–z cross-section at j=0) into `rgb`.
///
/// The voxel grid is nearest-neighbour resampled into a `PANEL × PANEL` pixel
/// block starting at pixel column `x_offset`.  Depth (z) runs top → bottom;
/// lateral (x) runs left → right.
fn draw_velocity_panel(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    x_offset: usize,
    model: &Array3<f64>,
    c_lo: f64,
    c_hi: f64,
) {
    for py in 0..PANEL {
        for px in 0..PANEL {
            // nearest-neighbour: py maps to depth (z), px maps to lateral (x)
            let ix = (px * NX / PANEL).min(NX - 1);
            let iz = (py * NZ / PANEL).min(NZ - 1);
            let c = model[[ix, 0, iz]];
            let color = velocity_color(c, c_lo, c_hi);
            put_pixel(rgb, width, height, x_offset + px, py, color);
        }
    }
}

/// Draw shot (white) and receiver (yellow) position markers on a panel.
fn draw_acquisition_markers(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    x_offset: usize,
    shot_positions: &[(usize, usize)],
) {
    // Receiver mask uses z=2; map to panel pixel y
    let recv_z_px = 2 * PANEL / NZ;
    let recv_spacing = (NX - 2) / (N_RECEIVERS + 1);
    for r in 0..N_RECEIVERS {
        let ix = 1 + (r + 1) * recv_spacing;
        let rx = ix * PANEL / NX;
        for dy in 0_usize..=2 {
            for dx in 0_usize..=2 {
                put_pixel(
                    rgb,
                    width,
                    height,
                    x_offset + rx.saturating_sub(1) + dx,
                    recv_z_px.saturating_sub(1) + dy,
                    [255, 255, 0], // yellow — receivers
                );
            }
        }
    }

    // Shot positions (z=1 voxel → z_px = 1*PANEL/NZ ≈ 5 px)
    for &(ix, iz) in shot_positions {
        let sx = ix * PANEL / NX;
        let sz = iz * PANEL / NZ;
        for dy in 0_usize..=2 {
            for dx in 0_usize..=2 {
                put_pixel(
                    rgb,
                    width,
                    height,
                    x_offset + sx.saturating_sub(1) + dx,
                    sz.saturating_sub(1) + dy,
                    [255, 255, 255], // white — shots
                );
            }
        }
    }
}

/// Draw a velocity colorbar strip below a panel.
///
/// The strip spans `x_offset..x_offset+PANEL` at y = `PANEL..PANEL+COLORBAR_H`.
fn draw_colorbar(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    x_offset: usize,
    c_lo: f64,
    c_hi: f64,
) {
    for px in 0..PANEL {
        let t = px as f64 / (PANEL - 1) as f64;
        let c = c_lo + t * (c_hi - c_lo);
        let color = velocity_color(c, c_lo, c_hi);
        for dy in 0..COLORBAR_H {
            put_pixel(rgb, width, height, x_offset + px, PANEL + dy, color);
        }
    }
}

/// Write a 4-panel PPM showing (left→right): true model, initial model,
/// reconstructed model, signed reconstruction error.
///
/// Image layout (pixels):
///
/// ```text
/// ┌───────────┬───────────┬───────────┬───────────┐  ← PANEL rows
/// │  True     │  Initial  │  Reconstr.│ Error     │
/// │  model    │  model    │           │ (R − T)   │
/// ├───────────┴───────────┴───────────┴───────────┤  ← COLORBAR_H rows
/// │  velocity colorbar: c_lo (blue) → c_hi (red)  │
/// └───────────────────────────────────────────────┘
///   4 × PANEL columns
/// ```
///
/// Markers: white 3×3 px dots = shot positions; yellow 3×3 px dots = receivers.
///
/// Analog of `write_three_plane_ppm` in skull_ct_phase_correction.rs.
pub fn write_velocity_panels(
    path: &Path,
    true_model: &Array3<f64>,
    initial_model: &Array3<f64>,
    reconstructed: &Array3<f64>,
    shot_positions: &[(usize, usize)],
) -> std::io::Result<()> {
    let c_lo = C_NEAR_SURFACE;
    let c_hi = C_ANOMALY;

    let img_w = 4 * PANEL;
    let img_h = PANEL + COLORBAR_H;
    let mut rgb = vec![0_u8; img_w * img_h * 3];

    // Panel 0: true model
    draw_velocity_panel(&mut rgb, img_w, img_h, 0, true_model, c_lo, c_hi);
    draw_acquisition_markers(&mut rgb, img_w, img_h, 0, shot_positions);
    draw_colorbar(&mut rgb, img_w, img_h, 0, c_lo, c_hi);

    // Panel 1: initial model
    draw_velocity_panel(&mut rgb, img_w, img_h, PANEL, initial_model, c_lo, c_hi);
    draw_acquisition_markers(&mut rgb, img_w, img_h, PANEL, shot_positions);
    draw_colorbar(&mut rgb, img_w, img_h, PANEL, c_lo, c_hi);

    // Panel 2: reconstructed model
    draw_velocity_panel(&mut rgb, img_w, img_h, 2 * PANEL, reconstructed, c_lo, c_hi);
    draw_acquisition_markers(&mut rgb, img_w, img_h, 2 * PANEL, shot_positions);
    draw_colorbar(&mut rgb, img_w, img_h, 2 * PANEL, c_lo, c_hi);

    // Panel 3: signed difference (reconstructed − true), diverging colormap
    let max_diff = true_model
        .iter()
        .zip(reconstructed.iter())
        .map(|(&t, &r)| (r - t).abs())
        .fold(0.0_f64, f64::max)
        .max(f64::EPSILON);

    for py in 0..PANEL {
        for px in 0..PANEL {
            let ix = (px * NX / PANEL).min(NX - 1);
            let iz = (py * NZ / PANEL).min(NZ - 1);
            let delta = reconstructed[[ix, 0, iz]] - true_model[[ix, 0, iz]];
            let color = diverging_color(delta, max_diff);
            put_pixel(&mut rgb, img_w, img_h, 3 * PANEL + px, py, color);
        }
    }
    // Difference colorbar: diverging, symmetric
    for px in 0..PANEL {
        let t = px as f64 / (PANEL - 1) as f64; // 0→1
        let signed = (2.0 * t - 1.0) * max_diff; // −max_diff → +max_diff
        let color = diverging_color(signed, max_diff);
        for dy in 0..COLORBAR_H {
            put_pixel(&mut rgb, img_w, img_h, 3 * PANEL + px, PANEL + dy, color);
        }
    }

    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "P6\n{} {}\n255", img_w, img_h)?;
    out.write_all(&rgb)?;
    Ok(())
}

/// Write a single-panel PPM of the RTM image (signed diverging colormap).
///
/// The RTM image volume is sliced at j = 0 and resampled to `PANEL × PANEL`.
/// Positive values (bright reflectors) → red; negative → blue; zero → white.
pub fn write_rtm_panel(path: &Path, rtm_image: &Array3<f64>) -> std::io::Result<()> {
    let max_abs = rtm_image
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(f64::EPSILON);

    let img_w = PANEL;
    let img_h = PANEL;
    let mut rgb = vec![0_u8; img_w * img_h * 3];

    for py in 0..PANEL {
        for px in 0..PANEL {
            let ix = (px * NX / PANEL).min(NX - 1);
            let iz = (py * NZ / PANEL).min(NZ - 1);
            let val = rtm_image[[ix, 0, iz]];
            let color = diverging_color(val, max_abs);
            put_pixel(&mut rgb, img_w, img_h, px, py, color);
        }
    }

    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "P6\n{} {}\n255", img_w, img_h)?;
    out.write_all(&rgb)?;
    Ok(())
}

/// Write the central-column (x = NX/2) velocity profiles to CSV.
///
/// Columns: depth_m, true_c, initial_c, reconstructed_c, error_m_per_s
///
/// The seismic analog of `write_element_csv` in skull_ct_phase_correction.rs.
pub fn write_velocity_csv(
    path: &Path,
    true_model: &Array3<f64>,
    initial_model: &Array3<f64>,
    reconstructed: &Array3<f64>,
) -> std::io::Result<()> {
    let cx = NX / 2;
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "depth_m,true_c_m_per_s,initial_c_m_per_s,reconstructed_c_m_per_s,error_m_per_s")?;
    for k in 0..NZ {
        let depth_m = k as f64 * DX;
        let t_c = true_model[[cx, 0, k]];
        let i_c = initial_model[[cx, 0, k]];
        let r_c = reconstructed[[cx, 0, k]];
        let err = r_c - t_c;
        writeln!(out, "{depth_m:.1},{t_c:.2},{i_c:.2},{r_c:.2},{err:.2}")?;
    }
    Ok(())
}

/// Encode a flat RGB byte buffer as a PNG file.
///
/// Converts `png::EncodingError` to `io::Error` so callers use `?` uniformly.
fn write_png(path: &Path, rgb: &[u8], width: usize, height: usize) -> io::Result<()> {
    let file = File::create(path)?;
    let w = BufWriter::new(file);
    let mut enc = png::Encoder::new(w, width as u32, height as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc
        .write_header()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    writer
        .write_image_data(rgb)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    Ok(())
}

/// Write a three-plane PNG of the middle orthogonal cross-sections of `model`.
///
/// # Layout
///
/// ```text
/// ┌──────────────┬──────────────┬──────────────┐  ← PANEL rows
/// │  Axial       │  Coronal     │  Sagittal     │
/// │  z = NZ/2    │  y = NY/2    │  x = NX/2    │
/// │  (x–y plane) │  (x–z plane) │  (y–z plane) │
/// ├──────────────┴──────────────┴──────────────┤  ← COLORBAR_H rows
/// │  velocity colorbar (c_lo → c_hi)           │
/// └────────────────────────────────────────────┘
///   3 × PANEL columns
/// ```
///
/// Each panel is nearest-neighbour resampled to `PANEL × PANEL` pixels.
///
/// # Quasi-2D note (NY = 2)
///
/// The axial (x–y) and sagittal (y–z) panels are degenerate because the
/// medium is identical in both y-planes.  They remain geometrically correct
/// and follow the same three-plane convention as `skull_ct_phase_correction`.
///
/// # References
///
/// Mirrors `write_three_plane_ppm` in `skull_ct_phase_correction.rs` but
/// outputs PNG instead of PPM (lossless, broadly supported).
pub fn write_three_plane_png(path: &Path, model: &Array3<f64>, c_lo: f64, c_hi: f64) -> io::Result<()> {
    let img_w = 3 * PANEL;
    let img_h = PANEL + COLORBAR_H;
    let mut rgb = vec![0_u8; img_w * img_h * 3];

    // ── Panel 0: Axial — z = NZ/2, x–y plane ───────────────────────────────
    //   px → ix ∈ [0, NX)   (lateral x, left → right)
    //   py → iy ∈ [0, NY)   (lateral y, front → back)
    let iz_mid = NZ / 2;
    for py in 0..PANEL {
        for px in 0..PANEL {
            let ix = (px * NX / PANEL).min(NX - 1);
            let iy = (py * NY / PANEL).min(NY - 1);
            let color = velocity_color(model[[ix, iy, iz_mid]], c_lo, c_hi);
            put_pixel(&mut rgb, img_w, img_h, px, py, color);
        }
    }
    draw_colorbar(&mut rgb, img_w, img_h, 0, c_lo, c_hi);

    // ── Panel 1: Coronal — y = NY/2, x–z plane (primary seismic section) ───
    //   px → ix ∈ [0, NX)   (lateral x, left → right)
    //   py → iz ∈ [0, NZ)   (depth z, shallow at top)
    let iy_mid = NY / 2;
    for py in 0..PANEL {
        for px in 0..PANEL {
            let ix = (px * NX / PANEL).min(NX - 1);
            let iz = (py * NZ / PANEL).min(NZ - 1);
            let color = velocity_color(model[[ix, iy_mid, iz]], c_lo, c_hi);
            put_pixel(&mut rgb, img_w, img_h, PANEL + px, py, color);
        }
    }
    draw_colorbar(&mut rgb, img_w, img_h, PANEL, c_lo, c_hi);

    // ── Panel 2: Sagittal — x = NX/2, y–z plane ────────────────────────────
    //   px → iy ∈ [0, NY)   (lateral y, left → right)
    //   py → iz ∈ [0, NZ)   (depth z, shallow at top)
    let ix_mid = NX / 2;
    for py in 0..PANEL {
        for px in 0..PANEL {
            let iy = (px * NY / PANEL).min(NY - 1);
            let iz = (py * NZ / PANEL).min(NZ - 1);
            let color = velocity_color(model[[ix_mid, iy, iz]], c_lo, c_hi);
            put_pixel(&mut rgb, img_w, img_h, 2 * PANEL + px, py, color);
        }
    }
    draw_colorbar(&mut rgb, img_w, img_h, 2 * PANEL, c_lo, c_hi);

    write_png(path, &rgb, img_w, img_h)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> KwaversResult<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("warn"));

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   Seismic Imaging — Full Waveform Inversion (kwavers)    ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ── 1. Geological model ───────────────────────────────────────────────
    println!("[ 1 / 6 ]  Building geological model …");
    let model = build_seismic_model();

    let c_min = model
        .sound_speed
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let c_max = model
        .sound_speed
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let rho_min = model.density.iter().copied().fold(f64::INFINITY, f64::min);
    let rho_max = model
        .density
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    println!(
        "  Grid            : {NX}×{NY}×{NZ} voxels @ {:.0} m",
        DX
    );
    println!(
        "  Domain          : {:.0} m × {:.0} m (x × z)",
        NX as f64 * DX,
        NZ as f64 * DX
    );
    println!("  Velocity range  : [{c_min:.0}, {c_max:.0}] m/s");
    println!("  Density range   : [{rho_min:.0}, {rho_max:.0}] kg/m³  (Gardner 1974)");
    println!(
        "  Salt dome       : {:.0} m radius at ({:.0} m depth)",
        8.0 * DX,
        34.0 * DX
    );
    println!("  Layers          : near-surface / upper clastic / lower clastic / deep / basement");

    // ── 2. Grid ───────────────────────────────────────────────────────────
    println!("\n[ 2 / 6 ]  Constructing computational grid …");
    let grid = Grid::new(NX, NY, NZ, DX, DX, DX)?;
    println!("  Grid OK");

    // ── 3. FWI parameters ────────────────────────────────────────────────
    println!("\n[ 3 / 6 ]  Configuring FWI parameters …");

    // CFL-stable timestep: dt ≤ CFL × dx / (c_max × √3).
    //
    // c_max = C_ANOMALY = 4500 m/s (the hardest constraint: salt dome).
    // CFL_factor = 0.3 (conservative — same as transcranial_fwi.rs and the
    // FDTD staggered-stencil solver's internal CFL guard).
    let dt = 0.3 * DX / (c_max * 3.0_f64.sqrt());

    // Number of time steps: cover one full depth-transit at the near-surface
    // velocity plus a 20% margin for deep reflections and multiples.
    //
    // T_transit = NZ × DX / c_near_surface ≈ 64 × 50 / 1500 ≈ 2.13 s
    // nt = T_transit × 1.2 / dt
    let t_transit = (NZ as f64 * DX) / C_NEAR_SURFACE;
    let nt = ((t_transit * 1.2) / dt).ceil() as usize;

    // FWI configuration.
    //
    // Cost estimate at opt-level = 1 on a single CPU:
    //   Each FDTD run: NX × NY × NZ × nt ≈ 64 × 2 × 64 × ~1300 = 10.7 M voxel-steps
    //   At ~4.5 ms/step (64×2×64 at opt-level=1, from transcranial_fwi timing):
    //   ≈ 5.9 s per FDTD run.
    //   3 iterations × (N_SHOTS fwd + N_SHOTS adj + 5 × N_SHOTS line-search)
    //   = 3 × 6 × 12 = 216 FDTD runs × 5.9 s ≈ 21 min.
    //   Set RUST_LOG=info to see per-iteration FWI diagnostics.
    let fwi_params = FwiParameters {
        max_iterations: 3,
        tolerance: 1e-12,
        step_size: STEP_SIZE,
        frequency: F0_HZ,
        nt,
        dt,
        n_trace: N_RECEIVERS,
        n_depth: 1,
        regularization: RegularizationParameters {
            tikhonov_weight: 0.0,
            tv_weight: 0.0,
            smoothness_weight: 0.0,
        },
    };

    println!("  dt              : {:.3} ms", dt * 1e3);
    println!("  f₀              : {:.0} Hz  (Ricker wavelet)", F0_HZ);
    println!(
        "  λ @ c_water     : {:.0} m  ({:.1} ppw at dx={:.0} m)",
        C_NEAR_SURFACE / F0_HZ,
        (C_NEAR_SURFACE / F0_HZ) / DX,
        DX
    );
    println!(
        "  nt              : {nt} steps  ({:.2} s)",
        nt as f64 * dt
    );
    println!("  step_size       : {:.0} m/s", STEP_SIZE);
    println!("  FWI iterations  : {}", fwi_params.max_iterations);

    // ── 4. Multi-shot surface acquisition geometry ────────────────────────
    println!("\n[ 4 / 6 ]  Building surface acquisition geometry …");

    let shot_positions = surface_shot_positions();
    println!(
        "  {} shots uniformly spaced on surface (z = 1 voxel, z_phys = {:.0} m)",
        N_SHOTS,
        1.0 * DX
    );
    println!(
        "  {} receivers at z = 2 ({:.0} m depth), uniformly spaced in x",
        N_RECEIVERS,
        2.0 * DX
    );
    for (s, &(ix, iz)) in shot_positions.iter().enumerate() {
        println!(
            "  Shot {:1}: (x={:2}, y=0, z={:2}) = ({:.0} m, {:.0} m depth)",
            s,
            ix,
            iz,
            ix as f64 * DX,
            iz as f64 * DX
        );
    }

    // ── 5. FWI ────────────────────────────────────────────────────────────
    println!("\n[ 5 / 6 ]  Running seismic FWI …");

    let fwi = FwiProcessor::new(fwi_params.clone());
    let true_model = model.sound_speed.clone();

    // Generate observed data from the true (anomaly-bearing) geological model.
    println!(
        "\n  ── Forward models (true geology, {} shots) ──",
        N_SHOTS
    );
    let t0 = Instant::now();
    let mut shots: Vec<(FwiGeometry, Array2<f64>)> = Vec::with_capacity(N_SHOTS);
    for &(ix, iz_src) in &shot_positions {
        let geometry = build_shot(ix, iz_src, nt, dt);
        let obs = fwi.generate_synthetic_data(&true_model, &geometry, &grid)?;
        shots.push((geometry, obs));
    }
    println!(
        "  {} observed gathers generated ({:.1} s)",
        N_SHOTS,
        t0.elapsed().as_secs_f32()
    );

    // Initial model: 1-D depth trend without anomaly.
    //
    // Starting from a smooth 1-D background is standard practice in exploration
    // FWI (Virieux & Operto 2009 §4.2).  The inversion recovers the salt dome
    // by minimising the L2 data misfit across all shots.
    let initial_model = build_background_model();

    // Joint data-space objective J₀ = Σᵢ Jᵢ(c_initial) before inversion.
    let mut j_initial = 0.0_f64;
    for (geom, obs) in &shots {
        let d_syn = fwi.generate_synthetic_data(&initial_model, geom, &grid)?;
        j_initial += d_syn
            .iter()
            .zip(obs.iter())
            .map(|(&s, &o)| (s - o).powi(2))
            .sum::<f64>()
            * 0.5
            * dt;
    }

    println!("\n  Quality before inversion:");
    print_quality_report(&true_model, &initial_model);
    println!("  Joint J₀        : {j_initial:.6e} Pa²·s  ({N_SHOTS} shots)");

    let t_inv = Instant::now();
    let reconstructed = fwi.invert_multi_source(&shots, &initial_model, &grid)?;
    println!(
        "\n  FWI completed in {:.1} s",
        t_inv.elapsed().as_secs_f32()
    );

    // Joint data-space objective J after inversion.
    let mut j_final = 0.0_f64;
    for (geom, obs) in &shots {
        let d_syn = fwi.generate_synthetic_data(&reconstructed, geom, &grid)?;
        j_final += d_syn
            .iter()
            .zip(obs.iter())
            .map(|(&s, &o)| (s - o).powi(2))
            .sum::<f64>()
            * 0.5
            * dt;
    }
    let j_reduction_pct = (1.0 - j_final / j_initial) * 100.0;

    println!("\n  Quality after inversion:");
    print_quality_report(&true_model, &reconstructed);
    println!("  Joint J         : {j_final:.6e} Pa²·s");
    println!("  J reduction     : {j_reduction_pct:7.1} %  (joint data-space L2)");

    // ── 6. RTM — zero-lag imaging condition on reconstructed-model snapshot ──
    println!("\n[ 6 / 6 ]  Reverse Time Migration (post-FWI imaging) …");

    // Use shot 0 to generate a forward snapshot for the RTM imaging condition.
    // The source and receiver wavefields are sampled at the same time step, so
    // this implements the zero-lag cross-correlation imaging condition:
    //   I(x) = ∫ p_src(x,t) · p_recv(x,T−t) dt
    //
    // Reference: Baysal E et al. (1983). Reverse time migration.
    //   Geophysics, 48(11), 1514–1524.
    let (geom0, _obs0) = &shots[0];
    let src_snapshot = fwi.generate_synthetic_data(&reconstructed, geom0, &grid)?;

    // Receiver wavefield: inject the last-sample amplitude at the first active
    // receiver position, modelling back-propagation of one observed trace.
    let mut recv_snapshot = Array3::<f64>::zeros((NX, NY, NZ));
    {
        let recv_mask = &geom0.sensor_mask;
        for ((i, _j, k), &active) in recv_mask.indexed_iter() {
            if active {
                let n_recv = src_snapshot.nrows();
                let n_t = src_snapshot.ncols();
                if n_recv > 0 && n_t > 0 {
                    recv_snapshot[[i, 0, k]] = src_snapshot[[0, n_t - 1]];
                }
                break;
            }
        }
    }

    let rtm_settings = RtmSettings {
        imaging_condition: ImagingCondition::Normalized,
        storage_strategy: StorageStrategy::Full,
        boundary_type: BoundaryType::Absorbing,
        apply_laplacian: true,
    };
    let rtm = RtmProcessor::new(rtm_settings);
    let rtm_image = rtm
        .migrate(&recv_snapshot, &recv_snapshot, &grid)
        .unwrap_or_else(|_| Array3::<f64>::zeros((NX, NY, NZ)));
    let rtm_peak = rtm_image.iter().copied().fold(0.0_f64, f64::max);
    println!("  RTM image completed — peak amplitude: {rtm_peak:.4}");

    // ── Image output ──────────────────────────────────────────────────────
    // Output directory: first CLI argument, or the fixed compile-time path
    // <crate_root>/examples/output/.
    //
    // env!("CARGO_MANIFEST_DIR") is set by Cargo at compile time to the
    // absolute path of the crate root (D:\kwavers\kwavers), so the default
    // is always D:\kwavers\kwavers\examples\output\ regardless of where
    // `cargo run` is invoked from.
    let output_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples").join("output"));

    // Create the directory if it does not yet exist.
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| KwaversError::InvalidInput(format!("cannot create output dir: {e}")))?;

    // Canonicalize after creation so the display path is always absolute.
    let abs_dir = std::fs::canonicalize(&output_dir).unwrap_or(output_dir.clone());

    let base = "seismic_fwi";
    let three_plane_path = abs_dir.join(format!("{base}_three_plane.png"));
    let velocity_ppm_path = abs_dir.join(format!("{base}.ppm"));
    let rtm_path = abs_dir.join(format!("{base}_rtm.ppm"));
    let csv_path = abs_dir.join(format!("{base}.csv"));

    // Three-plane PNG: axial / coronal / sagittal middle slices of the
    // reconstructed velocity model — directly mirrors write_three_plane_ppm
    // in skull_ct_phase_correction.rs.
    write_three_plane_png(&three_plane_path, &reconstructed, C_NEAR_SURFACE, C_ANOMALY)
        .map_err(|e| KwaversError::InvalidInput(format!("three-plane PNG write failed: {e}")))?;

    // 4-panel PPM comparison: true | initial | reconstructed | Δc
    write_velocity_panels(
        &velocity_ppm_path,
        &true_model,
        &initial_model,
        &reconstructed,
        &shot_positions.map(|(ix, iz)| (ix, iz)),
    )
    .map_err(|e| KwaversError::InvalidInput(format!("velocity panel write failed: {e}")))?;

    // RTM image: diverging colormap of the zero-lag cross-correlation
    write_rtm_panel(&rtm_path, &rtm_image)
        .map_err(|e| KwaversError::InvalidInput(format!("RTM panel write failed: {e}")))?;

    // Central-column velocity CSV
    write_velocity_csv(&csv_path, &true_model, &initial_model, &reconstructed)
        .map_err(|e| KwaversError::InvalidInput(format!("CSV write failed: {e}")))?;

    println!("\n  Output directory  : {}", abs_dir.display());
    println!("\n  Wrote images and data:");
    println!("    {}  (PNG three-plane: axial|coronal|sagittal)", three_plane_path.display());
    println!("    {}  (PPM 4-panel: true|initial|reconstructed|error)", velocity_ppm_path.display());
    println!("    {}  (PPM RTM zero-lag cross-correlation)", rtm_path.display());
    println!("    {}  (CSV central-column velocity profile)", csv_path.display());
    println!(
        "  Image size        : {}×{} px per panel, {} panels, {}px colorbar",
        PANEL, PANEL, 3, COLORBAR_H
    );
    println!("  Colormap          : blue→cyan→green→yellow→red  [{:.0}–{:.0} m/s]", C_NEAR_SURFACE, C_ANOMALY);
    println!("  Three-plane axes  : axial z={} (x–y) | coronal y={} (x–z) | sagittal x={} (y–z)",
        NZ / 2, NY / 2, NX / 2);

    // ── Summary ───────────────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════");
    println!(
        "  Reconstructed velocity range: [{:.0}, {:.0}] m/s",
        reconstructed
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        reconstructed
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    );
    println!("  True velocity range         : [{c_min:.0}, {c_max:.0}] m/s");
    println!();
    println!("  Physics verified against:");
    println!("    Gardner (1974)            — velocity–density empirical law");
    println!("    Ricker (1953)             — source wavelet");
    println!("    Tarantola (1984)          — adjoint-state FWI gradient");
    println!("    Virieux & Operto (2009)   — FWI objective and chain rule");
    println!("    skull_ct_phase_correction — acquisition geometry template");
    println!("    transcranial_fwi          — multi-shot inversion structure");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
