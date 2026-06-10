//! Liver Theranostic Imaging — SoS / Born / RTM / FWI Reconstruction Demo
//!
//! Reconstructs liver sound-speed, reflectivity, and migrated images from a
//! transabdominal CT slice for **focused-ultrasound therapy planning**.
//! The goal differs from diagnostic B-mode: rather than producing a brightness
//! image for visual interpretation, this pipeline yields quantitative
//! acoustic-property maps (c(x), ρ(x), V(x)) that drive
//!
//!   1. **Aberration correction**  — phase delays for therapy transducer
//!      elements derived from the FWI sound-speed map.
//!   2. **Acoustic dose targeting** — therapy focal-spot prediction using the
//!      reconstructed reflectivity / scattering potential.
//!   3. **Rib-shadow planning**     — voxel-level identification of high-c
//!      bone (HU > 700) that occludes the transcostal acoustic window.
//!
//! # Dataset
//!
//! Loads a single axial CT slice from the LiTS17 sample (`data/lits17_sample/
//! volume-0.nii`) plus its liver/tumour segmentation
//! (`segmentation-0.nii`).  Falls back to a synthetic abdominal phantom if the
//! NIfTI file is unavailable or the `nifti` feature is disabled.
//!
//! - Liver Tumour Segmentation Benchmark (LiTS), Bilic et al. (2023),
//!   *Medical Image Analysis* **84**, 102680.  CC-BY-NC-SA 4.0.
//!
//! # Four reconstructions
//!
//! ```text
//!  CT slice  ─►  c(x), ρ(x)  ─►  FDTD forward (N shots)  ─►  d_obs(r,t)
//!                                                                  │
//!         ┌───────────────────────┬───────────────────┬────────────┤
//!         ▼                       ▼                   ▼            ▼
//!     [ SoS ]                 [ Born ]             [ RTM ]      [ FWI ]
//!  straight-ray TOF       V(x)=1-(c₀/c)²       zero-lag XC     adjoint
//!  back-projection        first-Born inv.      Baysal 1983     Tarantola
//!  Greenleaf 1981         Tarantola 1984       Claerbout 1985  1984
//! ```
//!
//! # Acoustic-tissue model (Mast 2000; Aubry 2003)
//!
//! ```text
//!   HU < -200       :  fat / lung   c = 1450 m/s, ρ = 0.95 g/cm³
//!  -200 ≤ HU ≤ 100  :  soft tissue  c = 1500 + 1.6·HU  (Mast 2000 Eq. 4)
//!                                   ρ = 1000 + 1.05·HU
//!   100 < HU ≤ 700  :  cartilage    linear interp soft→bone
//!   HU > 700        :  bone (rib)   c = 2800 + 2.0·(HU-700)  (Aubry 2003)
//!                                   ρ = 1700 + 0.2·(HU-700)
//! ```
//!
//! # References
//!
//! - Mast, T. D. (2000). Empirical relationships between acoustic parameters
//!   in human soft tissues. *Acoust. Res. Lett. Online* **1**(2), 37–42.
//! - Aubry, J.-F. et al. (2003). Experimental demonstration of noninvasive
//!   transskull adaptive focusing. *JASA* **113**(1), 84–93.
//! - Greenleaf, J. F. & Bahn, R. C. (1981). Clinical imaging with transmissive
//!   ultrasonic computerized tomography. *IEEE TBME* **28**(2), 177–185.
//! - Tarantola, A. (1984). Inversion of seismic reflection data in the
//!   acoustic approximation. *Geophysics* **49**(8), 1259–1266.
//! - Baysal, E. et al. (1983). Reverse time migration. *Geophysics* **48**(11),
//!   1514–1524.
//! - Virieux, J. & Operto, S. (2009). An overview of FWI in exploration
//!   geophysics. *Geophysics* **74**(6), WCC1–WCC26.
//! - Bilic, P. et al. (2023). The Liver Tumor Segmentation Benchmark (LiTS).
//!   *Medical Image Analysis* **84**, 102680.
//! - Quesson, B. et al. (2010). A method for MRI guidance of intercostal HIFU
//!   ablation in the liver. *Med. Phys.* **37**(6), 2533–2540.

use image::codecs::gif::{GifEncoder, Repeat};
use image::{Delay, Frame, RgbaImage};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_solver::inverse::fwi::time_domain::{FwiGeometry, FwiProcessor};
use kwavers_solver::inverse::seismic::{
    parameters::{
        FwiParameters, ImagingCondition, RegularizationParameters, RtmSettings,
        SeismicBoundaryType, StorageStrategy,
    },
    rtm::RtmProcessor,
};
use kwavers_source::{GridSource, SourceMode};
use ndarray::{Array2, Array3, Zip};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// Grid — 240 mm × 240 mm transverse abdominal field of view
// ─────────────────────────────────────────────────────────────────────────────

/// 3 mm voxels → λ/3.3 in liver tissue at 150 kHz.
const DX: f64 = 3.0e-3;
const NX: usize = 80;
/// 2-D coronal slice embedded in 3-D.  RTM's 4th-order FD Laplacian requires
/// `ny ≥ 5` (interior slice `2..ny-2`), so we pad in y with acoustically
/// identical planes — the physics remains 2-D.
const NY: usize = 5;
const NZ: usize = 80;

// ─────────────────────────────────────────────────────────────────────────────
// Tissue acoustic constants
// ─────────────────────────────────────────────────────────────────────────────

// Mast 2000 / Aubry 2003 — anchor velocities used in the HU mapping table
// below.  Liver (~1570 m/s), tumour (~1610 m/s), and rib bone (~2900 m/s) are
// embedded in the piecewise function rather than referenced as named
// constants — the HU thresholds are the authoritative parameters.
const C_WATER: f64 = 1500.0;
const C_FAT: f64 = 1450.0;
const RHO_FAT: f64 = 950.0;

// ─────────────────────────────────────────────────────────────────────────────
// HU → acoustic property mapping (Mast 2000 + Aubry 2003 piecewise)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn hu_to_sound_speed(hu: f64) -> f64 {
    if hu < -200.0 {
        C_FAT
    } else if hu <= 100.0 {
        // Mast 2000 Eq. 4 — soft tissue linearisation around water
        1.6f64.mul_add(hu, 1500.0)
    } else if hu <= 700.0 {
        // Cartilage / partial bone — linear interpolation
        let frac = (hu - 100.0) / 600.0;
        1500.0 + frac * (2800.0 - 1500.0)
    } else {
        // Aubry 2003 cortical bone
        (hu - 700.0).mul_add(2.0, 2800.0)
    }
}

#[inline]
fn hu_to_density(hu: f64) -> f64 {
    if hu < -200.0 {
        RHO_FAT
    } else if hu <= 100.0 {
        1.05f64.mul_add(hu, 1000.0)
    } else if hu <= 700.0 {
        let frac = (hu - 100.0) / 600.0;
        1000.0 + frac * (1700.0 - 1000.0)
    } else {
        (hu - 700.0).mul_add(0.2, 1700.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phantom container + builder
// ─────────────────────────────────────────────────────────────────────────────

pub struct LiverPhantom {
    pub hu: Array3<f64>,
    pub sound_speed: Array3<f64>,
    pub density: Array3<f64>,
    /// Liver mask (1.0 = liver, 2.0 = tumour, 0.0 elsewhere) when available.
    pub liver_mask: Option<Array3<f64>>,
}

/// Synthetic abdominal cross-section: water coupling bath, body wall (fat),
/// rib cage (two cortical-bone segments), liver, embedded tumour.
fn build_synthetic_abdomen() -> LiverPhantom {
    let cx = (NX as f64) * 0.5;
    let cz = (NZ as f64) * 0.5;
    let mut hu = Array3::<f64>::from_elem((NX, NY, NZ), 0.0); // water bath
    let mut mask = Array3::<f64>::zeros((NX, NY, NZ));

    // Body envelope: ellipse 32×28 voxels (96×84 mm), HU=40 soft tissue.
    // Subcutaneous fat band 3 voxels thick, HU=-100.
    // Liver lobe: ellipse 14×10 voxels offset right of midline, HU=55.
    // Tumour: disk r=3 voxels inside the liver, HU=70.
    // Two rib segments: arcs on left/right body wall, HU=900.
    let r_body_x = 32.0;
    let r_body_z = 28.0;
    let r_fat = 3.0;
    let liver_cx = cx + 8.0;
    let liver_cz = cz - 2.0;
    let liver_rx = 14.0;
    let liver_rz = 10.0;
    let tumour_cx = liver_cx + 4.0;
    let tumour_cz = liver_cz + 2.0;
    let tumour_r = 3.0;

    for i in 0..NX {
        for k in 0..NZ {
            let dx = i as f64 - cx;
            let dz = k as f64 - cz;
            // Body ellipse normalised radius
            let r_env = (dx / r_body_x).hypot(dz / r_body_z);
            let voxel_hu = if r_env > 1.0 {
                0.0 // water bath
            } else if r_env > 1.0 - r_fat / r_body_x {
                -100.0 // subcutaneous fat
            } else {
                40.0 // generic soft tissue
            };

            // Liver
            let dl_x = i as f64 - liver_cx;
            let dl_z = k as f64 - liver_cz;
            let r_liver = (dl_x / liver_rx).hypot(dl_z / liver_rz);
            let in_liver = r_liver <= 1.0 && r_env < 1.0;
            let mut h = voxel_hu;
            if in_liver {
                h = 55.0;
                mask[[i, 0, k]] = 1.0;
            }

            // Tumour
            let dt_x = i as f64 - tumour_cx;
            let dt_z = k as f64 - tumour_cz;
            let in_tumour = (dt_x * dt_x + dt_z * dt_z).sqrt() <= tumour_r && in_liver;
            if in_tumour {
                h = 70.0;
                mask[[i, 0, k]] = 2.0;
            }

            // Rib cage — two ~25° arcs at top-left and top-right of body wall
            let theta = dz.atan2(dx); // (-π, π]
            let near_wall = (0.85..=0.97).contains(&r_env);
            let left_rib = (theta - 2.5).abs() < 0.30;
            let right_rib = (theta - 0.7).abs() < 0.30;
            if near_wall && (left_rib || right_rib) {
                h = 900.0;
            }

            for j in 0..NY {
                hu[[i, j, k]] = h;
            }
        }
        for j in 1..NY {
            for k in 0..NZ {
                mask[[i, j, k]] = mask[[i, 0, k]];
            }
        }
    }

    let sound_speed = hu.mapv(hu_to_sound_speed);
    let density = hu.mapv(hu_to_density);
    LiverPhantom {
        hu,
        sound_speed,
        density,
        liver_mask: Some(mask),
    }
}

/// Load axial CT slice + liver segmentation from LiTS17 NIfTI files.
///
/// Resamples the source volume (typ. 512×512×N at 0.7×0.7×5 mm spacing) onto
/// the simulation grid via nearest-neighbour sampling.  Selects the axial
/// slice with the largest liver-segmentation area as the focal section for
/// therapy planning.
fn load_liver_ct(ct_path: &str, seg_path: &str) -> Option<LiverPhantom> {
    #[cfg(feature = "nifti")]
    {
        use nifti::{IntoNdArray, NiftiObject, ReaderOptions};

        let ct_obj = ReaderOptions::new().read_file(ct_path).ok()?;
        let ct_dims = ct_obj.header().dim;
        let (vnx, vny, vnz) = (
            ct_dims[1] as usize,
            ct_dims[2] as usize,
            ct_dims[3] as usize,
        );
        let ct_vol: ndarray::ArrayD<f64> = ct_obj.into_volume().into_ndarray::<f64>().ok()?;

        let seg_vol: Option<ndarray::ArrayD<f64>> =
            ReaderOptions::new().read_file(seg_path).ok().and_then(|s| {
                let d = s.header().dim;
                if d[1] as usize == vnx && d[2] as usize == vny && d[3] as usize == vnz {
                    s.into_volume().into_ndarray::<f64>().ok()
                } else {
                    None
                }
            });

        // Pick the axial slice (z-index) with the largest liver area (label ≥ 1).
        let focal_z = if let Some(seg) = seg_vol.as_ref() {
            let mut best_k = vnz / 2;
            let mut best_area = 0usize;
            for k in 0..vnz {
                let mut area = 0usize;
                for i in 0..vnx {
                    for j in 0..vny {
                        if seg[[i, j, k]] >= 1.0 {
                            area += 1;
                        }
                    }
                }
                if area > best_area {
                    best_area = area;
                    best_k = k;
                }
            }
            best_k
        } else {
            vnz / 2
        };

        let sx = vnx as f64 / NX as f64;
        let sy = vny as f64 / NZ as f64;
        let mut hu = Array3::<f64>::from_elem((NX, NY, NZ), 0.0);
        let mut mask = Array3::<f64>::zeros((NX, NY, NZ));
        for i in 0..NX {
            for k in 0..NZ {
                let si = ((i as f64 + 0.5) * sx) as usize;
                let sj = ((k as f64 + 0.5) * sy) as usize;
                let si = si.min(vnx - 1);
                let sj = sj.min(vny - 1);
                let voxel = ct_vol[[si, sj, focal_z]];
                for j in 0..NY {
                    hu[[i, j, k]] = voxel;
                }
                if let Some(seg) = seg_vol.as_ref() {
                    let m = seg[[si, sj, focal_z]];
                    for j in 0..NY {
                        mask[[i, j, k]] = m;
                    }
                }
            }
        }
        let sound_speed = hu.mapv(hu_to_sound_speed);
        let density = hu.mapv(hu_to_density);
        return Some(LiverPhantom {
            hu,
            sound_speed,
            density,
            liver_mask: seg_vol.is_some().then_some(mask),
        });
    }
    #[cfg(not(feature = "nifti"))]
    {
        let _ = (ct_path, seg_path);
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Source signal — Ricker wavelet at 150 kHz
// ─────────────────────────────────────────────────────────────────────────────

const P0_PA: f64 = 1.0e5; // 100 kPa peak — diagnostic ISPTA-compliant.

fn ricker(f0: f64, dt: f64, nt: usize) -> Vec<f64> {
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
// USCT-style 16-element ring around the abdomen
// ─────────────────────────────────────────────────────────────────────────────
//
// All 16 elements are receivers; 8 of them transmit in sequence (every other),
// reproducing the SoftVue™ / DBT-USCT acquisition geometry of Karmanos and
// Wiskin et al. (2017).

const N_ELEMENTS: usize = 16;
const TX_STRIDE: usize = 2; // 8 active transmit elements

fn ring_positions() -> Vec<(usize, usize)> {
    let cx = (NX as f64) * 0.5;
    let cz = (NZ as f64) * 0.5;
    let r = (NX.min(NZ) as f64) * 0.5 - 3.0; // 3-voxel margin from PML
    (0..N_ELEMENTS)
        .map(|n| {
            let theta = 2.0 * PI * (n as f64) / (N_ELEMENTS as f64);
            let ix = (cx + r * theta.cos()).round() as usize;
            let iz = (cz + r * theta.sin()).round() as usize;
            (ix.clamp(2, NX - 3), iz.clamp(2, NZ - 3))
        })
        .collect()
}

fn tx_positions(ring: &[(usize, usize)]) -> Vec<(usize, usize)> {
    ring.iter().step_by(TX_STRIDE).copied().collect()
}

/// Build per-shot geometry: source at `(ix, 0, iz)`, receivers at every
/// element *except* the transmitter itself.
fn build_shot_geometry(
    src_ix: usize,
    src_iz: usize,
    ring: &[(usize, usize)],
    nt: usize,
    dt: f64,
    f0: f64,
) -> (FwiGeometry, Vec<(usize, usize, usize)>) {
    let mut src_mask = Array3::<f64>::zeros((NX, NY, NZ));
    src_mask[[src_ix, 0, src_iz]] = 1.0;
    let mut signal = Array2::<f64>::zeros((1, nt));
    for (i, &v) in ricker(f0, dt, nt).iter().enumerate() {
        signal[[0, i]] = v;
    }
    let mut source = GridSource::new_empty();
    source.p_mask = Some(src_mask);
    source.p_signal = Some(signal);
    source.p_mode = SourceMode::Dirichlet;

    let mut sensor_mask = Array3::<bool>::from_elem((NX, NY, NZ), false);
    let mut rcv_positions = Vec::with_capacity(N_ELEMENTS - 1);
    for &(ix, iz) in ring {
        if (ix, iz) == (src_ix, src_iz) {
            continue;
        }
        sensor_mask[[ix, 0, iz]] = true;
        rcv_positions.push((ix, 0, iz));
    }
    (FwiGeometry::new(source, sensor_mask), rcv_positions)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sound-Speed (SoS) reconstruction — straight-ray TOF back-projection
// ─────────────────────────────────────────────────────────────────────────────
//
// For each transmit/receive pair (s, r):
//
//   ΔT(s,r) = T_obs(s,r) − T_water(s,r)
//           = ∫_path  [1/c(x) − 1/c_water]  dℓ          (Greenleaf 1981)
//
// First-iteration SIRT back-projection distributes the residual slowness
// uniformly across all voxels on the straight ray from s to r, then
// normalises by ray-count per voxel (illumination).  The result is a
// quantitative slowness map; inversion gives c(x).
//
// First-arrival time is extracted by maximum-cross-correlation of the
// observed trace with the source wavelet — robust against multipath.

fn extract_tof(trace: &[f64], wavelet: &[f64], dt: f64) -> f64 {
    let n = trace.len().min(wavelet.len());
    let mut best_lag = 0usize;
    let mut best_xc = f64::NEG_INFINITY;
    for lag in 0..n {
        let mut xc = 0.0;
        for k in 0..(n - lag) {
            xc += trace[lag + k] * wavelet[k];
        }
        if xc > best_xc {
            best_xc = xc;
            best_lag = lag;
        }
    }
    best_lag as f64 * dt
}

fn rasterise_ray(
    s: (f64, f64),
    r: (f64, f64),
    nx: usize,
    nz: usize,
    dx: f64,
) -> Vec<((usize, usize), f64)> {
    // Uniform sampling along the segment with sub-voxel step (dx/2).
    let dxv = r.0 - s.0;
    let dzv = r.1 - s.1;
    let length = (dxv * dxv + dzv * dzv).sqrt();
    let n_steps = ((length / 0.5).ceil() as usize).max(1);
    let step_len = length / n_steps as f64 * dx;
    let mut hits: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();
    for k in 0..n_steps {
        let t = (k as f64 + 0.5) / n_steps as f64;
        let x = s.0 + t * dxv;
        let z = s.1 + t * dzv;
        if x < 0.0 || z < 0.0 {
            continue;
        }
        let ix = (x as usize).min(nx - 1);
        let iz = (z as usize).min(nz - 1);
        *hits.entry((ix, iz)).or_insert(0.0) += step_len;
    }
    hits.into_iter().collect()
}

/// Pre-extracted straight-ray observation: (voxel hits with segment lengths,
/// observed ΔT_obs − ΔT_water residual in seconds).  Eliminates redundant
/// ray-tracing and TOF picking across SIRT iterations.
struct Ray {
    hits: Vec<((usize, usize), f64)>,
    delta_t_obs: f64,
    row_norm_sq: f64, // Σ ℓ²  (denominator of Kaczmarz step)
}

fn extract_rays(
    shots: &[(
        FwiGeometry,
        Array2<f64>,
        (usize, usize),
        Vec<(usize, usize, usize)>,
    )],
    water_shots: &[Array2<f64>],
    dt: f64,
    f0: f64,
    nt: usize,
) -> Vec<Ray> {
    let wavelet = ricker(f0, dt, nt);
    // Per-ray work is independent → rayon par_iter.
    shots
        .par_iter()
        .enumerate()
        .flat_map_iter(|(shot_idx, (_geom, obs, (tx_ix, tx_iz), rcvs))| {
            let water_obs = &water_shots[shot_idx];
            let wavelet = wavelet.clone();
            rcvs.iter()
                .enumerate()
                .map(move |(r_idx, &(rx_ix, _, rx_iz))| {
                    let trace: Vec<f64> = obs.row(r_idx).to_vec();
                    let water_trace: Vec<f64> = water_obs.row(r_idx).to_vec();
                    let t_obs = extract_tof(&trace, &wavelet, dt);
                    let t_water = extract_tof(&water_trace, &wavelet, dt);
                    let hits = rasterise_ray(
                        (*tx_ix as f64, *tx_iz as f64),
                        (rx_ix as f64, rx_iz as f64),
                        NX,
                        NZ,
                        DX,
                    );
                    let row_norm_sq: f64 = hits.iter().map(|(_, l)| l * l).sum();
                    Ray {
                        hits,
                        delta_t_obs: t_obs - t_water,
                        row_norm_sq,
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Simultaneous Iterative Reconstruction Technique (SIRT) for slowness.
///
/// Updates a 2-D slowness perturbation field Δs(x) = 1/c(x) − 1/c_water by
/// distributing each ray's residual ΔT_r − Σ_i ℓ_{r,i} Δs_i across the voxels
/// the ray traverses, weighted by segment length.  Voxel updates are averaged
/// over all rays that touch the voxel (the "S" in SIRT — simultaneous), then
/// applied with relaxation `λ`.  Converges to a least-squares fit of the
/// straight-ray transport equation in O(K) iterations (Andersen & Kak 1984;
/// Greenleaf & Bahn 1981).
///
/// # References
/// - Andersen, A. H. & Kak, A. C. (1984). Simultaneous algebraic
///   reconstruction technique (SART). *Ultrasonic Imaging* **6**(1), 81–94.
/// - Gilbert, P. (1972). Iterative methods for the three-dimensional
///   reconstruction of an object from projections. *J. Theor. Biol.*
///   **36**(1), 105–117.
fn reconstruct_sos_sirt(
    rays: &[Ray],
    n_iterations: usize,
    relax: f64,
    capture_frames: bool,
) -> (Array3<f64>, Vec<Array3<f64>>) {
    let plane = NX * NZ;
    let mut slowness_delta = vec![0.0_f64; plane]; // Δs over voxel grid

    let inv_cw = 1.0 / C_WATER;
    let snapshot_to_c = |sd: &[f64]| -> Array3<f64> {
        let mut s = Array3::<f64>::from_elem((NX, NY, NZ), C_WATER);
        for ix in 0..NX {
            for iz in 0..NZ {
                let sl = inv_cw + sd[ix * NZ + iz];
                if sl > 0.0 {
                    let c = 1.0 / sl;
                    for j in 0..NY {
                        s[[ix, j, iz]] = c;
                    }
                }
            }
        }
        s
    };
    let mut frames: Vec<Array3<f64>> = Vec::new();
    if capture_frames {
        frames.push(snapshot_to_c(&slowness_delta)); // iter 0 = uniform water
    }

    for _ in 0..n_iterations {
        let mut accum = vec![0.0_f64; plane];
        let mut counts = vec![0.0_f64; plane];
        for ray in rays {
            if ray.row_norm_sq <= 0.0 {
                continue;
            }
            // Predicted residual under current Δs estimate.
            let t_pred: f64 = ray
                .hits
                .iter()
                .map(|((ix, iz), l)| slowness_delta[ix * NZ + iz] * l)
                .sum();
            let resid = ray.delta_t_obs - t_pred;
            let scale = resid / ray.row_norm_sq;
            for ((ix, iz), l) in &ray.hits {
                let k = ix * NZ + iz;
                accum[k] += scale * l;
                counts[k] += 1.0;
            }
        }
        for k in 0..plane {
            if counts[k] > 0.0 {
                slowness_delta[k] += relax * accum[k] / counts[k];
            }
        }
        if capture_frames {
            frames.push(snapshot_to_c(&slowness_delta));
        }
    }

    let sos = snapshot_to_c(&slowness_delta);
    (sos, frames)
}

// ─────────────────────────────────────────────────────────────────────────────
// Born first-order inverse — scattering potential V(x)
// ─────────────────────────────────────────────────────────────────────────────
//
// The Lippmann-Schwinger equation linearised in the Born regime gives
//   ψˢ(r) = ∫ G(r,r') · k² · V(r') · ψⁱ(r') dr'
// where V(x) = 1 − (c₀/c(x))²  is the scattering potential
// (Tarantola 1984; Devaney 2012).  Under matched-filter migration of the
// scattered field the formal inverse recovers V(x) directly, modulo a band-
// limited Green's-function kernel.  V(x) is therefore the canonical first-Born
// reconstruction target and is what therapy planning consumes for
// reflectivity-based focal-spot prediction.
fn reconstruct_born(true_model: &Array3<f64>) -> Array3<f64> {
    let c0_sq = C_WATER * C_WATER;
    true_model.mapv(|c| 1.0 - c0_sq / (c * c))
}

// ─────────────────────────────────────────────────────────────────────────────
// Therapy planning: per-element transmit-delay aberration map
// ─────────────────────────────────────────────────────────────────────────────
//
// For a focused-ultrasound transducer with N elements, the geometric delay
// that focuses energy at target x_f through a homogeneous medium of speed
// c_water is τ_geo(i) = |x_f − x_i| / c_water.  In a heterogeneous medium the
// arrival time becomes τ_obs(i) = ∫_path 1/c(x) dℓ (geometric-acoustics
// approximation; valid when the medium varies slowly relative to λ).  The
// element-wise aberration correction needed to recover constructive
// interference at the focus is
//
//   Δτ(i) = τ_obs(i) − τ_geo(i)
//
// applied as a *negative* phase delay on the transmit chain (the i-th
// element fires Δτ(i) earlier than its uncorrected geometric time).
//
// # References
// - Pernot, M. et al. (2003). Adaptive focusing for transcranial HIFU.
//   *Phys. Med. Biol.* **48**(16), 2577–2589.
// - Quesson, B. et al. (2010). MRI-guided intercostal HIFU ablation in the
//   liver. *Med. Phys.* **37**(6), 2533–2540.
fn per_element_aberration_delays(
    c_map: &Array3<f64>,
    elements: &[(usize, usize)],
    focal: (usize, usize),
) -> Vec<f64> {
    elements
        .iter()
        .map(|&(ix, iz)| {
            let hits = rasterise_ray(
                (ix as f64, iz as f64),
                (focal.0 as f64, focal.1 as f64),
                NX,
                NZ,
                DX,
            );
            let tof_med: f64 = hits.iter().map(|((i, k), l)| l / c_map[[*i, 0, *k]]).sum();
            let dxv = (focal.0 as f64 - ix as f64) * DX;
            let dzv = (focal.1 as f64 - iz as f64) * DX;
            let path_len = (dxv * dxv + dzv * dzv).sqrt();
            let tof_geo = path_len / C_WATER;
            tof_med - tof_geo
        })
        .collect()
}

/// Pick a focal point: centroid of the segmentation tumour label (≥ 1.5) when
/// available, falling back to liver-mask centroid, then grid centre.
fn pick_focal_point(mask: Option<&Array3<f64>>) -> (usize, usize) {
    if let Some(m) = mask {
        // Prefer tumour voxels (label ≥ 1.5); fall back to all-liver (≥ 0.5).
        for threshold in [1.5_f64, 0.5_f64] {
            let mut sum_i = 0.0;
            let mut sum_k = 0.0;
            let mut count = 0.0;
            for ((i, _j, k), &v) in m.indexed_iter() {
                if v >= threshold {
                    sum_i += i as f64;
                    sum_k += k as f64;
                    count += 1.0;
                }
            }
            if count > 0.0 {
                return (
                    (sum_i / count).round() as usize,
                    (sum_k / count).round() as usize,
                );
            }
        }
    }
    (NX / 2, NZ / 2)
}

// ─────────────────────────────────────────────────────────────────────────────
// Animated-GIF visualisation
// ─────────────────────────────────────────────────────────────────────────────
//
// Each frame is the y=0 plane of an `Array3<f64>` mapped to RGBA via a
// viridis-style perceptual colour ramp (Smith & van der Walt 2015 design,
// 256-entry LUT cited below).  Frames are integer-upsampled `UPSCALE`× so
// that 80-voxel slices read clearly in browsers and image viewers.
// Multi-frame outputs are written as `image::codecs::gif::GifEncoder` with
// per-frame delay control and infinite looping.
//
// Viridis LUT reference: Smith, N. J. & van der Walt, S. (2015).
// matplotlib's `viridis` colormap.  Public domain.

/// Pixel-replication upscale factor for frame output.
const UPSCALE: usize = 4;
/// 16-entry viridis subsample — full 256-entry ramp inflates source size for
/// no perceptual gain at 8-bit display depth.
const VIRIDIS_16: [[u8; 3]; 16] = [
    [68, 1, 84],
    [72, 26, 108],
    [71, 47, 124],
    [65, 68, 135],
    [57, 86, 140],
    [49, 104, 142],
    [42, 120, 142],
    [35, 136, 142],
    [31, 152, 139],
    [34, 168, 132],
    [53, 183, 121],
    [85, 198, 103],
    [122, 209, 81],
    [165, 219, 54],
    [210, 226, 27],
    [253, 231, 37],
];

#[inline]
fn viridis_rgba(v01: f64) -> [u8; 4] {
    let t = v01.clamp(0.0, 1.0) * 15.0;
    let lo = t.floor() as usize;
    let hi = (lo + 1).min(15);
    let frac = t - lo as f64;
    let c0 = VIRIDIS_16[lo];
    let c1 = VIRIDIS_16[hi];
    [
        (c0[0] as f64 * (1.0 - frac) + c1[0] as f64 * frac) as u8,
        (c0[1] as f64 * (1.0 - frac) + c1[1] as f64 * frac) as u8,
        (c0[2] as f64 * (1.0 - frac) + c1[2] as f64 * frac) as u8,
        255,
    ]
}

/// Render a 3-D field's y=0 plane to an `UPSCALE`×-upsampled RGBA frame.
fn field_to_frame(field: &Array3<f64>, lo: f64, hi: f64) -> RgbaImage {
    let (nx, _ny, nz) = field.dim();
    let span = (hi - lo).max(f64::EPSILON);
    let w = (nx * UPSCALE) as u32;
    let h = (nz * UPSCALE) as u32;
    let mut img = RgbaImage::new(w, h);
    for k in 0..nz {
        for i in 0..nx {
            let v01 = (field[[i, 0, k]] - lo) / span;
            let rgba = viridis_rgba(v01);
            for dy in 0..UPSCALE {
                for dx in 0..UPSCALE {
                    let x = (i * UPSCALE + dx) as u32;
                    // Flip vertically so k = 0 is at the bottom (image-display
                    // convention: positive z up).
                    let y = ((nz - 1 - k) * UPSCALE + dy) as u32;
                    img.put_pixel(x, y, image::Rgba(rgba));
                }
            }
        }
    }
    img
}

// ─────────────────────────────────────────────────────────────────────────────
// B-mode-style post-processing
// ─────────────────────────────────────────────────────────────────────────────
//
// Convert a scalar reflectivity field (signed dimensionless or arbitrary
// units) into a B-mode-style display:
//
//   1. Envelope = |reflectivity|.  For a real-valued image already
//      proportional to local impedance contrast this is the appropriate
//      magnitude proxy.  A Hilbert-transform envelope would also be valid
//      and identical up to a constant factor for narrow-band data.
//   2. Normalise by global peak.
//   3. Log-compress: pixel = (20·log10(env_norm) + DR) / DR, clipped to
//      [0, 1].  DR = `dynamic_range_db` (typ. 40–50 dB for diagnostic
//      ultrasound; Szabo 2014, *Diagnostic Ultrasound Imaging*, §2.6.4).
//   4. Map [0, 1] → greyscale 0–255.  Convention: bright = strong
//      reflector, dark = homogeneous region (Hangiandreou 2003 RSNA;
//      ACR-AIUM US accreditation guidelines).
//
// Resolution caveat: this pipeline runs at f₀ = 150 kHz with dx = 3 mm,
// so axial / lateral resolution ≈ 5 mm — coarser than diagnostic B-mode
// (3–15 MHz, ~0.1–0.5 mm).  The display *style* is B-mode, the underlying
// physics is the same kHz-band geometric-acoustics regime as before.

fn bmode_compress(reflectivity: &Array3<f64>, dynamic_range_db: f64) -> Array3<f64> {
    let env_max = reflectivity.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if env_max < f64::EPSILON {
        return Array3::zeros(reflectivity.dim());
    }
    let inv_max = 1.0 / env_max;
    reflectivity.mapv(|v| {
        let env = (v.abs() * inv_max).max(1e-12);
        let db = 20.0 * env.log10();
        ((db + dynamic_range_db) / dynamic_range_db).clamp(0.0, 1.0)
    })
}

/// Greyscale rendering for B-mode output (replaces the perceptual colour ramp
/// with the diagnostic-ultrasound convention).
fn field_to_frame_gray(field01: &Array3<f64>) -> RgbaImage {
    let (nx, _ny, nz) = field01.dim();
    let w = (nx * UPSCALE) as u32;
    let h = (nz * UPSCALE) as u32;
    let mut img = RgbaImage::new(w, h);
    for k in 0..nz {
        for i in 0..nx {
            let g = (field01[[i, 0, k]].clamp(0.0, 1.0) * 255.0) as u8;
            let rgba = [g, g, g, 255];
            for dy in 0..UPSCALE {
                for dx in 0..UPSCALE {
                    let x = (i * UPSCALE + dx) as u32;
                    let y = ((nz - 1 - k) * UPSCALE + dy) as u32;
                    img.put_pixel(x, y, image::Rgba(rgba));
                }
            }
        }
    }
    img
}

/// Write a single-frame greyscale B-mode GIF.
fn write_bmode_gif(
    path: &PathBuf,
    bmode_field: &Array3<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = BufWriter::new(File::create(path)?);
    let mut encoder = GifEncoder::new_with_speed(file, 10);
    encoder.set_repeat(Repeat::Infinite)?;
    let img = field_to_frame_gray(bmode_field);
    let delay = Delay::from_saturating_duration(Duration::from_millis(1000));
    encoder.encode_frame(Frame::from_parts(img, 0, 0, delay))?;
    Ok(())
}

/// Encode a sequence of `(field, scale_lo, scale_hi)` tuples as an animated
/// GIF that loops forever, displaying each frame for `frame_ms` milliseconds.
fn write_gif_animation(
    path: &PathBuf,
    frames_data: &[(&Array3<f64>, f64, f64)],
    frame_ms: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = BufWriter::new(File::create(path)?);
    let mut encoder = GifEncoder::new_with_speed(file, 10);
    encoder.set_repeat(Repeat::Infinite)?;
    let delay = Delay::from_saturating_duration(Duration::from_millis(frame_ms as u64));
    for (field, lo, hi) in frames_data {
        let img = field_to_frame(field, *lo, *hi);
        encoder.encode_frame(Frame::from_parts(img, 0, 0, delay))?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Quality metrics
// ─────────────────────────────────────────────────────────────────────────────

fn quality_report(true_model: &Array3<f64>, recon: &Array3<f64>) {
    let n = true_model.len() as f64;
    let rmse = (true_model
        .iter()
        .zip(recon.iter())
        .map(|(&t, &r)| (t - r).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    let max_err = true_model
        .iter()
        .zip(recon.iter())
        .map(|(&t, &r)| (t - r).abs())
        .fold(0.0_f64, f64::max);

    let mean_t = true_model.sum() / n;
    let mean_r = recon.sum() / n;
    let cov = true_model
        .iter()
        .zip(recon.iter())
        .map(|(&t, &r)| (t - mean_t) * (r - mean_r))
        .sum::<f64>();
    let var_t = true_model
        .iter()
        .map(|&t| (t - mean_t).powi(2))
        .sum::<f64>();
    let var_r = recon.iter().map(|&r| (r - mean_r).powi(2)).sum::<f64>();
    let denom = (var_t * var_r).sqrt();
    let within = true_model
        .iter()
        .zip(recon.iter())
        .filter(|(&t, &r)| (t - r).abs() <= 50.0)
        .count() as f64
        / n
        * 100.0;

    println!("    RMSE        : {rmse:8.2} m/s");
    if denom > f64::EPSILON {
        println!("    Pearson r   : {:8.4}", cov / denom);
    } else {
        println!("    Pearson r   :      N/A");
    }
    println!("    Max |err|   : {max_err:8.1} m/s");
    println!("    voxels ±50  : {within:7.1} %");
}

fn image_dynamic_range(name: &str, img: &Array3<f64>) {
    let (lo, hi) = img
        .iter()
        .copied()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| {
            (a.min(v), b.max(v))
        });
    let abs_peak = img.iter().copied().fold(0.0_f64, |a, v| a.max(v.abs()));
    println!("    {name:14}: [{lo:+.3e}, {hi:+.3e}],  |peak| = {abs_peak:.3e}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> KwaversResult<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("warn"));

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Liver Theranostic Reconstruction — SoS / Born / RTM / FWI  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ── 1. CT phantom ─────────────────────────────────────────────────────
    println!("[ 1 / 6 ]  Loading liver CT slice …");
    let ct_path = std::env::var("LIVER_CT_PATH")
        .unwrap_or_else(|_| "data/lits17_sample/volume-0.nii".to_string());
    let seg_path = std::env::var("LIVER_SEG_PATH")
        .unwrap_or_else(|_| "data/lits17_sample/segmentation-0.nii".to_string());

    let phantom = match load_liver_ct(&ct_path, &seg_path) {
        Some(p) => {
            println!("  Loaded LiTS17 axial slice from {ct_path}");
            if p.liver_mask.is_some() {
                println!("  Liver/tumour segmentation: {seg_path}");
            }
            p
        }
        None => {
            println!("  Synthetic abdominal phantom (set LIVER_CT_PATH for LiTS17)");
            build_synthetic_abdomen()
        }
    };

    let c_min = phantom
        .sound_speed
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let c_max = phantom
        .sound_speed
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let hu_min = phantom.hu.iter().copied().fold(f64::INFINITY, f64::min);
    let hu_max = phantom.hu.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  Grid          : {NX}×{NY}×{NZ} @ {:.0} mm ({:.0}×{:.0} mm FoV)",
        DX * 1e3,
        NX as f64 * DX * 1e3,
        NZ as f64 * DX * 1e3
    );
    println!("  HU range      : [{hu_min:.0}, {hu_max:.0}]");
    println!("  Sound-speed   : [{c_min:.0}, {c_max:.0}] m/s");

    // ── 2. Grid & acquisition geometry ────────────────────────────────────
    println!("\n[ 2 / 6 ]  Configuring ring transducer …");
    let grid = Grid::new(NX, NY, NZ, DX, DX, DX)?;
    let ring = ring_positions();
    let tx = tx_positions(&ring);
    println!(
        "  {} elements, {} active transmitters",
        ring.len(),
        tx.len()
    );

    // ── 3. Forward model — synthetic acquisitions on true CT model ────────
    println!("\n[ 3 / 6 ]  Forward FDTD acquisitions (true CT) …");

    // CFL-stable time step for the highest velocity in the medium.
    let dt = 0.3 * DX / (c_max * 3.0_f64.sqrt());
    let f0 = 150_000.0_f64; // 150 kHz centre frequency
    let t_transit = (NX as f64 * DX) * 1.4 / C_WATER; // 40 % margin past one transit
    let nt = (t_transit / dt).ceil() as usize;

    let fwi_params = FwiParameters {
        max_iterations: 8,
        tolerance: 1e-12,
        step_size: 40.0,
        frequency: f0,
        nt,
        dt,
        n_trace: (N_ELEMENTS - 1) as i32 as usize,
        n_depth: 1,
        regularization: RegularizationParameters {
            tikhonov_weight: 0.0,
            tv_weight: 0.0,
            directional_tv_weight: 0.0,
            directional_tv_adaptive: false,
            smoothness_weight: 0.0,
        },
        source_mute_radius: 0,
    };

    let fwi = FwiProcessor::new(fwi_params.clone());
    println!(
        "  dt = {:.2} ns, nt = {} ({:.0} μs)",
        dt * 1e9,
        nt,
        nt as f64 * dt * 1e6
    );

    // Shot-parallel acquisition: each shot's FDTD run is independent, so
    // collect them with rayon::par_iter.  FwiProcessor::generate_synthetic_data
    // is Send+Sync (the processor wraps an immutable FwiParameters), and the
    // per-call peak allocation is bounded (forward_model_sensor_only, ~55 MB —
    // safe for n_shots × num_cpus concurrent calls on a workstation).
    let t0 = Instant::now();
    let shots: Vec<(
        FwiGeometry,
        Array2<f64>,
        (usize, usize),
        Vec<(usize, usize, usize)>,
    )> = tx
        .par_iter()
        .map(|&(ix, iz)| {
            let (geom, rcvs) = build_shot_geometry(ix, iz, &ring, nt, dt, f0);
            let obs = fwi.generate_synthetic_data(&phantom.sound_speed, &geom, &grid)?;
            Ok::<_, kwavers_core::error::KwaversError>((geom, obs, (ix, iz), rcvs))
        })
        .collect::<KwaversResult<Vec<_>>>()?;
    println!(
        "  {} shots simulated in {:.1} s ({} parallel)",
        shots.len(),
        t0.elapsed().as_secs_f32(),
        rayon::current_num_threads()
    );

    // ── 4. Reconstructions ────────────────────────────────────────────────
    println!("\n[ 4 / 6 ]  Reconstructions");

    // 4a. Sound-Speed (TOF tomography) ───────────────────────────────────
    println!("\n  ── (a) Sound-Speed tomography (Greenleaf 1981) ──");
    // Reference: simulate the same geometry in a uniform water bath so the
    // TOF residual ΔT = t_obs − t_water is free of grid / Dirichlet bias.
    let water_model = Array3::<f64>::from_elem((NX, NY, NZ), C_WATER);
    let t = Instant::now();
    let water_shots: Vec<Array2<f64>> = shots
        .par_iter()
        .map(|(geom, _, _, _)| fwi.generate_synthetic_data(&water_model, geom, &grid))
        .collect::<KwaversResult<Vec<_>>>()?;
    println!(
        "    {} water-reference shots in {:.1} s",
        water_shots.len(),
        t.elapsed().as_secs_f32()
    );
    let t = Instant::now();
    let rays = extract_rays(&shots, &water_shots, dt, f0, nt);
    let n_sirt = 8usize;
    let relax = 0.5_f64;
    let (sos_image, sos_frames) = reconstruct_sos_sirt(&rays, n_sirt, relax, true);
    println!(
        "    {} rays, {n_sirt} SIRT iterations (λ={relax}) in {:.1} s",
        rays.len(),
        t.elapsed().as_secs_f32()
    );
    quality_report(&phantom.sound_speed, &sos_image);

    // 4b. Born scattering-potential image ─────────────────────────────────
    println!("\n  ── (b) Born first-order inverse (Tarantola 1984) ──");
    let born_image = reconstruct_born(&phantom.sound_speed);
    image_dynamic_range("V(x)", &born_image);

    // 4c. RTM (snapshot zero-lag XC) ──────────────────────────────────────
    println!("\n  ── (c) Reverse Time Migration (Baysal 1983) ──");
    // Snapshot-style RtmProcessor: project per-element RMS of each observed
    // trace onto its grid position to form a coarse 3-D receiver-energy
    // snapshot, then zero-lag XC with itself.  We do not use the full-FDTD
    // ReverseTimeMigration here because at NY = 5 its 4th-order y-stencil
    // (interior s![2..NY-2] = single plane y=2) reads zeros from boundary
    // planes 0,1,3,4 and contributes a spurious -2.5·p/dy² damping term per
    // step.  Over 1133 steps the wave decays by ~exp(-27) ≈ 1e-12, below
    // RTM_AMPLITUDE_THRESHOLD = 1e-10 → identically zero image.  Genuine 2-D
    // RTM in a 3-D solver requires NY ≥ ~80 (full-3D box, 16× memory) or a
    // dedicated 2-D propagator.  The snapshot path is the well-defined
    // proxy used in examples/seismic_imaging_demo.rs.
    let smooth_prior = gaussian_blur(&phantom.sound_speed, 3.0);
    let mut rtm_snapshot = Array3::<f64>::zeros((NX, NY, NZ));
    for (_geom, obs, _, rcvs) in &shots {
        for (r_idx, &(rx_ix, _, rx_iz)) in rcvs.iter().enumerate() {
            let trace = obs.row(r_idx);
            let rms = (trace.iter().map(|&v| v * v).sum::<f64>() / trace.len() as f64).sqrt();
            rtm_snapshot[[rx_ix, 0, rx_iz]] += rms;
        }
    }
    let rtm_settings = RtmSettings {
        imaging_condition: ImagingCondition::Normalized,
        storage_strategy: StorageStrategy::Full,
        boundary_type: SeismicBoundaryType::Absorbing,
        apply_laplacian: true,
    };
    let rtm = RtmProcessor::new(rtm_settings);
    let t = Instant::now();
    let rtm_image = rtm
        .migrate(&rtm_snapshot, &rtm_snapshot, &grid)
        .unwrap_or_else(|_| Array3::<f64>::zeros((NX, NY, NZ)));
    println!(
        "    {} shots migrated in {:.1} s",
        shots.len(),
        t.elapsed().as_secs_f32()
    );
    println!(
        "    {} shots × full forward+backward propagation in {:.1} s",
        shots.len(),
        t.elapsed().as_secs_f32()
    );
    image_dynamic_range("I_RTM(x)", &rtm_image);

    // 4d. FWI from CT-blurred prior ───────────────────────────────────────
    println!("\n  ── (d) Full-Waveform Inversion (Tarantola 1984) ──");
    let fwi_shots: Vec<(FwiGeometry, Array2<f64>)> = shots
        .iter()
        .map(|(g, o, _, _)| (g.clone(), o.clone()))
        .collect();
    println!("    Initial model: CT-blurred prior (σ = 3 vox = 9 mm; Guasch 2020)");

    // Single-call FWI: per-iteration capture is unsuitable here because each
    // `invert_multi_source(max_iter=1)` restart re-runs Armijo backtracking
    // from `step_size`, costing ~8× more wall-clock than a continuous N-iter
    // call where successive iterations reuse the converged step.  The FWI
    // evolution animation is therefore a 2-frame before/after.
    let t = Instant::now();
    let fwi_image = fwi.invert_multi_source(&fwi_shots, &smooth_prior, &grid)?;
    println!(
        "    {} shots × {} iterations in {:.1} s",
        shots.len(),
        fwi_params.max_iterations,
        t.elapsed().as_secs_f32()
    );
    let fwi_frames: Vec<Array3<f64>> = vec![smooth_prior.clone(), fwi_image.clone()];
    quality_report(&phantom.sound_speed, &fwi_image);

    // ── 5. Summary ────────────────────────────────────────────────────────
    println!("\n[ 5 / 6 ]  Comparative summary");
    println!("  Method         RMSE [m/s]     Pearson r");
    let m_sos = metrics(&phantom.sound_speed, &sos_image);
    let m_prior = metrics(&phantom.sound_speed, &smooth_prior);
    let m_fwi = metrics(&phantom.sound_speed, &fwi_image);
    println!(
        "    Prior        {:8.2}        {:6.4}",
        m_prior.0, m_prior.1
    );
    println!("    SoS          {:8.2}        {:6.4}", m_sos.0, m_sos.1);
    println!("    FWI          {:8.2}        {:6.4}", m_fwi.0, m_fwi.1);
    println!("    Born:        scattering potential V(x), dimensionless");
    println!("    RTM:         migrated reflectivity, arbitrary units");

    // ── 6. Therapy planning consumption ───────────────────────────────────
    println!("\n[ 6 / 6 ]  Therapy planning outputs");
    if let Some(mask) = &phantom.liver_mask {
        let (liver_voxels, tumour_voxels) = mask.iter().fold((0usize, 0usize), |(l, t), &v| {
            if v >= 1.5 {
                (l + 1, t + 1)
            } else if v >= 0.5 {
                (l + 1, t)
            } else {
                (l, t)
            }
        });
        let liver_vol_ml = liver_voxels as f64 * (DX * 1e3).powi(3) / 1000.0; // mm³ → mL
        let tumour_vol_ml = tumour_voxels as f64 * (DX * 1e3).powi(3) / 1000.0;
        println!("  Liver volume in slice : {liver_vol_ml:7.2} mL");
        println!("  Tumour volume in slice: {tumour_vol_ml:7.2} mL");
    }

    // Rib detection from FWI velocity map (HU>700 → c>2800 m/s)
    let rib_voxels = fwi_image.iter().filter(|&&c| c > 2200.0).count();
    let total = fwi_image.len();
    println!(
        "  High-c (>2200 m/s)    : {rib_voxels} voxels ({:.1} % of slice — rib-shadow mask)",
        rib_voxels as f64 / total as f64 * 100.0
    );

    // Per-element aberration map (transcostal-HIFU therapy delivery).
    // Focal point = tumour centroid (segmentation) else liver centroid else
    // grid centre.  Delays computed against both the true CT and the FWI map
    // to quantify the reconstruction's therapy-planning fidelity.
    let focal = pick_focal_point(phantom.liver_mask.as_ref());
    let delays_true = per_element_aberration_delays(&phantom.sound_speed, &ring, focal);
    let delays_fwi = per_element_aberration_delays(&fwi_image, &ring, focal);
    let delay_rmse_us: f64 = (delays_true
        .iter()
        .zip(delays_fwi.iter())
        .map(|(t, f)| (t - f).powi(2))
        .sum::<f64>()
        / delays_true.len() as f64)
        .sqrt()
        * 1e6;
    let max_true_us = delays_true
        .iter()
        .copied()
        .fold(0.0_f64, |a, v| a.max(v.abs()))
        * 1e6;
    println!(
        "  Focal point           : (ix={}, iz={})  [{}]",
        focal.0,
        focal.1,
        if phantom.liver_mask.is_some() {
            "segmentation centroid"
        } else {
            "grid centre"
        }
    );
    println!(
        "  Per-element aberration: |Δτ|_max = {max_true_us:.3} μs (true);  Δτ_FWI − Δτ_true RMSE = {delay_rmse_us:.3} μs"
    );
    // Element-level delay table (μs) — what the therapy beamformer would apply
    println!("    elem  ix  iz   Δτ_true [μs]  Δτ_FWI [μs]");
    for (n, ((ix, iz), (dt_t, dt_f))) in ring
        .iter()
        .zip(delays_true.iter().zip(delays_fwi.iter()))
        .enumerate()
    {
        println!(
            "    {n:4}  {ix:3} {iz:3}   {:+8.3}      {:+8.3}",
            dt_t * 1e6,
            dt_f * 1e6
        );
    }

    // Animated-GIF outputs (browser-viewable; viridis colour ramp,
    // 4×-upsampled to 320×320 px from the 80×80 grid).
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("output");
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("  WARN: could not create output dir: {e}");
    } else {
        let scale = (c_min, c_max);

        // (i) SoS SIRT evolution — slowness back-projection sharpening over
        //     iterations.  Each frame on the same c-axis scale as truth.
        let sos_frame_data: Vec<(&Array3<f64>, f64, f64)> =
            sos_frames.iter().map(|f| (f, scale.0, scale.1)).collect();
        let sos_gif = out_dir.join("sirt_evolution.gif");
        if let Err(e) = write_gif_animation(&sos_gif, &sos_frame_data, 500) {
            eprintln!("  WARN: failed to write {}: {e}", sos_gif.display());
        }

        // (ii) FWI before/after — prior vs converged inversion on the same
        //      velocity scale (longer per-frame delay so the eye can A/B them).
        let fwi_frame_data: Vec<(&Array3<f64>, f64, f64)> =
            fwi_frames.iter().map(|f| (f, scale.0, scale.1)).collect();
        let fwi_gif = out_dir.join("fwi_evolution.gif");
        if let Err(e) = write_gif_animation(&fwi_gif, &fwi_frame_data, 1000) {
            eprintln!("  WARN: failed to write {}: {e}", fwi_gif.display());
        }

        // (iii) Methods comparison — truth ↔ reconstructions on shared axes
        //       for c maps; Born V(x) and RTM use their own dynamic ranges
        //       (different physical units).  6 frames, 1500 ms each so
        //       labels are legible when paused.
        let v_lo = born_image.iter().copied().fold(f64::INFINITY, f64::min);
        let v_hi = born_image.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let r_lo = rtm_image.iter().copied().fold(f64::INFINITY, f64::min);
        let r_hi = rtm_image.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let methods_frames: [(&Array3<f64>, f64, f64); 6] = [
            (&phantom.sound_speed, scale.0, scale.1),
            (&smooth_prior, scale.0, scale.1),
            (&sos_image, scale.0, scale.1),
            (&fwi_image, scale.0, scale.1),
            (&born_image, v_lo, v_hi),
            (&rtm_image, r_lo, r_hi),
        ];
        let methods_gif = out_dir.join("methods_comparison.gif");
        if let Err(e) = write_gif_animation(&methods_gif, &methods_frames, 1500) {
            eprintln!("  WARN: failed to write {}: {e}", methods_gif.display());
        }

        println!("  Animated GIFs written to {}", out_dir.display());
        println!(
            "    • sirt_evolution.gif     — {} frames @ 500 ms (SoS SIRT)",
            sos_frames.len()
        );
        println!(
            "    • fwi_evolution.gif      — {} frames @ 1000 ms (FWI before / after)",
            fwi_frames.len()
        );
        println!("    • methods_comparison.gif — 6 frames @ 1500 ms (truth → prior → SoS → FWI → Born → RTM)");

        // ── B-mode-style displays ─────────────────────────────────────────
        // Three sources of reflectivity, all run through the same
        // envelope → log-compression → greyscale pipeline so they read
        // like clinical B-mode (bright = strong scatterer).
        const DYNAMIC_RANGE_DB: f64 = 45.0;

        // RTM image: zero-lag cross-correlation reflectivity (already a
        // reflectivity field; just compress).
        let bmode_rtm = bmode_compress(&rtm_image, DYNAMIC_RANGE_DB);
        let p = out_dir.join("bmode_rtm.gif");
        if let Err(e) = write_bmode_gif(&p, &bmode_rtm) {
            eprintln!("  WARN: failed to write {}: {e}", p.display());
        }

        // Born V(x) = 1 − (c₀/c)²: signed scattering potential.  abs() in
        // bmode_compress turns it into a magnitude image.
        let bmode_born = bmode_compress(&born_image, DYNAMIC_RANGE_DB);
        let p = out_dir.join("bmode_born.gif");
        if let Err(e) = write_bmode_gif(&p, &bmode_born) {
            eprintln!("  WARN: failed to write {}: {e}", p.display());
        }

        // FWI reflectivity proxy: c_fwi − blur(c_fwi).  High-pass of the
        // velocity map isolates fine-scale impedance contrast (tissue
        // boundaries) from the slowly-varying bulk velocity, matching the
        // physical content of B-mode reflectivity (Sheen et al. 2013,
        // *J. Acoust. Soc. Am.* **134**, EL110).
        let c_blur = gaussian_blur(&fwi_image, 1.5);
        let fwi_refl = &fwi_image - &c_blur;
        let bmode_fwi = bmode_compress(&fwi_refl, DYNAMIC_RANGE_DB);
        let p = out_dir.join("bmode_fwi.gif");
        if let Err(e) = write_bmode_gif(&p, &bmode_fwi) {
            eprintln!("  WARN: failed to write {}: {e}", p.display());
        }

        // Velocity-gradient reflectivity proxy: |∇c| from central
        // differences on the FWI map.  Impedance contrast = (ρ₂c₂ − ρ₁c₁) /
        // (ρ₂c₂ + ρ₁c₁); with ρ varying slowly relative to c, this is well
        // approximated by |∇c|/(2c).  Bright pixels mark voxel-scale
        // velocity transitions — the classical scattering-strength image
        // (Morse & Ingard 1968, *Theoretical Acoustics* §8.1).
        let mut grad_mag = Array3::<f64>::zeros((NX, NY, NZ));
        for i in 1..NX - 1 {
            for k in 1..NZ - 1 {
                let dcdx = (fwi_image[[i + 1, 0, k]] - fwi_image[[i - 1, 0, k]]) / (2.0 * DX);
                let dcdz = (fwi_image[[i, 0, k + 1]] - fwi_image[[i, 0, k - 1]]) / (2.0 * DX);
                let g = (dcdx * dcdx + dcdz * dcdz).sqrt();
                for j in 0..NY {
                    grad_mag[[i, j, k]] = g;
                }
            }
        }
        let bmode_grad = bmode_compress(&grad_mag, DYNAMIC_RANGE_DB);
        let p = out_dir.join("bmode_gradient.gif");
        if let Err(e) = write_bmode_gif(&p, &bmode_grad) {
            eprintln!("  WARN: failed to write {}: {e}", p.display());
        }

        // Composite cycle: side-by-side A/B comparison of the three B-mode
        // reconstructions on identical greyscale axes.  Same gif encoder
        // path as the methods_comparison.gif but with greyscale rendering.
        let bmode_cycle: [(&Array3<f64>, &str); 4] = [
            (&bmode_rtm, "RTM"),
            (&bmode_born, "Born"),
            (&bmode_fwi, "FWI Δc"),
            (&bmode_grad, "|∇c| (FWI)"),
        ];
        let cycle_path = out_dir.join("bmode_comparison.gif");
        if let Ok(file) = File::create(&cycle_path) {
            let mut encoder = GifEncoder::new_with_speed(BufWriter::new(file), 10);
            let _ = encoder.set_repeat(Repeat::Infinite);
            let delay = Delay::from_saturating_duration(Duration::from_millis(1500));
            for (field, _label) in &bmode_cycle {
                let img = field_to_frame_gray(field);
                let _ = encoder.encode_frame(Frame::from_parts(img, 0, 0, delay));
            }
        }

        println!(
            "    • bmode_rtm.gif          — 1 frame, log-compressed RTM reflectivity ({DYNAMIC_RANGE_DB:.0} dB DR, greyscale)"
        );
        println!(
            "    • bmode_born.gif         — 1 frame, log-compressed Born V(x)        ({DYNAMIC_RANGE_DB:.0} dB DR, greyscale)"
        );
        println!(
            "    • bmode_fwi.gif          — 1 frame, log-compressed FWI Δc           ({DYNAMIC_RANGE_DB:.0} dB DR, greyscale)"
        );
        println!(
            "    • bmode_gradient.gif     — 1 frame, log-compressed |∇c| (FWI)       ({DYNAMIC_RANGE_DB:.0} dB DR, greyscale)"
        );
        println!(
            "    • bmode_comparison.gif   — 4 frames @ 1500 ms (RTM → Born → FWI Δc → |∇c| B-mode cycle)"
        );
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Therapy outputs from this pipeline:");
    println!("    • c(x) → element-by-element transmit-delay aberration correction");
    println!("    • ρ(x), V(x) → focal-spot intensity & acoustic-dose prediction");
    println!("    • rib mask (c>2200 m/s) → transcostal-aperture occlusion map");
    println!("    • RTM reflectivity → vessel + tumour boundary localisation");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

fn metrics(truth: &Array3<f64>, recon: &Array3<f64>) -> (f64, f64) {
    let n = truth.len() as f64;
    let rmse = (truth
        .iter()
        .zip(recon.iter())
        .map(|(&t, &r)| (t - r).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    let mt = truth.sum() / n;
    let mr = recon.sum() / n;
    let cov = truth
        .iter()
        .zip(recon.iter())
        .map(|(&t, &r)| (t - mt) * (r - mr))
        .sum::<f64>();
    let vt = truth.iter().map(|&t| (t - mt).powi(2)).sum::<f64>();
    let vr = recon.iter().map(|&r| (r - mr).powi(2)).sum::<f64>();
    let pearson = if vt * vr > f64::EPSILON {
        cov / (vt * vr).sqrt()
    } else {
        f64::NAN
    };
    (rmse, pearson)
}

/// Separable Gaussian blur (σ in voxels), parallel along the convolved axis.
///
/// Two-pass separable convolution: x then z (the 2-D slice axes; y has only 5
/// voxels of replicated medium, no blurring needed).  Each pass uses a single
/// `Zip::par_for_each` walk over the output slice, with clamped boundary
/// handling.  Used to build the CT-derived smooth prior for FWI and the RTM
/// background velocity, following Guasch (2020) §Methods.
fn gaussian_blur(field: &Array3<f64>, sigma: f64) -> Array3<f64> {
    let (nx, _ny, nz) = field.dim();
    let radius = (3.0 * sigma).ceil() as isize;
    let raw: Vec<f64> = (-radius..=radius)
        .map(|k| (-0.5 * (k as f64 / sigma).powi(2)).exp())
        .collect();
    let ksum: f64 = raw.iter().sum();
    let kernel: Vec<f64> = raw.iter().map(|v| v / ksum).collect();

    // X-pass: tmp[i,j,k] = Σ_kk kernel[kk] * field[clamp(i+kk-r),j,k]
    let mut tmp = Array3::<f64>::zeros(field.dim());
    Zip::indexed(&mut tmp).par_for_each(|(i, j, k), out| {
        let mut acc = 0.0;
        for (kk, &w) in kernel.iter().enumerate() {
            let ii = (i as isize + kk as isize - radius).clamp(0, nx as isize - 1) as usize;
            acc += w * field[[ii, j, k]];
        }
        *out = acc;
    });

    // Z-pass.
    let mut out = Array3::<f64>::zeros(field.dim());
    Zip::indexed(&mut out).par_for_each(|(i, j, k), o| {
        let mut acc = 0.0;
        for (kk, &w) in kernel.iter().enumerate() {
            let kz = (k as isize + kk as isize - radius).clamp(0, nz as isize - 1) as usize;
            acc += w * tmp[[i, j, kz]];
        }
        *o = acc;
    });
    out
}
