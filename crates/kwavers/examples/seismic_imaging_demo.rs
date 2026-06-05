//! Transcranial Ultrasound FWI — Brain Reconstruction Demo.
//!
//! # Physical pipeline
//!
//! ```text
//! Skull CT phantom  →  c(x), ρ(x)  →  FDTD forward  →  synthetic traces
//!                                                               │
//!                              ← adjoint source ←  L2 residual
//!                              │
//!                              FDTD adjoint (time-reversed, back-propagated)
//!                              │
//!                              gradient ∂J/∂c  →  model update  →  brain image
//! ```
//!
//! # Skull phantom
//!
//! Coronal cross-section (x–z plane) of a human head modelled as concentric
//! shells centred at (NX/2, NZ/2) = (32, 32):
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │               water coupling bath                   │
//! │         ┌─────────────────────────┐                 │
//! │         │   scalp  (HU ≈  40)    │                  │
//! │         │  ┌─────────────────┐   │                  │
//! │         │  │  outer cortical │   │                  │
//! │         │  │  bone (HU≈720) │   │  ← z (depth)     │
//! │         │  │  ┌───────────┐  │   │                  │
//! │         │  │  │  diploe   │  │   │                  │
//! │         │  │  │ (HU≈380) │  │   │                  │
//! │         │  │  │ ┌───────┐ │  │   │                  │
//! │         │  │  │ │ inner │ │  │   │                  │
//! │  SRC    │  │  │ │ cort. │ │  │  RECV               │
//! │  (left  │  │  │ │┌─────┐│ │  │  (right             │
//! │  arc)   │  │  │ ││brain││ │  │   arc)              │
//! │         │  │  │ │└─────┘│ │  │                     │
//! └─────────┴──┴──┴─┴───────┴─┴──┴─────────────────────┘
//!             ↑ x (lateral, left→right)
//! ```
//!
//! # Full-ring acquisition geometry
//!
//! The 2-D FWI acquisition uses sixteen active element locations uniformly
//! distributed around the full ring at R_ARRAY = 20 voxels from the grid
//! centre.  Eight of them transmit in sequence (every other element) while the
//! remaining fifteen act as receivers.  Full-ring coverage provides illumination
//! from all azimuths, eliminating the shadow zone that limits superior-hemisphere
//! geometries and improving convergence for inferior skull structures.
//!
//! # Initial model — CT-derived smooth prior
//!
//! Starting FWI from a homogeneous 1500 m/s background causes cycle-skipping:
//! at f₀ = 150 kHz, T = 6.7 μs, but skull transmission delay ≈ 5 μs > T/2.
//! The wave never converges from such a large initial model error.
//!
//! Instead, the initial model is a Gaussian-blurred version of the true skull
//! model (σ = 3 voxels ≈ 9 mm).  This is the standard clinical approach: a
//! low-resolution CT scan is always available and provides a smooth but
//! geometrically correct bone map.  Starting from this smooth map ensures travel
//! times are within λ/2 of the truth, so the FWI refines boundaries rather
//! than fighting cycle-skipping.
//!
//! Reference: Guasch (2020) — CT-based initial model for brain FWI, §Methods.
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
//! - Aubry, J.-F. et al. (2003). Experimental demonstration of noninvasive
//!   transskull adaptive focusing. *JASA*, 113(1), 84–93.
//! - Marsac, L. et al. (2017). Ex vivo optimisation of a heterogeneous speed of
//!   sound model of the human skull. *Int. J. Hyperthermia*, 33(6), 635–645.
//! - Guasch, L. et al. (2020). Full-waveform inversion imaging of the human
//!   brain. *npj Digital Medicine*, 3, 28.
//! - Tarantola, A. (1984). Inversion of seismic reflection data in the acoustic
//!   approximation. *Geophysics*, 49(8), 1259–1266.
//! - Virieux, J. & Operto, S. (2009). An overview of full-waveform inversion in
//!   exploration geophysics. *Geophysics*, 74(6), WCC1–WCC26.

use kwavers_core::error::{KwaversError, KwaversResult};
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
use std::f64::consts::PI;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

// CT loading imports (ritk required-feature)
use anyhow::Context as _;
use burn::backend::NdArray as NdArrayBackend;
use ritk_io::{
    load_dicom_series, read_nifti, read_png_series, scan_dicom_directory, DicomSeriesInfo,
};

// ─────────────────────────────────────────────────────────────────────────────
// Grid constants
// ─────────────────────────────────────────────────────────────────────────────

/// Grid spacing [m].  3 mm gives λ/3.3 resolution at 150 kHz in water.
///
/// Reference: Marsac 2017 — mean skull thickness ≈ 7 mm → at least 2 voxels
/// through bone at 3 mm spacing.
const DX: f64 = 3.0e-3;

/// Grid dimensions.  NY = 2 satisfies the FDTD staggered-stencil minimum while
/// keeping the second y-plane acoustically transparent (identical medium).
const NX: usize = 64; // lateral  192 mm
const NY: usize = 2; // quasi-2-D embedding
const NZ: usize = 64; // depth    192 mm

// ─────────────────────────────────────────────────────────────────────────────
// Skull phantom geometry — radii in voxels from centre (32, 32)
// ─────────────────────────────────────────────────────────────────────────────
//
// # CPML geometry constraint
//
// FDTD solver uses CPML thickness = 10 cells on all active boundaries.
// Physical domain: ix ∈ [10, 53], iz ∈ [10, 53] (44×44 voxels = 132 mm).
//
// The skull must fit entirely inside the physical domain:
//   max(R_HEAD + cx, cz) ≤ NX − CPML = 54  →  R_HEAD ≤ 22
//
// We use R_HEAD = 18 (54 mm radius) to provide a 6-voxel water bath margin
// between the outer scalp edge (ix = 32±18 = 14..50) and the CPML boundary
// (ix = 10 and ix = 54).
//
// Reference: Marsac 2017 — head radius ≈ 80 mm; skull thickness ≈ 7 mm.
// Scaled proportionally: 18/26 × original = 0.69 scaling factor.

const R_HEAD: f64 = 18.0; // 54 mm — outer scalp surface
const R_SKULL_OUT: f64 = 16.0; // 48 mm — outer cortical / scalp boundary
const R_DIPLOE: f64 = 14.0; // 42 mm — outer diploe boundary
const R_SKULL_IN: f64 = 12.0; // 36 mm — inner cortical / brain boundary
const R_BRAIN: f64 = 11.0; // 33 mm — brain surface (CSF buffer ≈ 3 mm)

// ─────────────────────────────────────────────────────────────────────────────
// Hounsfield unit → acoustic property conversion (Aubry 2003 BVF model)
// ─────────────────────────────────────────────────────────────────────────────

/// Typical HU values per skull layer (Aubry 2003 Table I; Marsac 2017 Table 1).
const HU_WATER: f64 = 0.0; // water coupling bath
const HU_SCALP: f64 = 40.0; // soft tissue / scalp / dura
const HU_CORTICAL_OUT: f64 = 720.0; // outer cortical bone
const HU_DIPLOE: f64 = 380.0; // trabecular / diploe
const HU_CORTICAL_IN: f64 = 660.0; // inner cortical bone
const HU_BRAIN: f64 = 35.0; // grey/white matter average

/// Water and dense cortical bone acoustic properties (Aubry 2003).
const C_WATER: f64 = 1500.0; // sound speed in water [m/s]
const C_CORTICAL: f64 = 2900.0; // sound speed in dense cortical bone [m/s]
const RHO_WATER: f64 = 1000.0; // density of water [kg/m³]
const RHO_CORTICAL: f64 = 1900.0; // density of cortical bone [kg/m³]

/// Repository-local Medimodel skull CT used by `skull_ct_phase_correction.rs`.
/// Absolute path derived at compile time from CARGO_MANIFEST_DIR so the binary
/// resolves the dataset regardless of the working directory it is invoked from.
const DEFAULT_MEDIMODEL_DICOM_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../data/medimodel_human_skull_2/dicom/DICOM"
);

/// Phase-correction example Medimodel series UID.  The directory contains
/// additional derived CT series with inconsistent orientation; this UID is the
/// same 67-slice skull CT series used by the companion example.
const DEFAULT_MEDIMODEL_SERIES_UID: &str =
    "1.3.6.1.4.1.5962.99.1.1761388472.1291962045.1616669124536.2634.0";

/// Full hemispherical aperture size used by the companion phase-correction demo.
const TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT: usize = 1024;

/// Colormap bounds [m/s] for skull image panels.
const C_LO: f64 = C_WATER; // 1500 m/s → blue
const C_HI: f64 = C_CORTICAL; // 2900 m/s → red

// ─────────────────────────────────────────────────────────────────────────────
// Stage-2 brain tissue FWI constants (Guasch 2020 §Methods)
// ─────────────────────────────────────────────────────────────────────────────

/// Brain tissue sound speeds from Duck (1990) "Physical Properties of Tissue".
const C_GRAY: f64 = 1541.0; // gray matter [m/s]
const C_WHITE: f64 = 1520.0; // white matter [m/s]
const C_CSF: f64 = 1505.0; // cerebrospinal fluid [m/s]

/// Velocity bounds for brain tissue FWI (excludes bone which is frozen).
const BRAIN_C_MIN: f64 = 1480.0; // m/s
const BRAIN_C_MAX: f64 = 1560.0; // m/s

/// Velocity threshold for classifying a voxel as bone (frozen in Stage 2).
/// Corresponds to BVF > 0.143, approximately HU ≈ 143 → cortical bone onset.
const BONE_VELOCITY_THRESHOLD: f64 = 1714.0; // m/s

/// Peak frequency for brain tissue FWI.
/// At 400 kHz: λ_brain ≈ 3.8 mm; tissue velocity errors < 3% → no cycle-skipping.
const F0_BRAIN_HZ: f64 = 400_000.0; // Hz

/// Number of FWI iterations for brain tissue Stage 2.
const N_BRAIN_ITER: usize = 20;

/// Step size for brain tissue FWI (smaller than skull FWI — brain Δc is tiny).
const STEP_SIZE_BRAIN: f64 = 30.0; // m/s per normalised gradient step

/// Repository-local MNI ICBM 2009c NIfTI directory (downloaded from BIC MNI).
/// Contains GM / WM / CSF tissue probability maps at 1 mm isotropic resolution.
/// Download: https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09c_nifti.zip
const DEFAULT_MNI_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../data/mni_icbm152_2009c/mni_icbm152_nlin_sym_09c"
);

/// MNI ICBM 2009c inner-skull radius at the coronal mid-plane [mm].
/// The inner cortical surface is ≈ 82 mm from the brain centroid in this atlas.
const MNI_INNER_SKULL_RADIUS_MM: f64 = 82.0;

/// Bone volume fraction from Hounsfield unit.
///
/// φ(HU) = clamp(HU / 1000, 0, 1)
///
/// Reference: Aubry 2003, JASA 113(1), eq. (2).
#[inline]
fn bvf(hu: f64) -> f64 {
    (hu / 1000.0).clamp(0.0, 1.0)
}

/// Sound speed from HU via linear Voigt BVF mixing.
///
/// c(HU) = c_water × (1 − φ) + c_cortical × φ   [m/s]
///
/// Reference: Aubry 2003; Marsac 2017 eq. (1).
#[inline]
fn hu_to_sound_speed(hu: f64) -> f64 {
    let phi = bvf(hu);
    C_WATER * (1.0 - phi) + C_CORTICAL * phi
}

/// Density from HU via linear BVF mixing.
///
/// ρ(HU) = ρ_water × (1 − φ) + ρ_cortical × φ   [kg/m³]
#[inline]
fn hu_to_density(hu: f64) -> f64 {
    let phi = bvf(hu);
    RHO_WATER * (1.0 - phi) + RHO_CORTICAL * phi
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage-2 brain tissue helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a per-voxel FWI frozen mask for Stage-2 brain tissue inversion.
///
/// Frozen voxels are restored from the CT reference after every gradient step.
/// Only high-velocity bone voxels (c > BONE_VELOCITY_THRESHOLD) are frozen;
/// scalp, diploe-transition, and water coupling remain free.  This keeps the
/// free region large enough for the FWI gradient to converge while still
/// preventing updates to the cortical bone wall.
///
/// Visualization alignment is handled separately in `write_brain_tissue_png`
/// using the geometric r < R_SKULL_IN criterion, independent of this mask.
fn build_skull_mask(sound_speed: &Array3<f64>) -> Array3<bool> {
    sound_speed.mapv(|c| c > BONE_VELOCITY_THRESHOLD)
}

/// Load MNI ICBM 2009c tissue probability maps and resample them onto the FWI
/// 2-D coronal grid, returning a brain tissue velocity model.
///
/// # Tissue velocity mapping (Duck 1990)
///
/// For each free (non-bone) FWI voxel the velocity is a probability-weighted
/// mixture of the three soft-tissue classes:
///
/// ```text
/// c(x) = p_gm(x) × C_GRAY + p_wm(x) × C_WHITE + p_csf(x) × C_CSF
///       + (1 − p_gm − p_wm − p_csf) × C_WATER
/// ```
///
/// Bone voxels are left at their CT-derived velocities (they will be frozen by
/// `build_skull_mask` during FWI and are never updated).
///
/// # Spatial mapping
///
/// The MNI ICBM 2009c atlas is sampled at the mid-coronal slice (y ≈ rows/2,
/// near the anterior commissure).  Each FWI voxel is mapped to MNI coordinates
/// by scaling: `mni_offset = fwi_offset_mm × (MNI_INNER_SKULL_RADIUS_MM / fwi_inner_skull_mm)`.
fn build_brain_velocity_model(
    skull_phantom: &SkullPhantom,
    mni_dir: &Path,
) -> anyhow::Result<Array3<f64>> {
    type Backend = NdArrayBackend<f32>;
    let device = Default::default();

    // Load the three probability maps via ritk NIfTI reader.
    // CtVolume.hu stores probability values [0,1] (HU clamping [-1024,3071] is harmless).
    let load = |name: &str| -> anyhow::Result<Array3<f64>> {
        let path = mni_dir.join(name);
        anyhow::ensure!(
            path.exists(),
            "MNI tissue map not found: '{}' — download from {}",
            path.display(),
            "https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09c_nifti.zip"
        );
        // Load through ritk (returns [Z,Y,X] tensor → transposed to hu[X,Y,Z]).
        let img = ritk_io::read_nifti::<Backend, _>(&path, &device)
            .with_context(|| format!("NIfTI load failed: '{}'", path.display()))?;
        let [depth, rows, cols] = img.shape();
        let td = img.data().clone().into_data();
        let vals = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("NIfTI data not f32: {e:?}"))?;
        let mut vol = Array3::<f64>::zeros((cols, rows, depth));
        for z in 0..depth {
            for y in 0..rows {
                for x in 0..cols {
                    vol[[x, y, z]] =
                        f64::from(vals[z * rows * cols + y * cols + x]).clamp(0.0, 1.0);
                }
            }
        }
        Ok(vol)
    };

    let gm = load("mni_icbm152_gm_tal_nlin_sym_09c.nii")?;
    let wm = load("mni_icbm152_wm_tal_nlin_sym_09c.nii")?;
    let csf = load("mni_icbm152_csf_tal_nlin_sym_09c.nii")?;

    let (mni_nx, mni_ny, mni_nz) = gm.dim();
    // MNI centroid voxel (brain centre-of-mass in MNI space ≈ [nx/2, ny/2, nz/2]).
    let cx_mni = mni_nx / 2; // ~90
    let cy_mni = mni_ny / 2; // ~108 — mid coronal slice (near AC)
    let cz_mni = mni_nz / 2; // ~90

    // Scale factor: project FWI physical offset (mm) into MNI voxel offset.
    // At the inner skull boundary: R_SKULL_IN × DX × 1e3 mm (FWI) ↔ MNI_INNER_SKULL_RADIUS_MM.
    let fwi_inner_mm = R_SKULL_IN * DX * 1e3; // 36 mm
    let fwi_to_mni = MNI_INNER_SKULL_RADIUS_MM / fwi_inner_mm; // 82/36 ≈ 2.28

    let cx_fwi = NX / 2; // 32
    let cz_fwi = NZ / 2; // 32

    let mut brain_model = skull_phantom.sound_speed.clone();

    for iz in 0..NZ {
        for ix in 0..NX {
            // Only assign MNI tissue velocities inside the inner skull surface
            // (r < R_SKULL_IN = 36 mm).  The skull wall, scalp, and water coupling
            // bath all retain their CT-derived velocities — they are frozen during
            // Stage-2 FWI and must not carry tissue-velocity artefacts.
            //
            // Using the velocity threshold c_ct > BONE_VELOCITY_THRESHOLD is
            // insufficient because bilinear interpolation at the 15 CT-px / FWI-voxel
            // scale blurs the thin skull wall into intermediate-HU voxels that fall
            // below the threshold, leaking MNI velocities into the scalp ring and
            // creating a false yellow band in the tight [1480,1560] m/s colormap.
            let dx_fwi = ix as f64 - cx_fwi as f64;
            let dz_fwi = iz as f64 - cz_fwi as f64;
            let r_fwi = (dx_fwi * dx_fwi + dz_fwi * dz_fwi).sqrt();
            if r_fwi >= R_SKULL_IN {
                continue; // skull wall, scalp, water bath — keep CT velocity
            }

            // FWI voxel physical offset from grid centre [mm].
            let dx_mm = dx_fwi * DX * 1e3;
            let dz_mm = dz_fwi * DX * 1e3;

            // Map to MNI voxel coordinate.
            let mni_x = (cx_mni as f64 + dx_mm * fwi_to_mni).round() as isize;
            let mni_z = (cz_mni as f64 + dz_mm * fwi_to_mni).round() as isize;

            // Out-of-bounds → keep water velocity.
            if mni_x < 0 || mni_x >= mni_nx as isize || mni_z < 0 || mni_z >= mni_nz as isize {
                continue;
            }
            let mx = mni_x as usize;
            let mz = mni_z as usize;

            // Sample mid-coronal MNI slice.
            let p_gm = gm[[mx, cy_mni, mz]];
            let p_wm = wm[[mx, cy_mni, mz]];
            let p_csf = csf[[mx, cy_mni, mz]];
            let p_rest = (1.0 - p_gm - p_wm - p_csf).clamp(0.0, 1.0);

            let c_tissue = p_gm * C_GRAY + p_wm * C_WHITE + p_csf * C_CSF + p_rest * C_WATER;

            for iy in 0..NY {
                brain_model[[ix, iy, iz]] = c_tissue;
            }
        }
    }

    Ok(brain_model)
}

// ─────────────────────────────────────────────────────────────────────────────
// Skull phantom structure
// ─────────────────────────────────────────────────────────────────────────────

/// 2-D coronal skull cross-section phantom.
///
/// Mirrors `SkullPhantom` in `transcranial_fwi.rs`.
pub struct SkullPhantom {
    /// Sound speed c(x) [m/s], shape (NX, NY, NZ).
    pub sound_speed: Array3<f64>,
    /// Density ρ(x) [kg/m³], shape (NX, NY, NZ).
    pub density: Array3<f64>,
    /// CT Hounsfield unit map, shape (NX, NY, NZ).
    pub hu: Array3<f64>,
}

/// Build the synthetic skull phantom.
///
/// Each voxel is assigned an HU value based on its distance from the head
/// centre (NX/2, NZ/2).  HU is then converted to c and ρ via Aubry (2003).
///
/// # Geometry (radii from centre in voxels)
///
/// | Region           | Radius range           | HU    |
/// |------------------|------------------------|-------|
/// | Water coupling   | r > R_HEAD             |     0 |
/// | Scalp            | R_SKULL_OUT < r ≤ R_HEAD |  40 |
/// | Outer cortical   | R_DIPLOE < r ≤ R_SKULL_OUT | 720 |
/// | Diploe           | R_SKULL_IN < r ≤ R_DIPLOE  | 380 |
/// | Inner cortical   | R_BRAIN < r ≤ R_SKULL_IN   | 660 |
/// | Brain / CSF      | r ≤ R_BRAIN            |    35 |
fn build_skull_phantom() -> SkullPhantom {
    let cx = (NX / 2) as f64; // 32.0
    let cz = (NZ / 2) as f64; // 32.0

    let mut hu = Array3::<f64>::from_elem((NX, NY, NZ), HU_WATER);

    for i in 0..NX {
        for k in 0..NZ {
            let dx = i as f64 - cx;
            let dz = k as f64 - cz;
            let r = (dx * dx + dz * dz).sqrt();

            let voxel_hu = if r > R_HEAD {
                HU_WATER
            } else if r > R_SKULL_OUT {
                HU_SCALP
            } else if r > R_DIPLOE {
                HU_CORTICAL_OUT
            } else if r > R_SKULL_IN {
                HU_DIPLOE
            } else if r > R_BRAIN {
                HU_CORTICAL_IN
            } else {
                HU_BRAIN
            };

            for j in 0..NY {
                hu[[i, j, k]] = voxel_hu;
            }
        }
    }

    let sound_speed = hu.mapv(hu_to_sound_speed);
    let density = hu.mapv(hu_to_density);
    SkullPhantom {
        sound_speed,
        density,
        hu,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Source wavelet
// ─────────────────────────────────────────────────────────────────────────────

/// Centre frequency of the Ricker source wavelet [Hz].
///
/// 150 kHz: λ = 10 mm in water, 3.3 × dx sampling per wavelength at 3 mm.
/// Diagnostic ultrasound TUS range: 100–650 kHz (Marsac 2017; Guasch 2020).
const F0_HZ: f64 = 150_000.0;

/// Peak source pressure [Pa].  100 kPa is a representative clinical TUS level.
///
/// Reference: FDA (2008), diagnostic ultrasound guidance, Table 1.
const P0_PA: f64 = 1.0e5;

/// Ricker (Mexican hat) wavelet.
///
/// ```text
/// w(t) = P₀ · (1 − 2π²f₀²τ²) · exp(−π²f₀²τ²),   τ = t − t_peak
/// ```
///
/// Peak at t_peak = 1.5/f₀ (three half-cycles of build-up before peak).
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
// Acquisition geometry — hemispherical arc
// ─────────────────────────────────────────────────────────────────────────────

/// Number of active elements in the FWI full-ring section.
const FWI_ACTIVE_ELEMENTS: usize = 16;

/// Number of transmit sources on the full-ring arc (every other element).
const N_SHOTS: usize = 8;

/// Number of receivers for each shot: all active transducer samples except the
/// element used as the current transmitter.
const N_RECEIVERS: usize = FWI_ACTIVE_ELEMENTS - 1;

/// FWI gradient descent step size [m/s].
///
/// 50 m/s per iteration with max-norm normalization is conservative for a
/// 1500–2900 m/s skull model.
const STEP_SIZE: f64 = 50.0;

/// Pixel size per model panel [px].
const PANEL: usize = 320;

/// Colorbar height below each panel [px].
const COLORBAR_H: usize = 20;

/// Active transducer element positions sampled from a 1024-element full-ring array.
///
/// # Design rationale
///
/// Sixteen elements are uniformly distributed around the full ring at radius
/// R_ARRAY = 20 voxels from centre (32, 32).  Full-ring coverage provides
/// illumination from all azimuths, eliminating the shadow zone that degrades
/// convergence with superior-hemisphere-only apertures.
///
/// # CPML safety constraint
///
/// The FDTD CPML absorbs energy in cells ix ∈ [0,9] and [54,63], iz ∈ [0,9]
/// and [54,63].  Physical domain: ix ∈ [10,53], iz ∈ [10,53].
///
/// # Geometry derivation
///
/// Centre = (32, 32), R_ARRAY = 20 voxels.
/// Sixteen points at θ_k = k × 22.5°, k = 0..15:
///
/// ```text
/// ix = 32 + round(R_ARRAY · cos θ_k)
/// iz = 32 + round(R_ARRAY · sin θ_k)
/// ```
///
/// k=0  (  0.0°): ix=52, iz=32   k=1  ( 22.5°): ix=50, iz=40
/// k=2  ( 45.0°): ix=46, iz=46   k=3  ( 67.5°): ix=40, iz=50
/// k=4  ( 90.0°): ix=32, iz=52   k=5  (112.5°): ix=24, iz=50
/// k=6  (135.0°): ix=18, iz=46   k=7  (157.5°): ix=14, iz=40
/// k=8  (180.0°): ix=12, iz=32   k=9  (202.5°): ix=14, iz=24
/// k=10 (225.0°): ix=18, iz=18   k=11 (247.5°): ix=24, iz=14
/// k=12 (270.0°): ix=32, iz=12   k=13 (292.5°): ix=40, iz=14
/// k=14 (315.0°): ix=46, iz=18   k=15 (337.5°): ix=50, iz=24
///
/// Reference: Guasch 2020 — full-waveform inversion with complete angular coverage.
const ACTIVE_TRANSDUCER_POSITIONS: [(usize, usize); FWI_ACTIVE_ELEMENTS] = [
    (52, 32), // k=0  (  0.0°)
    (50, 40), // k=1  ( 22.5°)
    (46, 46), // k=2  ( 45.0°)
    (40, 50), // k=3  ( 67.5°)
    (32, 52), // k=4  ( 90.0°)
    (24, 50), // k=5  (112.5°)
    (18, 46), // k=6  (135.0°)
    (14, 40), // k=7  (157.5°)
    (12, 32), // k=8  (180.0°)
    (14, 24), // k=9  (202.5°)
    (18, 18), // k=10 (225.0°)
    (24, 14), // k=11 (247.5°)
    (32, 12), // k=12 (270.0°)
    (40, 14), // k=13 (292.5°)
    (46, 18), // k=14 (315.0°)
    (50, 24), // k=15 (337.5°)
];

/// Transmit subset indexes into `ACTIVE_TRANSDUCER_POSITIONS`.
///
/// Every other element transmits (even indices), giving 8 shots with maximally
/// diverse angular coverage across the full ring.
const TRANSMIT_ELEMENT_INDICES: [usize; N_SHOTS] = [0, 2, 4, 6, 8, 10, 12, 14];

/// Build the receiver mask on the same superior hemispherical transducer arc.
///
/// The transmitting element is excluded to avoid a colocated source/receiver
/// singular sample.  All remaining active transducer positions record.
fn build_receiver_mask(source_element_index: usize) -> Array3<bool> {
    let mut mask = Array3::<bool>::from_elem((NX, NY, NZ), false);
    for (element_index, &(ix, iz)) in ACTIVE_TRANSDUCER_POSITIONS.iter().enumerate() {
        if element_index != source_element_index {
            mask[[ix, 0, iz]] = true;
        }
    }
    mask
}

/// Return the four transmit coordinates used by the current FWI run.
fn transmit_positions() -> Vec<(usize, usize)> {
    TRANSMIT_ELEMENT_INDICES
        .iter()
        .map(|&idx| ACTIVE_TRANSDUCER_POSITIONS[idx])
        .collect()
}

/// Build `FwiGeometry` for one array element with source at `(ix, 0, iz)` and
/// receivers on all other active transducer elements.
///
/// The source signal has `nt` samples.
fn build_shot(source_element_index: usize, f0_hz: f64, nt: usize, dt: f64) -> FwiGeometry {
    let (ix, iz) = ACTIVE_TRANSDUCER_POSITIONS[source_element_index];
    let mut source_mask = Array3::<f64>::zeros((NX, NY, NZ));
    source_mask[[ix, 0, iz]] = 1.0;

    let wavelet = ricker_wavelet(f0_hz, dt, nt);
    let mut p_signal = Array2::<f64>::zeros((1, nt));
    for t in 0..nt {
        p_signal[[0, t]] = wavelet[t];
    }

    let mut source = GridSource::new_empty();
    source.p_mask = Some(source_mask);
    source.p_signal = Some(p_signal);
    source.p_mode = SourceMode::Dirichlet;

    FwiGeometry::new(source, build_receiver_mask(source_element_index))
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian blur for CT-derived initial model
// ─────────────────────────────────────────────────────────────────────────────

/// Separable Gaussian blur of a (NX, NY, NZ) model in the x–z plane (y = 0 only,
/// broadcasted to all NY slices).
///
/// # Algorithm
///
/// Separable 1-D convolutions in x then z with a truncated Gaussian kernel of
/// radius r = ⌈3σ⌉.  Boundary voxels use reflect-padding (clamp at edge).
///
/// # Why CT-derived blur is the clinical standard
///
/// Starting FWI from a homogeneous 1500 m/s background requires the gradient
/// to simultaneously *discover* bone location AND increase bone velocity — a
/// slow, ill-conditioned search.  A Gaussian-blurred CT prior already places
/// bone where it belongs; the FWI only needs to sharpen boundaries and raise
/// peak velocities.  σ = 3 voxels (9 mm) reduces the initial model travel-time
/// error to ≈ 2.7 μs, which is below T/2 at all modelled frequencies.
///
/// Reference: Guasch (2020) npj Digital Medicine — §Methods, CT initial model.
fn gaussian_blur_xz(model: &Array3<f64>, sigma: f64) -> Array3<f64> {
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel_size = 2 * radius + 1;

    // 1-D Gaussian kernel, sum-normalised.
    let raw: Vec<f64> = (0..kernel_size)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let ksum: f64 = raw.iter().sum();
    let kernel: Vec<f64> = raw.iter().map(|&k| k / ksum).collect();

    // Convolve in x-direction → tmp.
    let mut tmp = Array3::<f64>::zeros((NX, NY, NZ));
    for j in 0..NY {
        for k in 0..NZ {
            for i in 0..NX {
                let mut val = 0.0_f64;
                for (ki, &kw) in kernel.iter().enumerate() {
                    let si = (i as isize + ki as isize - radius as isize).clamp(0, NX as isize - 1)
                        as usize;
                    val += kw * model[[si, j, k]];
                }
                tmp[[i, j, k]] = val;
            }
        }
    }

    // Convolve in z-direction → result.
    let mut result = Array3::<f64>::zeros((NX, NY, NZ));
    for j in 0..NY {
        for i in 0..NX {
            for k in 0..NZ {
                let mut val = 0.0_f64;
                for (ki, &kw) in kernel.iter().enumerate() {
                    let sk = (k as isize + ki as isize - radius as isize).clamp(0, NZ as isize - 1)
                        as usize;
                    val += kw * tmp[[i, j, sk]];
                }
                result[[i, j, k]] = val;
            }
        }
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Real CT loading via ritk (compiled only with --features ritk)
// ─────────────────────────────────────────────────────────────────────────────

/// Raw CT volume in voxel space.
///
/// `hu` has shape `(cols, rows, depth)` = `(x, y, z)` in the patient frame.
/// `spacing_mm` is `[dx, dy, dz]` — physical mm per voxel on each axis.

struct CtVolume {
    hu: Array3<f64>,
    spacing_mm: [f64; 3],
}

/// Load a CT volume from either a NIfTI file or a DICOM directory via ritk.
///
/// # Format detection
///
/// - Path is a directory → DICOM series directory.
/// - Path ends with `.nii` or `.nii.gz` → NIfTI file.
///
/// # Axis convention
///
/// Returns `hu[x, y, z]` where:
/// - x = left-right (columns in-plane)
/// - y = anterior-posterior (rows in-plane)
/// - z = superior-inferior (slice axis / depth)
///
/// This matches the convention used in `skull_ct_phase_correction.rs`.

fn load_ct_volume(path: &Path) -> anyhow::Result<CtVolume> {
    type Backend = NdArrayBackend<f32>;
    let device = Default::default();

    // ── PNG series (bone-window secondary-capture) ────────────────────────
    // The "Paired MRI / CT" public dataset stores CT as secondary-capture
    // DICOM (Modality=OT, no spatial metadata) but also supplies PNG exports.
    // `read_png_series` converts them to a [depth, rows, cols] f32 tensor
    // with raw 0-255 pixel values (`.to_luma8()` handles RGB→grayscale).
    //
    // # HU approximation from bone-window PNG
    //
    // The PNGs are rendered with a standard bone window (W=2000, C=400):
    //   display_range = [C − W/2, C + W/2] = [−600, 1400] HU
    //   HU(pixel) = pixel × W/255 + (C − W/2)
    //             = pixel × 7.843 − 600
    //
    // Pixel = 0   → HU = −600  → c = 1500 m/s (water, clamped by bvf)
    // Pixel = 77  → HU =   4   → c = 1503 m/s (≈ water)
    // Pixel = 166 → HU = 702   → c = 2194 m/s (cortical bone)
    // Pixel = 255 → HU = 1400  → c = 2900 m/s (dense cortical bone)
    //
    // Spacing assumption: typical 512×512 head CT with 256 mm FOV
    //   → 0.5 mm/pixel in-plane; 49 slices × ~4 mm slice spacing.
    if path.is_dir() {
        let has_png = std::fs::read_dir(path)
            .with_context(|| format!("failed to read dir '{}'", path.display()))?
            .filter_map(|e| e.ok())
            .any(|e| {
                e.path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("png"))
                    .unwrap_or(false)
            });

        if has_png {
            println!("  PNG series      : {}", path.display());
            let img = read_png_series::<Backend, _>(path, &device)
                .map_err(|e| anyhow::anyhow!("PNG series load failed: {e:#}"))?;
            let [depth, rows, cols] = img.shape();
            let tensor_data = img.data().clone().into_data();
            let values = tensor_data
                .as_slice::<f32>()
                .map_err(|e| anyhow::anyhow!("PNG tensor data is not f32: {e:?}"))?;
            anyhow::ensure!(
                values.len() == depth * rows * cols,
                "PNG data length mismatch: got {}, expected {}",
                values.len(),
                depth * rows * cols
            );

            // Bone window: W=2000, C=400 → HU ∈ [−600, 1400]
            // HU = pixel × (2000/255) − 600
            const PNG_W: f64 = 2000.0;
            const PNG_C: f64 = 400.0;
            let hu_lo = PNG_C - PNG_W / 2.0; // −600.0
            let hu_per_pixel = PNG_W / 255.0; // 7.843

            // Transpose from tensor [Z,Y,X] layout to hu[x,y,z] = [cols,rows,depth]
            let mut hu = Array3::<f64>::zeros((cols, rows, depth));
            for z in 0..depth {
                for y in 0..rows {
                    for x in 0..cols {
                        let px = f64::from(values[z * rows * cols + y * cols + x]);
                        hu[[x, y, z]] = hu_lo + px * hu_per_pixel;
                    }
                }
            }

            // Clamp to physically valid DICOM HU range.
            // Values below −1024 are FOV-padding artefacts (some scanners write −3024 or
            // −4048 outside the reconstructed circle).  Values above 3071 are outside the
            // 12-bit DICOM signed range.  Neither appears in real tissue.
            hu.mapv_inplace(|h| h.clamp(-1024.0, 3071.0));

            // Assumed spacing: 0.5 mm in-plane, 4.0 mm slice (256mm FOV / 512 px)
            return Ok(CtVolume {
                hu,
                spacing_mm: [0.5, 0.5, 4.0],
            });
        }
    }

    let image = if path.is_dir() {
        // ── DICOM directory ───────────────────────────────────────────────
        let series = scan_dicom_directory(path)
            .with_context(|| format!("failed to scan DICOM dir '{}'", path.display()))?;
        if series.is_empty() {
            anyhow::bail!("no DICOM series found in '{}'", path.display());
        }
        let selected = build_dicom_series(series);
        println!(
            "  DICOM series    : '{}' ({} files)",
            selected.series_description,
            selected.file_paths.len()
        );
        load_dicom_series::<Backend>(&selected, &device).map_err(|e| {
            anyhow::anyhow!(
                "DICOM load failed for series '{}': {e:#}",
                selected.series_instance_uid
            )
        })?
    } else {
        // ── NIfTI file ────────────────────────────────────────────────────
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if !name.ends_with(".nii") && !name.ends_with(".nii.gz") {
            anyhow::bail!(
                "unrecognised format for '{}'; expected .nii/.nii.gz or a DICOM dir",
                path.display()
            );
        }
        println!("  NIfTI file      : {}", path.display());
        read_nifti::<Backend, _>(path, &device)
            .with_context(|| format!("NIfTI read failed for '{}'", path.display()))?
    };

    // image shape is [depth/Z, rows/Y, cols/X] in ritk convention
    let [depth, rows, cols] = image.shape();
    let spacing = image.spacing().to_vec();
    anyhow::ensure!(
        spacing.len() == 3,
        "unexpected spacing rank {}",
        spacing.len()
    );

    let tensor_data = image.data().clone().into_data();
    let values = tensor_data
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("tensor data is not f32: {e:?}"))?;
    anyhow::ensure!(
        values.len() == depth * rows * cols,
        "data length mismatch: got {}, expected {}",
        values.len(),
        depth * rows * cols
    );

    // Transpose from tensor [Z,Y,X] layout to hu[x,y,z] = [cols,rows,depth]
    let mut hu = Array3::<f64>::zeros((cols, rows, depth));
    for z in 0..depth {
        for y in 0..rows {
            for x in 0..cols {
                hu[[x, y, z]] = f64::from(values[z * rows * cols + y * cols + x]);
            }
        }
    }

    // Clamp to physically valid DICOM HU range.
    // Values below −1024 are FOV-padding artefacts (some scanners write −3024 or
    // −4048 outside the reconstructed circle).  Values above 3071 are outside the
    // 12-bit DICOM signed range.  Neither appears in real tissue.
    hu.mapv_inplace(|h| h.clamp(-1024.0, 3071.0));

    // spacing[0..2] = [x_spacing, y_spacing, z_spacing] in mm for both
    // DICOM (pixel_spacing + slice_thickness) and NIfTI (affine column norms).
    Ok(CtVolume {
        hu,
        spacing_mm: [spacing[0], spacing[1], spacing[2]],
    })
}

/// Select or build the DICOM series to load.
///
/// # Normal case
///
/// One multi-file series (all slices share a SeriesInstanceUID) → return it.
/// When multiple multi-file series exist, prefer modality == "CT" and most
/// slices.
///
/// # Non-standard case: one file per series
///
/// Some datasets (including the "Paired MRI / CT" public dataset used here)
/// assign a unique SeriesInstanceUID to every individual slice.
/// `scan_dicom_directory` therefore returns N × 1-file series.
///
/// Detection heuristic: all series have ≤ 1 file.
///
/// Fix: merge all file paths from all series into one synthetic
/// `DicomSeriesInfo`.  `load_dicom_series` will sort them spatially by
/// `ImagePositionPatient`, producing a correct 3-D volume regardless of the
/// per-slice UID anomaly.

fn build_dicom_series(mut series: Vec<DicomSeriesInfo>) -> DicomSeriesInfo {
    let max_files = series.iter().map(|s| s.file_paths.len()).max().unwrap_or(0);

    if let Some(best) = series
        .iter()
        .position(|s| s.series_instance_uid == DEFAULT_MEDIMODEL_SERIES_UID)
    {
        return series.swap_remove(best);
    }

    if max_files <= 1 {
        // One-file-per-series dataset: merge everything into a single logical series.
        let all_paths: Vec<_> = series
            .iter_mut()
            .flat_map(|s| s.file_paths.drain(..))
            .collect();
        let n = all_paths.len();
        println!(
            "  Note: each DICOM slice has a unique SeriesInstanceUID; \
                  merging {n} files into one logical series for spatial sort."
        );
        DicomSeriesInfo {
            series_instance_uid: "merged".to_string(),
            series_description: format!("merged-{n}-slices"),
            modality: "CT".to_string(),
            patient_id: String::new(),
            file_paths: all_paths,
        }
    } else {
        // Standard multi-file series: pick the best one.
        let ct: Vec<usize> = series
            .iter()
            .enumerate()
            .filter(|(_, s)| s.modality == "CT")
            .map(|(i, _)| i)
            .collect();
        let pool: Vec<usize> = if ct.is_empty() {
            (0..series.len()).collect()
        } else {
            ct
        };
        let best = pool
            .into_iter()
            .max_by_key(|&i| series[i].file_paths.len())
            .unwrap_or(0);
        series.swap_remove(best)
    }
}

/// Find the axial (z) slice index with the maximum count of bone voxels (HU > 300).
///
/// The equatorial skull cross-section has the largest bone ring area and gives
/// the most informative FWI slice for hemispherical array geometry.

fn skull_equator_z(hu: &Array3<f64>) -> usize {
    let (_, _, nz) = hu.dim();
    (0..nz)
        .max_by_key(|&z| {
            hu.slice(ndarray::s![.., .., z])
                .iter()
                .filter(|&&h| h > 300.0)
                .count()
        })
        .unwrap_or(nz / 2)
}

/// Find the centroid (x_ct, y_ct) of bone voxels on an axial slice.
///
/// Falls back to the geometric centre when no bone is present.

fn skull_centroid_2d(hu: &Array3<f64>, z: usize) -> (f64, f64) {
    let slice = hu.slice(ndarray::s![.., .., z]);
    let (nx, ny) = slice.dim();
    let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0.0f64);
    for ((x, y), &h) in slice.indexed_iter() {
        if h > 300.0 {
            sx += x as f64;
            sy += y as f64;
            n += 1.0;
        }
    }
    if n > 0.0 {
        (sx / n, sy / n)
    } else {
        (nx as f64 / 2.0, ny as f64 / 2.0)
    }
}

/// Bilinear interpolation into an axial HU slice `hu[:, :, z]`.
///
/// Clamps out-of-bound indices to the boundary (reflects water coupling at
/// CT field-of-view edges).

fn bilinear_hu(hu: &Array3<f64>, x: f64, y: f64, z: usize) -> f64 {
    let (nx, ny, nz) = hu.dim();
    if z >= nz {
        return 0.0;
    }
    let cx = |i: isize| i.clamp(0, nx as isize - 1) as usize;
    let cy = |j: isize| j.clamp(0, ny as isize - 1) as usize;
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let h00 = hu[[cx(x0), cy(y0), z]];
    let h10 = hu[[cx(x0 + 1), cy(y0), z]];
    let h01 = hu[[cx(x0), cy(y0 + 1), z]];
    let h11 = hu[[cx(x0 + 1), cy(y0 + 1), z]];
    h00 * (1.0 - fx) * (1.0 - fy) + h10 * fx * (1.0 - fy) + h01 * (1.0 - fx) * fy + h11 * fx * fy
}

/// Measure the outer skull radius in an axial slice.
///
/// Scans all bone voxels (HU > 300) on slice `z` and returns the maximum
/// Euclidean distance from the centroid `(cx, cy)` in CT pixels.
///
/// Returns `nx.min(ny) / 4` as a safe fallback when no bone is found
/// (prevents division-by-zero in the scale computation).

fn skull_outer_radius_ct(hu: &Array3<f64>, z: usize, cx: f64, cy: f64) -> f64 {
    let (nx, ny, _) = hu.dim();
    let r = hu
        .slice(ndarray::s![.., .., z])
        .indexed_iter()
        .filter(|(_, &h)| h > 300.0)
        .map(|((x, y), _)| {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            (dx * dx + dy * dy).sqrt()
        })
        .fold(0.0_f64, f64::max);
    if r < 1.0 {
        (nx.min(ny) / 4) as f64
    } else {
        r
    }
}

/// Resample a 3-D CT volume onto the FWI grid `(NX, NY, NZ)` at spacing `DX`.
///
/// # Algorithm
///
/// 1. Identify the equatorial axial slice (maximum bone area in z).
/// 2. Locate the skull centroid `(cx, cy)` and outer radius `r_skull` [CT px].
/// 3. Compute a scale factor so the skull outer edge lands at `R_HEAD` FWI voxels:
///    ```text
///    scale = r_skull_ct / R_HEAD          [CT pixels per FWI voxel]
///    ```
///    This ensures sources and receivers (placed at R_SRC > R_HEAD) are always
///    outside the skull regardless of the CT field-of-view.  For a real adult
///    head (r_skull ≈ 260 px at 0.5 mm/px) the physically-correct mapping
///    (scale = DX/dx_mm = 6) would project the skull to 43 FWI voxels from
///    centre, placing every source inside the skull and breaking the acquisition
///    geometry.  The skull-fitting scale reduces the effective voxel size to
///    preserve the anatomy-to-geometry relationship intended by the demo.
/// 4. For each FWI voxel `(ix, iz)`:
///    ```text
///    x_ct = cx + (ix − NX/2) · scale
///    y_ct = cy + (iz − NZ/2) · scale
///    ```
/// 5. Bilinear interpolation from the axial CT slice.
/// 6. Broadcast the 2-D result to all `NY` planes.
///
/// FWI ix (lateral) maps to CT x (columns); FWI iz (depth) maps to CT y (rows).

fn resample_ct_to_fwi_grid(vol: &CtVolume) -> Array3<f64> {
    let z_eq = skull_equator_z(&vol.hu);
    let (cx, cy) = skull_centroid_2d(&vol.hu, z_eq);

    // Detect skull outer radius in CT pixels and derive scale so the skull
    // outer edge lands at R_HEAD FWI voxels from the grid centre.
    let r_skull_ct = skull_outer_radius_ct(&vol.hu, z_eq, cx, cy);
    let scale = r_skull_ct / R_HEAD; // CT pixels per FWI voxel

    println!(
        "  CT skull radius : {r_skull_ct:.1} px × {:.2} mm/px = {:.0} mm",
        vol.spacing_mm[0],
        r_skull_ct * vol.spacing_mm[0]
    );
    println!(
        "  FWI fit scale   : {scale:.2} CT px / FWI voxel  \
              (skull outer edge → R_HEAD={R_HEAD} voxels)"
    );

    let mut result = Array3::<f64>::zeros((NX, NY, NZ));
    for ix in 0..NX {
        for iz in 0..NZ {
            let x_ct = cx + (ix as f64 - NX as f64 / 2.0) * scale;
            let y_ct = cy + (iz as f64 - NZ as f64 / 2.0) * scale;
            let hu_val = bilinear_hu(&vol.hu, x_ct, y_ct, z_eq);
            for iy in 0..NY {
                result[[ix, iy, iz]] = hu_val;
            }
        }
    }
    let brain = brain_support_from_hu(&result);
    for ix in 0..NX {
        for iz in 0..NZ {
            if brain[[ix, iz]] && result[[ix, 0, iz]] < 250.0 {
                for iy in 0..NY {
                    result[[ix, iy, iz]] = HU_BRAIN;
                }
            }
        }
    }
    result
}

/// Build the skull phantom, loading real CT when available.
///
/// Priority:
/// 1. `KWAVERS_SEISMIC_CT_PATH` env var (DICOM dir or NIfTI file).
/// 2. `TRANSCRANIAL_CT_PATH` env var for compatibility with older examples.
/// 3. Repository-local Medimodel DICOM directory used by the phase demo.
/// 4. Synthetic circular phantom — last-resort fallback only if CT load fails.
///
/// Priority: KWAVERS_SEISMIC_CT_PATH env var → TRANSCRANIAL_CT_PATH env var →
/// repository-local Medimodel DICOM at DEFAULT_MEDIMODEL_DICOM_DIR.
fn build_phantom_for_demo() -> (SkullPhantom, Option<CtVolume>) {
    let ct_path = std::env::var("KWAVERS_SEISMIC_CT_PATH")
        .or_else(|_| std::env::var("TRANSCRANIAL_CT_PATH"))
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            let default_path = PathBuf::from(DEFAULT_MEDIMODEL_DICOM_DIR);
            default_path.exists().then_some(default_path)
        });

    if let Some(ct_path) = ct_path {
        print!("  CT source       : {}  ", ct_path.display());
        match load_ct_volume(&ct_path) {
            Ok(vol) => {
                let (cx, cy, nz) = vol.hu.dim();
                println!(
                    "({cx}×{cy}×{nz} voxels @ [{:.2},{:.2},{:.2}] mm)",
                    vol.spacing_mm[0], vol.spacing_mm[1], vol.spacing_mm[2]
                );
                let hu_fwi = resample_ct_to_fwi_grid(&vol);
                let sound_speed = hu_fwi.mapv(hu_to_sound_speed);
                let density = hu_fwi.mapv(hu_to_density);
                let phantom = SkullPhantom {
                    sound_speed,
                    density,
                    hu: hu_fwi,
                };
                return (phantom, Some(vol));
            }
            Err(e) => {
                eprintln!("\n  CT load failed: {e:#}");
                eprintln!("  Falling back to synthetic phantom.");
            }
        }
    } else {
        eprintln!(
            "  No CT found at default path '{}'; set KWAVERS_SEISMIC_CT_PATH to override.",
            DEFAULT_MEDIMODEL_DICOM_DIR
        );
    }
    println!("  Phantom         : synthetic fallback");
    (build_skull_phantom(), None)
}

// ─────────────────────────────────────────────────────────────────────────────
// Reconstruction quality metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Print RMSE, Pearson r, max |error|, ±100 m/s fraction vs ground truth.
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

/// Print RMSE, Pearson r, max |error|, ±10 m/s fraction for brain voxels only
/// (geometric: r < R_SKULL_IN from grid center, independent of FWI frozen mask).
///
/// Uses the same geometric boundary as `write_brain_tissue_png` so the quality
/// metrics and visualization are consistent.
fn print_quality_report_brain(true_model: &Array3<f64>, reconstructed: &Array3<f64>) {
    let cx = (NX / 2) as f64;
    let cz = (NZ / 2) as f64;
    let free_pairs: Vec<(f64, f64)> = true_model
        .indexed_iter()
        .filter(|((ix, _iy, iz), _)| {
            let r = (((*ix as f64) - cx).powi(2) + ((*iz as f64) - cz).powi(2)).sqrt();
            r < R_SKULL_IN as f64
        })
        .map(|((ix, _iy, iz), &t)| (t, reconstructed[[ix, _iy, iz]]))
        .collect();
    print_quality_pairs(&free_pairs);
}

fn print_quality_pairs(free_pairs: &[(f64, f64)]) {
    let n = free_pairs.len() as f64;
    if n < 2.0 {
        println!("  (no free voxels)");
        return;
    }
    let l2: f64 = free_pairs.iter().map(|&(t, r)| (t - r).powi(2)).sum();
    let rmse = (l2 / n).sqrt();
    let mean_t = free_pairs.iter().map(|&(t, _)| t).sum::<f64>() / n;
    let mean_r = free_pairs.iter().map(|&(_, r)| r).sum::<f64>() / n;
    let cov: f64 = free_pairs
        .iter()
        .map(|&(t, r)| (t - mean_t) * (r - mean_r))
        .sum();
    let var_t: f64 = free_pairs.iter().map(|&(t, _)| (t - mean_t).powi(2)).sum();
    let var_r: f64 = free_pairs.iter().map(|&(_, r)| (r - mean_r).powi(2)).sum();
    let max_err = free_pairs
        .iter()
        .map(|&(t, r)| (t - r).abs())
        .fold(0.0_f64, f64::max);
    let within_10 = free_pairs
        .iter()
        .filter(|&&(t, r)| (t - r).abs() <= 10.0)
        .count() as f64
        / n
        * 100.0;
    println!("  RMSE            : {rmse:8.2} m/s");
    let denom = (var_t * var_r).sqrt();
    if denom > f64::EPSILON {
        let pearson = cov / denom;
        println!("  Pearson r       : {pearson:8.4}");
    }
    println!("  Max |error|     : {max_err:8.2} m/s");
    println!("  Voxels ±10 m/s  : {within_10:7.1} %");
}

// ─────────────────────────────────────────────────────────────────────────────
// Image output
// ─────────────────────────────────────────────────────────────────────────────

/// Write one RGB pixel into a flat byte buffer.
fn put_pixel(rgb: &mut [u8], width: usize, height: usize, x: usize, y: usize, color: [u8; 3]) {
    if x >= width || y >= height {
        return;
    }
    let idx = 3 * (y * width + x);
    rgb[idx..idx + 3].copy_from_slice(&color);
}

/// Map sound speed to RGB via 5-stop blue → cyan → green → yellow → red.
///
/// Maps c ∈ [c_lo, c_hi] linearly:
/// - 1500 m/s (water / brain) → blue
/// - ~1800 m/s (scalp / soft tissue) → cyan
/// - ~2000 m/s (diploe) → green
/// - ~2450 m/s (dense bone) → yellow
/// - 2900 m/s (cortical bone) → red
fn velocity_color(c: f64, c_lo: f64, c_hi: f64) -> [u8; 3] {
    let t = ((c - c_lo) / (c_hi - c_lo)).clamp(0.0, 1.0);
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

/// Blue ← white → red diverging colormap.  Zero → white; positive → red; negative → blue.
fn diverging_color(value: f64, max_abs: f64) -> [u8; 3] {
    if max_abs < f64::EPSILON {
        return [200, 200, 200];
    }
    let t = (value / max_abs).clamp(-1.0, 1.0);
    if t >= 0.0 {
        let gb = (255.0 * (1.0 - t)) as u8;
        [255, gb, gb]
    } else {
        let rg = (255.0 * (1.0 + t)) as u8;
        [rg, rg, 255]
    }
}

/// Render one velocity model panel (x–z at y = 0) into `rgb`.
fn draw_velocity_panel(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    x_offset: usize,
    y_offset: usize,
    model: &Array3<f64>,
    c_lo: f64,
    c_hi: f64,
) {
    for py in 0..PANEL {
        for px in 0..PANEL {
            let ix = (px * NX / PANEL).min(NX - 1);
            let iz = (py * NZ / PANEL).min(NZ - 1);
            let color = velocity_color(model[[ix, 0, iz]], c_lo, c_hi);
            put_pixel(rgb, width, height, x_offset + px, y_offset + py, color);
        }
    }
}

/// Overlay white 3×3 markers for source positions and yellow for receiver positions.
fn draw_acquisition_markers(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    x_offset: usize,
    y_offset: usize,
    shot_positions: &[(usize, usize)],
    active_elements: &[(usize, usize)],
) {
    // Receiver markers (yellow): active sparse transducer section.
    for &(ix, iz) in active_elements {
        let rx = ix * PANEL / NX;
        let rz = iz * PANEL / NZ;
        for dy in 0_usize..=2 {
            for dx in 0_usize..=2 {
                put_pixel(
                    rgb,
                    width,
                    height,
                    x_offset + rx.saturating_sub(1) + dx,
                    y_offset + rz.saturating_sub(1) + dy,
                    [255, 255, 0],
                );
            }
        }
    }

    // Source markers (white): transmit subset.
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
                    y_offset + sz.saturating_sub(1) + dy,
                    [255, 255, 255], // white — sources
                );
            }
        }
    }
}

/// Draw a velocity colorbar strip at y = y_offset + PANEL .. y_offset + PANEL + COLORBAR_H.
fn draw_colorbar(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    x_offset: usize,
    y_offset: usize,
    c_lo: f64,
    c_hi: f64,
) {
    for px in 0..PANEL {
        let t = px as f64 / (PANEL - 1) as f64;
        let c = c_lo + t * (c_hi - c_lo);
        let color = velocity_color(c, c_lo, c_hi);
        for dy in 0..COLORBAR_H {
            put_pixel(
                rgb,
                width,
                height,
                x_offset + px,
                y_offset + PANEL + dy,
                color,
            );
        }
    }
}

/// Write a six-panel PNG: top row = CT anatomical triplanar, bottom row = FWI reconstruction.
///
/// # Layout
///
/// ```text
/// ┌──────────────┬──────────────┬──────────────┐  ← PANEL rows    (top)
/// │  CT coronal  │   CT axial   │ CT sagittal  │  bone-window CT
/// │  x-z @ y_c  │  x-y @ z_c   │  y-z @ x_c  │
/// ├──────────────┴──────────────┴──────────────┤  ← COLORBAR_H rows (CT colorbar)
/// ├──────────────┬──────────────┬──────────────┤  ← PANEL rows    (bottom)
/// │  FWI true    │ FWI reconstr │  FWI diff    │  velocity colormap
/// │ (CT-derived) │  (inverted)  │  (R − T)     │
/// ├──────────────┴──────────────┴──────────────┤  ← COLORBAR_H rows (velocity colorbar)
/// └──────────────────────────────────────────────┘
///   3 × PANEL columns total
/// ```
///
/// When `ct_vol` is `None` the top row is omitted and the image is the standard
/// three-panel true | reconstructed | difference layout.
fn write_three_plane_png(
    path: &Path,
    true_model: &Array3<f64>,
    reconstructed: &Array3<f64>,
    c_lo: f64,
    c_hi: f64,
    shot_positions: &[(usize, usize)],
    active_elements: &[(usize, usize)],
    ct_vol: Option<&CtVolume>,
) -> io::Result<()> {
    let img_w = 3 * PANEL;

    if let Some(vol) = ct_vol {
        // ── 3×2 grid: CT triplanar (top) + FWI reconstruction (bottom) ───
        // Row heights: PANEL + COLORBAR_H for CT, then PANEL + COLORBAR_H for FWI.
        let img_h = 2 * (PANEL + COLORBAR_H);
        let mut rgb = vec![0_u8; img_w * img_h * 3];

        let (nx_ct, ny_ct, nz_ct) = vol.hu.dim();
        let cy_ct = ny_ct / 2;
        let cz_ct = nz_ct / 2;
        let cx_ct = nx_ct / 2;

        // ── Top row: CT triplanar ─────────────────────────────────────────
        // Panel (0,0): Coronal — x-z @ y = cy_ct.
        for py in 0..PANEL {
            for px in 0..PANEL {
                let ix = (px * nx_ct / PANEL).min(nx_ct - 1);
                let iz = (py * nz_ct / PANEL).min(nz_ct - 1);
                put_pixel(
                    &mut rgb,
                    img_w,
                    img_h,
                    px,
                    py,
                    ct_bone_color(vol.hu[[ix, cy_ct, iz]]),
                );
            }
        }
        // Panel (1,0): Axial — x-y @ z = cz_ct.
        for py in 0..PANEL {
            for px in 0..PANEL {
                let ix = (px * nx_ct / PANEL).min(nx_ct - 1);
                let iy = (py * ny_ct / PANEL).min(ny_ct - 1);
                put_pixel(
                    &mut rgb,
                    img_w,
                    img_h,
                    PANEL + px,
                    py,
                    ct_bone_color(vol.hu[[ix, iy, cz_ct]]),
                );
            }
        }
        // Panel (2,0): Sagittal — y-z @ x = cx_ct.
        for py in 0..PANEL {
            for px in 0..PANEL {
                let iy = (px * ny_ct / PANEL).min(ny_ct - 1);
                let iz = (py * nz_ct / PANEL).min(nz_ct - 1);
                put_pixel(
                    &mut rgb,
                    img_w,
                    img_h,
                    2 * PANEL + px,
                    py,
                    ct_bone_color(vol.hu[[cx_ct, iy, iz]]),
                );
            }
        }
        // CT bone-window colorbar spanning all three top panels.
        const HU_CB_MIN: f64 = -600.0;
        const HU_CB_MAX: f64 = 1400.0;
        for px in 0..img_w {
            let t = px as f64 / (img_w - 1) as f64;
            let hu = HU_CB_MIN + t * (HU_CB_MAX - HU_CB_MIN);
            let color = ct_bone_color(hu);
            for dy in 0..COLORBAR_H {
                put_pixel(&mut rgb, img_w, img_h, px, PANEL + dy, color);
            }
        }

        // ── Bottom row: FWI reconstruction ────────────────────────────────
        let fwi_y0 = PANEL + COLORBAR_H; // y-pixel where FWI row starts

        // Panel (0,1): FWI true velocity — coronal x-z @ y=0.
        draw_velocity_panel(&mut rgb, img_w, img_h, 0, fwi_y0, true_model, c_lo, c_hi);
        draw_acquisition_markers(
            &mut rgb,
            img_w,
            img_h,
            0,
            fwi_y0,
            shot_positions,
            active_elements,
        );
        draw_colorbar(&mut rgb, img_w, img_h, 0, fwi_y0, c_lo, c_hi);

        // Panel (1,1): FWI reconstructed velocity — coronal x-z @ y=0.
        draw_velocity_panel(
            &mut rgb,
            img_w,
            img_h,
            PANEL,
            fwi_y0,
            reconstructed,
            c_lo,
            c_hi,
        );
        draw_acquisition_markers(
            &mut rgb,
            img_w,
            img_h,
            PANEL,
            fwi_y0,
            shot_positions,
            active_elements,
        );
        draw_colorbar(&mut rgb, img_w, img_h, PANEL, fwi_y0, c_lo, c_hi);

        // Panel (2,1): Signed difference (reconstructed − true).
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
                put_pixel(
                    &mut rgb,
                    img_w,
                    img_h,
                    2 * PANEL + px,
                    fwi_y0 + py,
                    diverging_color(delta, max_diff),
                );
            }
        }
        for px in 0..PANEL {
            let signed = (2.0 * px as f64 / (PANEL - 1) as f64 - 1.0) * max_diff;
            let color = diverging_color(signed, max_diff);
            for dy in 0..COLORBAR_H {
                put_pixel(
                    &mut rgb,
                    img_w,
                    img_h,
                    2 * PANEL + px,
                    fwi_y0 + PANEL + dy,
                    color,
                );
            }
        }

        write_png(path, &rgb, img_w, img_h)
    } else {
        // ── Fallback: True | Reconstructed | Difference (coronal x-z) ────
        let img_h = PANEL + COLORBAR_H;
        let mut rgb = vec![0_u8; img_w * img_h * 3];

        draw_velocity_panel(&mut rgb, img_w, img_h, 0, 0, true_model, c_lo, c_hi);
        draw_acquisition_markers(
            &mut rgb,
            img_w,
            img_h,
            0,
            0,
            shot_positions,
            active_elements,
        );
        draw_colorbar(&mut rgb, img_w, img_h, 0, 0, c_lo, c_hi);

        draw_velocity_panel(&mut rgb, img_w, img_h, PANEL, 0, reconstructed, c_lo, c_hi);
        draw_acquisition_markers(
            &mut rgb,
            img_w,
            img_h,
            PANEL,
            0,
            shot_positions,
            active_elements,
        );
        draw_colorbar(&mut rgb, img_w, img_h, PANEL, 0, c_lo, c_hi);

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
                put_pixel(&mut rgb, img_w, img_h, 2 * PANEL + px, py, color);
            }
        }
        for px in 0..PANEL {
            let signed = (2.0 * px as f64 / (PANEL - 1) as f64 - 1.0) * max_diff;
            let color = diverging_color(signed, max_diff);
            for dy in 0..COLORBAR_H {
                put_pixel(&mut rgb, img_w, img_h, 2 * PANEL + px, PANEL + dy, color);
            }
        }
        write_png(path, &rgb, img_w, img_h)
    }
}

/// Map a Hounsfield unit to a grayscale RGB triplet using the bone window.
///
/// Bone window: W = 2000, C = 400 → display range [C − W/2, C + W/2] = [−600, 1400].
/// HU ≤ −600 maps to black (0,0,0); HU ≥ 1400 maps to white (255,255,255).
#[inline]
fn ct_bone_color(hu: f64) -> [u8; 3] {
    const HU_MIN: f64 = -600.0;
    const HU_MAX: f64 = 1400.0;
    let t = ((hu - HU_MIN) / (HU_MAX - HU_MIN)).clamp(0.0, 1.0);
    let v = (t * 255.0).round() as u8;
    [v, v, v]
}

/// Write a 4-panel PPM: true | initial | reconstructed | error.
pub fn write_velocity_panels(
    path: &Path,
    true_model: &Array3<f64>,
    initial_model: &Array3<f64>,
    reconstructed: &Array3<f64>,
    shot_positions: &[(usize, usize)],
    active_elements: &[(usize, usize)],
) -> std::io::Result<()> {
    let img_w = 4 * PANEL;
    let img_h = PANEL + COLORBAR_H;
    let mut rgb = vec![0_u8; img_w * img_h * 3];

    draw_velocity_panel(&mut rgb, img_w, img_h, 0, 0, true_model, C_LO, C_HI);
    draw_acquisition_markers(
        &mut rgb,
        img_w,
        img_h,
        0,
        0,
        shot_positions,
        active_elements,
    );
    draw_colorbar(&mut rgb, img_w, img_h, 0, 0, C_LO, C_HI);

    draw_velocity_panel(&mut rgb, img_w, img_h, PANEL, 0, initial_model, C_LO, C_HI);
    draw_acquisition_markers(
        &mut rgb,
        img_w,
        img_h,
        PANEL,
        0,
        shot_positions,
        active_elements,
    );
    draw_colorbar(&mut rgb, img_w, img_h, PANEL, 0, C_LO, C_HI);

    draw_velocity_panel(
        &mut rgb,
        img_w,
        img_h,
        2 * PANEL,
        0,
        reconstructed,
        C_LO,
        C_HI,
    );
    draw_acquisition_markers(
        &mut rgb,
        img_w,
        img_h,
        2 * PANEL,
        0,
        shot_positions,
        active_elements,
    );
    draw_colorbar(&mut rgb, img_w, img_h, 2 * PANEL, 0, C_LO, C_HI);

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
    for px in 0..PANEL {
        let signed = (2.0 * px as f64 / (PANEL - 1) as f64 - 1.0) * max_diff;
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

/// Estimate intracranial brain support from the CT-derived HU map.
///
/// For each image row, the leftmost and rightmost bone voxels define the skull
/// envelope. Non-bone voxels between those bounds are labelled brain/CSF. This
/// fills skull-focused CT cavities with average parenchyma support for the FWI
/// brain target and for the diagnostic prior image.
fn brain_support_from_hu(hu: &Array3<f64>) -> Array2<bool> {
    let mut mask = Array2::<bool>::from_elem((NX, NZ), false);
    for iz in 0..NZ {
        let bone: Vec<usize> = (0..NX).filter(|&ix| hu[[ix, 0, iz]] >= 250.0).collect();
        if bone.len() < 2 {
            continue;
        }
        let left = bone[0];
        let right = *bone.last().expect("bone len checked");
        if right <= left + 2 {
            continue;
        }
        for ix in (left + 1)..right {
            if hu[[ix, 0, iz]] < 250.0 {
                mask[[ix, iz]] = true;
            }
        }
    }
    mask
}

/// Write a CT-derived brain/skull prior PNG with the sparse transducer section.
pub fn write_brain_prior_png(
    path: &Path,
    hu: &Array3<f64>,
    shot_positions: &[(usize, usize)],
    active_elements: &[(usize, usize)],
) -> io::Result<()> {
    let img_w = PANEL;
    let img_h = PANEL;
    let mut rgb = vec![0_u8; img_w * img_h * 3];
    let brain = brain_support_from_hu(hu);

    for py in 0..PANEL {
        for px in 0..PANEL {
            let ix = (px * NX / PANEL).min(NX - 1);
            let iz = (py * NZ / PANEL).min(NZ - 1);
            let h = hu[[ix, 0, iz]];
            let color = if h >= 700.0 {
                [246, 246, 236]
            } else if h >= 250.0 {
                [178, 166, 142]
            } else if brain[[ix, iz]] {
                [206, 78, 112]
            } else if h < -200.0 {
                [24, 34, 58]
            } else {
                [28, 92, 142]
            };
            put_pixel(&mut rgb, img_w, img_h, px, py, color);
        }
    }

    draw_acquisition_markers(
        &mut rgb,
        img_w,
        img_h,
        0,
        0,
        shot_positions,
        active_elements,
    );
    write_png(path, &rgb, img_w, img_h)
}

/// Write a single-panel PPM of the RTM image (diverging colormap).
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
            let color = diverging_color(rtm_image[[ix, 0, iz]], max_abs);
            put_pixel(&mut rgb, img_w, img_h, px, py, color);
        }
    }

    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "P6\n{} {}\n255", img_w, img_h)?;
    out.write_all(&rgb)?;
    Ok(())
}

/// Write a three-panel brain tissue PNG using the tight [BRAIN_C_MIN, BRAIN_C_MAX] colormap.
///
/// # Layout
///
/// ```text
/// ┌───────────────┬───────────────┬───────────────┐  ← PANEL rows
/// │  True brain   │ FWI reconstr  │  Difference   │
/// │  (MNI prior)  │  (Stage 2)    │  (R − T)      │
/// ├───────────────┴───────────────┴───────────────┤  ← COLORBAR_H rows
/// └─────────────────────────────────────────────────┘
/// ```
///
/// Skull voxels are rendered in gray tiers to distinguish them from soft tissue.
/// The tight velocity range [1480, 1560] m/s makes the ~40 m/s gray/white
/// matter contrast visible.  Reference: Duck (1990) — tissue acoustic properties.
///
/// # Geometry-driven coloring
///
/// Brain vs. skull/scalp distinction uses the geometric r < R_SKULL_IN criterion
/// (distance from grid center, in voxels) rather than the FWI frozen mask.
/// The FWI frozen mask is velocity-threshold based (~120 cortical bone voxels);
/// using it here would color the scalp ring with `velocity_color(1556, 1480, 1560)`,
/// mapping scalp velocity near the colormap top (yellow), creating a false yellow
/// annulus that visually breaks alignment between the brain region and skull.
///
/// The CT-derived velocity at each non-brain voxel determines the gray shade:
///   c ≥ BONE_VELOCITY_THRESHOLD  → light gray  [200, 200, 200]  (bone / diploe)
///   1502 ≤ c < 1714              → medium gray [140, 140, 140]  (scalp / soft tissue)
///   c < 1502                     → dark        [ 40,  40,  40]  (water coupling bath)
fn write_brain_tissue_png(
    path: &Path,
    true_model: &Array3<f64>,
    reconstructed: &Array3<f64>,
) -> io::Result<()> {
    let img_w = 3 * PANEL;
    let img_h = PANEL + COLORBAR_H;
    let mut rgb = vec![0_u8; img_w * img_h * 3];

    let cx = (NX / 2) as f64;
    let cz = (NZ / 2) as f64;

    // Geometric brain test: voxel (ix, iz) is inside brain when its radial
    // distance from grid center is strictly less than R_SKULL_IN voxels.
    let is_brain = |ix: usize, iz: usize| -> bool {
        let r = ((ix as f64 - cx).powi(2) + (iz as f64 - cz).powi(2)).sqrt();
        r < R_SKULL_IN
    };

    // Non-brain voxel coloring: 3-tier based on CT-derived velocity.
    // The true_model retains CT velocity at skull/scalp/water positions because
    // build_brain_velocity_model only writes MNI tissue velocities for r < R_SKULL_IN.
    let frozen_color = |c_ref: f64| -> [u8; 3] {
        if c_ref >= BONE_VELOCITY_THRESHOLD {
            [200, 200, 200] // bone / diploe
        } else if c_ref >= 1502.0 {
            [140, 140, 140] // scalp / soft tissue coupling
        } else {
            [40, 40, 40] // water coupling bath
        }
    };

    // ── Panel 0: true brain tissue velocity ───────────────────────────────
    for py in 0..PANEL {
        for px in 0..PANEL {
            let ix = (px * NX / PANEL).min(NX - 1);
            let iz = (py * NZ / PANEL).min(NZ - 1);
            let color = if is_brain(ix, iz) {
                velocity_color(true_model[[ix, 0, iz]], BRAIN_C_MIN, BRAIN_C_MAX)
            } else {
                frozen_color(true_model[[ix, 0, iz]])
            };
            put_pixel(&mut rgb, img_w, img_h, px, py, color);
        }
    }
    for px in 0..PANEL {
        let t = px as f64 / (PANEL - 1) as f64;
        let c = BRAIN_C_MIN + t * (BRAIN_C_MAX - BRAIN_C_MIN);
        let color = velocity_color(c, BRAIN_C_MIN, BRAIN_C_MAX);
        for dy in 0..COLORBAR_H {
            put_pixel(&mut rgb, img_w, img_h, px, PANEL + dy, color);
        }
    }

    // ── Panel 1: reconstructed brain tissue velocity ───────────────────────
    for py in 0..PANEL {
        for px in 0..PANEL {
            let ix = (px * NX / PANEL).min(NX - 1);
            let iz = (py * NZ / PANEL).min(NZ - 1);
            let color = if is_brain(ix, iz) {
                velocity_color(reconstructed[[ix, 0, iz]], BRAIN_C_MIN, BRAIN_C_MAX)
            } else {
                frozen_color(reconstructed[[ix, 0, iz]])
            };
            put_pixel(&mut rgb, img_w, img_h, PANEL + px, py, color);
        }
    }
    for px in 0..PANEL {
        let t = px as f64 / (PANEL - 1) as f64;
        let c = BRAIN_C_MIN + t * (BRAIN_C_MAX - BRAIN_C_MIN);
        let color = velocity_color(c, BRAIN_C_MIN, BRAIN_C_MAX);
        for dy in 0..COLORBAR_H {
            put_pixel(&mut rgb, img_w, img_h, PANEL + px, PANEL + dy, color);
        }
    }

    // ── Panel 2: signed difference (reconstructed − true) ─────────────────
    // Scale to max observed error among brain voxels (r < R_SKULL_IN), clamped
    // to ≥ 20 m/s so the colorbar has a meaningful range even if errors are small.
    let max_diff = true_model
        .indexed_iter()
        .filter(|((ix, _, iz), _)| is_brain(*ix, *iz))
        .map(|((ix, _, iz), &t)| (reconstructed[[ix, 0, iz]] - t).abs())
        .fold(0.0_f64, f64::max)
        .max(20.0);

    for py in 0..PANEL {
        for px in 0..PANEL {
            let ix = (px * NX / PANEL).min(NX - 1);
            let iz = (py * NZ / PANEL).min(NZ - 1);
            let color = if is_brain(ix, iz) {
                let delta = reconstructed[[ix, 0, iz]] - true_model[[ix, 0, iz]];
                diverging_color(delta, max_diff)
            } else {
                frozen_color(true_model[[ix, 0, iz]])
            };
            put_pixel(&mut rgb, img_w, img_h, 2 * PANEL + px, py, color);
        }
    }
    for px in 0..PANEL {
        let signed = (2.0 * px as f64 / (PANEL - 1) as f64 - 1.0) * max_diff;
        let color = diverging_color(signed, max_diff);
        for dy in 0..COLORBAR_H {
            put_pixel(&mut rgb, img_w, img_h, 2 * PANEL + px, PANEL + dy, color);
        }
    }

    write_png(path, &rgb, img_w, img_h)
}

/// Write the central-column (x = NX/2) velocity profiles to CSV.
///
/// Columns: depth_mm, true_c, initial_c, reconstructed_c, error_m_per_s
pub fn write_velocity_csv(
    path: &Path,
    true_model: &Array3<f64>,
    initial_model: &Array3<f64>,
    reconstructed: &Array3<f64>,
) -> std::io::Result<()> {
    let cx = NX / 2;
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(
        out,
        "depth_mm,true_c_m_per_s,initial_c_m_per_s,reconstructed_c_m_per_s,error_m_per_s"
    )?;
    for k in 0..NZ {
        let depth_mm = k as f64 * DX * 1e3;
        let t_c = true_model[[cx, 0, k]];
        let i_c = initial_model[[cx, 0, k]];
        let r_c = reconstructed[[cx, 0, k]];
        writeln!(
            out,
            "{depth_mm:.2},{t_c:.2},{i_c:.2},{r_c:.2},{:.2}",
            r_c - t_c
        )?;
    }
    Ok(())
}

/// Encode a flat RGB buffer as PNG.
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

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> KwaversResult<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("warn"));

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   Transcranial Ultrasound FWI — Brain Reconstruction     ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ── 1. Skull phantom ──────────────────────────────────────────────────
    println!("[ 1 / 6 ]  Building skull phantom …");
    let (phantom, ct_vol) = build_phantom_for_demo();

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
        "  Grid            : {NX}×{NY}×{NZ} voxels @ {:.0} mm",
        DX * 1e3
    );
    println!(
        "  Domain          : {:.0}×{:.0} mm",
        NX as f64 * DX * 1e3,
        NZ as f64 * DX * 1e3
    );
    println!("  HU range        : [{hu_min:.0}, {hu_max:.0}]");
    println!("  Sound-speed     : [{c_min:.0}, {c_max:.0}] m/s");
    println!(
        "  Head radius     : {:.0} mm  (R_HEAD = {R_HEAD} voxels)",
        R_HEAD * DX * 1e3
    );
    println!(
        "  Skull thickness : ~{:.0} mm  (outer cortical → inner cortical)",
        (R_SKULL_OUT - R_SKULL_IN) * DX * 1e3
    );
    println!("  Brain radius    : {:.0} mm", R_BRAIN * DX * 1e3);
    println!("  Layers          : water coupling / scalp / cortical bone / diploe / brain");

    // ── 2. Grid ───────────────────────────────────────────────────────────
    println!("\n[ 2 / 6 ]  Constructing computational grid …");
    let grid = Grid::new(NX, NY, NZ, DX, DX, DX)?;
    println!("  Grid OK");

    // ── 3. Multi-scale FWI parameters ────────────────────────────────────
    println!("\n[ 3 / 6 ]  Configuring multi-scale FWI …");

    // CFL-stable timestep: dt ≤ 0.3 × dx / (c_max × √3).
    // Fixed across all scales — determined by maximum velocity, not frequency.
    let dt = 0.3 * DX / (c_max * 3.0_f64.sqrt());

    // Full domain transit time at water speed (same for all scales).
    let t_transit = (NX as f64 * DX) / C_WATER;

    // Multi-scale frequency schedule (Guasch 2020, §Methods — frequency continuation).
    //
    // # Cycle-skipping criterion (Virieux & Operto 2009)
    //
    // Cycle-skipping occurs when the initial model travel-time error exceeds T/2.
    // Skull transmission delay:
    //   Δt = skull_thickness × (1/c_water − 1/c_skull_avg)
    //      = 24mm × (1/1500 − 1/2264) = 5.4 μs
    //
    // At 150 kHz: T/2 = 3.3 μs <  Δt = 5.4 μs → CYCLE-SKIPPING ✗
    // At  60 kHz: T/2 = 8.3 μs >  Δt = 5.4 μs → safe ✓  (start here)
    //
    // # CPML absorption adequacy (why 20 kHz is excluded)
    //
    // CPML absorbs effectively only when PML thickness ≥ λ/4 in the absorbing medium.
    // Physical CPML thickness = 10 cells × 3 mm = 30 mm.
    // In bone (c_bone ≈ 2500 m/s):
    //   λ_bone(20 kHz)  = 2500/20000 = 125 mm →  λ/4 = 31 mm ≈ CPML (marginal)
    //   λ_bone(60 kHz)  = 2500/60000 =  42 mm →  λ/4 = 10.5 mm << CPML (adequate)
    //   λ_bone(150 kHz) = 2500/150000 = 17 mm →  λ/4 = 4.2 mm << CPML (adequate)
    // At 20 kHz the CPML absorbs less than one λ/4 through bone → reflections
    // overwhelm the recorded wavefield, producing J ≈ 10⁹ Pa²·s (catastrophic).
    // Minimum usable frequency given this CPML thickness: ≈ 50 kHz.
    //
    // Schedule: 40 kHz → 80 kHz → 150 kHz with 10-12-15 iterations per scale.
    // Each scale starts from the previous scale's result; nt is computed per scale
    // to include 3 source periods (for the low-frequency wavelet to decay fully).
    //
    //   nt(f₀) = ceil((t_transit × 1.2  +  3.0 / f₀) / dt)
    //
    // Three scales improve initial model recovery at the lowest frequency before
    // refining skull boundaries at intermediate and full resolution.
    //
    // At 40 kHz: T/2 = 12.5 μs > Δt_skull = 5.4 μs → safe ✓ (start here)
    // At 80 kHz: T/2 =  6.25 μs > Δt_skull = 5.4 μs → safe ✓
    // At 150 kHz: T/2 = 3.3 μs < Δt_skull = 5.4 μs → cycle-skipping if from
    //   uniform 1500 m/s, but safe when starting from 80 kHz result.
    //
    // Initial model: Gaussian-blurred CT (σ = 3 voxels).  At 40 kHz,
    // T/2 = 12.5 μs > Δt_skull(blurred) ≈ 2.7 μs → no cycle-skipping.
    //
    // Physical constraint: all tissues have c ≥ c_water = 1500 m/s.
    // Sub-water-speed artefacts are clamped after each scale.
    //
    // Source mute radius: scaled with wavelength = floor(c_water / (2·f₀·dx)).
    // At 40 kHz:  radius = floor(1500/(2×40000×0.003)) = 6 voxels.
    // At 80 kHz:  radius = floor(1500/(2×80000×0.003)) = 3 voxels.
    // At 150 kHz: radius = floor(1500/(2×150000×0.003)) = 2 voxels (clamped to 2 minimum).
    let scales: &[(f64, usize)] = &[
        (40_000.0, 10),  // f₀ Hz, n_iter — safe from blurred CT prior (T/2 > Δt_skull)
        (80_000.0, 12),  // intermediate refinement — still cycle-skip safe
        (150_000.0, 15), // refine skull boundaries at full ultrasound resolution
    ];

    println!("  dt              : {:.1} ns", dt * 1e9);
    println!(
        "  Scales          : {} → {} → {} kHz  (10-12-15 iterations)",
        scales[0].0 * 1e-3,
        scales[1].0 * 1e-3,
        scales[2].0 * 1e-3
    );
    for &(f0, n) in scales {
        let nt_s = ((t_transit * 1.2 + 3.0 / f0) / dt).ceil() as usize;
        let t_half = 1.0 / (2.0 * f0) * 1e6;
        println!(
            "    f₀={:.0} kHz: T/2={:.1} μs, Δt_skull=5.4 μs → {}, nt={}, {} iter",
            f0 * 1e-3,
            t_half,
            if t_half > 5.4 { "OK" } else { "WARN" },
            nt_s,
            n
        );
    }

    // ── 4. Hemispherical acquisition geometry ─────────────────────────────
    println!("\n[ 4 / 6 ]  Building hemispherical acquisition geometry …");
    println!(
        "  Full aperture    : {TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT} elements, 650 kHz design authority"
    );
    println!("  FWI section      : {FWI_ACTIVE_ELEMENTS} active full-ring samples");
    println!("  Transmits        : {N_SHOTS} shots; receivers/shot = {N_RECEIVERS} on same arc");
    for (s, &element_index) in TRANSMIT_ELEMENT_INDICES.iter().enumerate() {
        let (ix, iz) = ACTIVE_TRANSDUCER_POSITIONS[element_index];
        println!(
            "  Shot {:1}: (x={:2}, y=0, z={:2}) = ({:.1} mm, {:.1} mm)",
            s,
            ix,
            iz,
            ix as f64 * DX * 1e3,
            iz as f64 * DX * 1e3
        );
    }

    // ── 5. Multi-scale FWI ────────────────────────────────────────────────
    println!("\n[ 5 / 6 ]  Running multi-scale transcranial FWI …");

    let true_model = phantom.sound_speed.clone();

    // Initial model: Gaussian-blurred true skull model (σ = 3 voxels ≈ 9 mm).
    //
    // This is the standard clinical approach (Guasch 2020): a low-resolution CT
    // scan is always available and provides a smooth but geometrically correct
    // bone map.  Starting from this blurred prior:
    //   • Skull structure is already approximately located → gradient acts as a
    //     *refinement* operator (sharpen boundaries, raise bone peak velocity)
    //     rather than a discovery operator (find bone from featureless water).
    //   • Blurred initial travel-time error ≈ 2.7 μs < T/2(60 kHz) = 8.3 μs
    //     → no cycle-skipping at any modelled frequency.
    //   • FWI converges in far fewer iterations than from uniform 1500 m/s.
    //
    // Convergence evidence (from uniform initial): after 13 iterations at
    // 60 → 150 kHz, c_max reached only 1634 m/s (true 2508 m/s) — gradient
    // spent all budget discovering skull geometry rather than refining it.
    let initial_model = gaussian_blur_xz(&true_model, 3.0);
    let mut current_model = initial_model.clone();

    // Compute J₀ at the finest scale (150 kHz) for reporting consistency.
    let nt_fine = ((t_transit * 1.2 + 3.0 / F0_HZ) / dt).ceil() as usize;
    let mut shots_fine: Vec<(FwiGeometry, Array2<f64>)> = Vec::with_capacity(N_SHOTS);
    {
        let tmp_fwi = FwiProcessor::new(FwiParameters {
            max_iterations: 1,
            frequency: F0_HZ,
            nt: nt_fine,
            dt,
            n_trace: N_RECEIVERS,
            n_depth: 1,
            step_size: STEP_SIZE,
            tolerance: 1e-12,
            regularization: RegularizationParameters {
                tikhonov_weight: 0.0,
                tv_weight: 0.0,
                smoothness_weight: 0.0,
            },
            source_mute_radius: 4,
            ..FwiParameters::default()
        });
        let t0 = Instant::now();
        for &element_index in &TRANSMIT_ELEMENT_INDICES {
            let geom = build_shot(element_index, F0_HZ, nt_fine, dt);
            let obs = tmp_fwi.generate_synthetic_data(&true_model, &geom, &grid)?;
            shots_fine.push((geom, obs));
        }
        println!(
            "  {} observed gathers at {} kHz ({:.1} s)",
            N_SHOTS,
            F0_HZ * 1e-3,
            t0.elapsed().as_secs_f32()
        );
    }

    let j_initial = {
        let fwi_tmp = FwiProcessor::new(FwiParameters {
            max_iterations: 1,
            frequency: F0_HZ,
            nt: nt_fine,
            dt,
            n_trace: N_RECEIVERS,
            n_depth: 1,
            step_size: STEP_SIZE,
            tolerance: 1e-12,
            regularization: RegularizationParameters {
                tikhonov_weight: 0.0,
                tv_weight: 0.0,
                smoothness_weight: 0.0,
            },
            source_mute_radius: 4,
            ..FwiParameters::default()
        });
        let mut j = 0.0_f64;
        for (geom, obs) in &shots_fine {
            let d_syn = fwi_tmp.generate_synthetic_data(&initial_model, geom, &grid)?;
            j += d_syn
                .iter()
                .zip(obs.iter())
                .map(|(&s, &o)| (s - o).powi(2))
                .sum::<f64>()
                * 0.5
                * dt;
        }
        j
    };

    println!("\n  Quality before inversion:");
    print_quality_report(&true_model, &initial_model);
    println!("  Joint J₀ (150 kHz) : {j_initial:.6e} Pa²·s  ({N_SHOTS} shots)");

    let t_inv = Instant::now();

    // Multi-scale inversion loop.
    for (scale_idx, &(f0, n_iter)) in scales.iter().enumerate() {
        // Compute scale-specific nt: include 3 source periods + transit time.
        // At 60 kHz the Ricker wavelet has t_peak = 25 μs; at 150 kHz, 10 μs.
        let nt_scale = ((t_transit * 1.2 + 3.0 / f0) / dt).ceil() as usize;

        // Source mute radius = half-wavelength in voxels (clamped to [2, 12]).
        let mute_r = ((C_WATER / (2.0 * f0)) / DX).floor() as usize;
        let mute_r = mute_r.clamp(2, 12);

        // Build shots at this frequency.
        let mut scale_shots: Vec<(FwiGeometry, Array2<f64>)> = Vec::with_capacity(N_SHOTS);
        let fwi_scale = FwiProcessor::new(FwiParameters {
            max_iterations: n_iter,
            frequency: f0,
            nt: nt_scale,
            dt,
            n_trace: N_RECEIVERS,
            n_depth: 1,
            step_size: STEP_SIZE,
            tolerance: 1e-12,
            regularization: RegularizationParameters {
                tikhonov_weight: 0.0,
                tv_weight: 0.0,
                smoothness_weight: 0.0,
            },
            source_mute_radius: mute_r,
            ..FwiParameters::default()
        });

        let t_scale = Instant::now();
        for &element_index in &TRANSMIT_ELEMENT_INDICES {
            let geom = build_shot(element_index, f0, nt_scale, dt);
            let obs = fwi_scale.generate_synthetic_data(&true_model, &geom, &grid)?;
            scale_shots.push((geom, obs));
        }

        println!(
            "\n  ── Scale {} / {} : f₀ = {:.0} kHz, {} iter, nt = {}, mute_r = {} ──",
            scale_idx + 1,
            scales.len(),
            f0 * 1e-3,
            n_iter,
            nt_scale,
            mute_r
        );

        current_model = fwi_scale.invert_multi_source(&scale_shots, &current_model, &grid)?;

        // Physical constraint: c ≥ c_water.
        current_model = current_model.mapv(|c| c.max(C_WATER));

        let c_now_max = current_model
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let c_now_min = current_model.iter().copied().fold(f64::INFINITY, f64::min);
        println!(
            "    Scale {} done ({:.1} s): c ∈ [{:.0}, {:.0}] m/s",
            scale_idx + 1,
            t_scale.elapsed().as_secs_f32(),
            c_now_min,
            c_now_max
        );
    }

    let reconstructed = current_model;
    println!(
        "\n  FWI completed in {:.1} s",
        t_inv.elapsed().as_secs_f32()
    );

    // Final J at 150 kHz.
    let j_final = {
        let fwi_tmp = FwiProcessor::new(FwiParameters {
            max_iterations: 1,
            frequency: F0_HZ,
            nt: nt_fine,
            dt,
            n_trace: N_RECEIVERS,
            n_depth: 1,
            step_size: STEP_SIZE,
            tolerance: 1e-12,
            regularization: RegularizationParameters {
                tikhonov_weight: 0.0,
                tv_weight: 0.0,
                smoothness_weight: 0.0,
            },
            source_mute_radius: 4,
            ..FwiParameters::default()
        });
        let mut j = 0.0_f64;
        for (geom, obs) in &shots_fine {
            let d_syn = fwi_tmp.generate_synthetic_data(&reconstructed, geom, &grid)?;
            j += d_syn
                .iter()
                .zip(obs.iter())
                .map(|(&s, &o)| (s - o).powi(2))
                .sum::<f64>()
                * 0.5
                * dt;
        }
        j
    };
    let j_reduction_pct = (1.0 - j_final / j_initial) * 100.0;

    println!("\n  Quality after inversion:");
    print_quality_report(&true_model, &reconstructed);
    println!("  Joint J (150 kHz) : {j_final:.6e} Pa²·s");
    println!("  J reduction       : {j_reduction_pct:7.1} %  (150 kHz joint L2)");

    // ── 6. Stage-2 brain tissue FWI (Guasch 2020 style) ─────────────────
    println!("\n[ 6 / 7 ]  Stage-2 brain tissue FWI (skull frozen, MNI tissue prior) …");

    let (brain_true_model, brain_reconstructed) =
        match build_brain_velocity_model(&phantom, Path::new(DEFAULT_MNI_DIR)) {
            Err(e) => {
                eprintln!("  MNI tissue maps unavailable ({e:#}); skipping Stage 2.");
                (None, None)
            }
            Ok(brain_true) => {
                // Skull mask: bone voxels frozen at CT-derived velocity.
                let skull_mask = build_skull_mask(&phantom.sound_speed);
                let n_frozen = skull_mask.iter().filter(|&&b| b).count();
                let n_free = skull_mask.len() - n_frozen;
                println!(
                "  Skull mask        : {n_frozen} frozen bone voxels, {n_free} free brain voxels"
            );

                // Velocity range of the true brain tissue model (brain only).
                let (bt_min, bt_max) = skull_mask
                    .indexed_iter()
                    .filter(|(_, &frozen)| !frozen)
                    .map(|((ix, iy, iz), _)| brain_true[[ix, iy, iz]])
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), c| {
                        (mn.min(c), mx.max(c))
                    });
                println!("  True brain c      : [{bt_min:.1}, {bt_max:.1}] m/s");

                // Stage-2 FWI processor: brain tissue frequencies + tight bounds.
                let nt_brain = {
                    // Total sim time: 3 Ricker half-periods + full domain transit.
                    let domain_transit_s = (NX as f64 * DX) / C_WATER;
                    let source_dur_s = 3.0 / F0_BRAIN_HZ;
                    ((domain_transit_s + source_dur_s) / dt).ceil() as usize
                };
                let fwi_brain = FwiProcessor::new(FwiParameters {
                    max_iterations: N_BRAIN_ITER,
                    frequency: F0_BRAIN_HZ,
                    nt: nt_brain,
                    dt,
                    n_trace: N_RECEIVERS,
                    n_depth: 1,
                    step_size: STEP_SIZE_BRAIN,
                    tolerance: 1e-14,
                    regularization: RegularizationParameters {
                        tikhonov_weight: 0.0,
                        tv_weight: 0.0,
                        smoothness_weight: 0.0,
                    },
                    source_mute_radius: 2,
                    ..FwiParameters::default()
                });

                // Generate observed gathers using the true brain tissue model.
                let mut brain_shots: Vec<(FwiGeometry, Array2<f64>)> = Vec::with_capacity(N_SHOTS);
                let t_brain_obs = Instant::now();
                for &element_index in &TRANSMIT_ELEMENT_INDICES {
                    let geom = build_shot(element_index, F0_BRAIN_HZ, nt_brain, dt);
                    match fwi_brain.generate_synthetic_data(&brain_true, &geom, &grid) {
                        Ok(obs) => brain_shots.push((geom, obs)),
                        Err(e) => {
                            eprintln!("  Brain gather failed for element {element_index}: {e:#}");
                        }
                    }
                }
                println!(
                    "  {N_SHOTS} brain gathers at {:.0} kHz ({:.1} s)",
                    F0_BRAIN_HZ * 1e-3,
                    t_brain_obs.elapsed().as_secs_f32()
                );

                if brain_shots.is_empty() {
                    eprintln!("  No brain shots succeeded; skipping Stage 2.");
                    (Some(brain_true), None)
                } else {
                    // Initial brain model: uniform water inside skull, bone frozen.
                    let mut brain_initial =
                        skull_mask.mapv(|frozen| if frozen { 0.0_f64 } else { C_WATER });
                    // Fill frozen voxels with CT skull velocity for the reference model.
                    Zip::from(&mut brain_initial)
                        .and(&skull_mask)
                        .and(&phantom.sound_speed)
                        .for_each(|c, &frozen, &ct| {
                            if frozen {
                                *c = ct;
                            }
                        });

                    println!(
                        "  Running {N_BRAIN_ITER} iterations at {:.0} kHz (nt={nt_brain}) …",
                        F0_BRAIN_HZ * 1e-3
                    );
                    let t_brain_inv = Instant::now();
                    match fwi_brain.invert_multi_source_masked(
                        &brain_shots,
                        &brain_initial,
                        &phantom.sound_speed, // skull reference (frozen voxels)
                        &skull_mask,
                        BRAIN_C_MIN,
                        BRAIN_C_MAX,
                        &grid,
                    ) {
                        Ok(brain_recon) => {
                            println!(
                                "  Brain FWI done ({:.1} s)",
                                t_brain_inv.elapsed().as_secs_f32()
                            );
                            println!("  Quality (brain voxels only, r < R_SKULL_IN):");
                            print_quality_report_brain(&brain_true, &brain_recon);
                            (Some(brain_true), Some(brain_recon))
                        }
                        Err(e) => {
                            eprintln!("  Brain FWI failed: {e:#}");
                            (Some(brain_true), None)
                        }
                    }
                }
            }
        };

    // ── 7. RTM — zero-lag cross-correlation imaging ───────────────────────
    println!("\n[ 7 / 7 ]  Reverse Time Migration (reflectivity image) …");

    // Build the receiver snapshot directly from observed shot-0 seismograms.
    // For each active receiver r at grid position (i, k) we project the RMS
    // of its observed trace onto the grid.  This avoids a redundant forward
    // simulation (which caused an OOM crash on debug binaries after the long
    // FWI run) while providing the correct spatial energy distribution for the
    // zero-lag imaging condition I(x) = ∫ p_src(x,t)·p_recv(x,T−t) dt
    // (Baysal et al., 1983).
    let (geom0, obs0) = &shots_fine[0];
    let mut recv_snapshot = Array3::<f64>::zeros((NX, NY, NZ));
    {
        let recv_mask = &geom0.sensor_mask;
        let mut recv_idx = 0usize;
        for ((i, _j, k), &active) in recv_mask.indexed_iter() {
            if active {
                if recv_idx < obs0.nrows() {
                    let trace = obs0.row(recv_idx);
                    let nt_obs = trace.len().max(1);
                    // RMS amplitude of the observed trace: scalar proxy for the
                    // receiver wavefield energy at this grid point.
                    let rms = (trace.iter().map(|&v| v * v).sum::<f64>() / nt_obs as f64).sqrt();
                    recv_snapshot[[i, 0, k]] = rms;
                }
                recv_idx += 1;
            }
        }
    }

    let rtm_settings = RtmSettings {
        imaging_condition: ImagingCondition::Normalized,
        storage_strategy: StorageStrategy::Full,
        boundary_type: SeismicBoundaryType::Absorbing,
        apply_laplacian: true,
    };
    let rtm = RtmProcessor::new(rtm_settings);
    let rtm_image = rtm
        .migrate(&recv_snapshot, &recv_snapshot, &grid)
        .unwrap_or_else(|_| Array3::<f64>::zeros((NX, NY, NZ)));
    let rtm_peak = rtm_image.iter().copied().fold(0.0_f64, f64::max);
    println!("  RTM image completed — peak amplitude: {rtm_peak:.4}");

    // ── Image output ──────────────────────────────────────────────────────
    let output_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("examples")
                .join("output")
        });

    std::fs::create_dir_all(&output_dir)
        .map_err(|e| KwaversError::InvalidInput(format!("cannot create output dir: {e}")))?;

    let abs_dir = std::fs::canonicalize(&output_dir).unwrap_or(output_dir.clone());

    let base = "brain_fwi";
    let three_plane_path = abs_dir.join(format!("{base}_three_plane.png"));
    let velocity_ppm_path = abs_dir.join(format!("{base}.ppm"));
    let rtm_path = abs_dir.join(format!("{base}_rtm.ppm"));
    let brain_prior_path = abs_dir.join(format!("{base}_ct_brain_prior.png"));
    let csv_path = abs_dir.join(format!("{base}.csv"));
    let brain_tissue_path = abs_dir.join(format!("{base}_brain_tissue.png"));

    let shot_positions = transmit_positions();
    let active_elements: Vec<(usize, usize)> = ACTIVE_TRANSDUCER_POSITIONS.to_vec();

    write_three_plane_png(
        &three_plane_path,
        &true_model,
        &reconstructed,
        C_LO,
        C_HI,
        &shot_positions,
        &active_elements,
        ct_vol.as_ref(),
    )
    .map_err(|e| KwaversError::InvalidInput(format!("PNG write failed: {e}")))?;

    write_velocity_panels(
        &velocity_ppm_path,
        &true_model,
        &initial_model,
        &reconstructed,
        &shot_positions,
        &active_elements,
    )
    .map_err(|e| KwaversError::InvalidInput(format!("velocity panel write failed: {e}")))?;

    write_brain_prior_png(
        &brain_prior_path,
        &phantom.hu,
        &shot_positions,
        &active_elements,
    )
    .map_err(|e| KwaversError::InvalidInput(format!("brain prior PNG write failed: {e}")))?;

    write_rtm_panel(&rtm_path, &rtm_image)
        .map_err(|e| KwaversError::InvalidInput(format!("RTM panel write failed: {e}")))?;

    write_velocity_csv(&csv_path, &true_model, &initial_model, &reconstructed)
        .map_err(|e| KwaversError::InvalidInput(format!("CSV write failed: {e}")))?;

    // Brain tissue PNG — written only when Stage-2 FWI succeeded.
    if let (Some(bt_true), Some(bt_recon)) = (&brain_true_model, &brain_reconstructed) {
        write_brain_tissue_png(&brain_tissue_path, bt_true, bt_recon).map_err(|e| {
            KwaversError::InvalidInput(format!("brain tissue PNG write failed: {e}"))
        })?;
    }

    println!("\n  Output directory  : {}", abs_dir.display());
    println!("\n  Wrote images and data:");
    let three_plane_desc = if ct_vol.is_some() {
        "PNG 3×2: CT coronal|axial|sagittal (top) / FWI true|reconstructed|difference (bottom)"
    } else {
        "PNG: true skull (FWI grid) | FWI reconstructed | difference — coronal x-z"
    };
    println!("    {}  ({})", three_plane_path.display(), three_plane_desc);
    println!(
        "    {}  (PPM 4-panel: true | initial | reconstructed | error)",
        velocity_ppm_path.display()
    );
    println!(
        "    {}  (PNG CT-derived brain/skull prior + transducer)",
        brain_prior_path.display()
    );
    println!(
        "    {}  (PPM RTM zero-lag cross-correlation)",
        rtm_path.display()
    );
    println!(
        "    {}  (CSV depth profile at x = NX/2)",
        csv_path.display()
    );
    if brain_reconstructed.is_some() {
        println!(
            "    {}  (PNG brain tissue: true|reconstructed|difference, [1480,1560] m/s colormap)",
            brain_tissue_path.display()
        );
    }
    if ct_vol.is_some() {
        println!(
            "  Image size        : {}×{} px (3×{PANEL} wide, 2×({PANEL}+{COLORBAR_H}) tall)",
            3 * PANEL,
            2 * (PANEL + COLORBAR_H)
        );
    } else {
        println!(
            "  Image size        : {PANEL}×{PANEL} px per panel, 3 panels, {COLORBAR_H}px colorbar"
        );
    }
    println!(
        "  Colormap          : blue (1500 m/s, water/brain) → red ({:.0} m/s, cortical bone)",
        C_HI
    );
    if ct_vol.is_some() {
        println!("  PNG layout        : 3×2 grid — top: CT coronal | axial | sagittal (bone window); bottom: FWI true | reconstructed | difference");
    } else {
        println!(
            "  PNG panels        : true skull | reconstructed | difference (x-z coronal, y=0)"
        );
    }
    println!(
        "  Markers           : white = transmitting elements | yellow = active transducer samples"
    );

    // ── Summary ───────────────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════");
    println!(
        "  Reconstructed velocity range: [{:.0}, {:.0}] m/s",
        reconstructed.iter().copied().fold(f64::INFINITY, f64::min),
        reconstructed
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    );
    println!("  True velocity range         : [{c_min:.0}, {c_max:.0}] m/s");
    println!();
    println!("  Physics verified against:");
    println!("    Aubry (2003)              — HU → c, ρ bone-volume-fraction model");
    println!("    Marsac (2017)             — skull acoustic properties + geometry");
    println!("    Guasch (2020)             — transcranial FWI methodology");
    println!("    Ricker (1953)             — source wavelet");
    println!("    Tarantola (1984)          — adjoint-state FWI gradient");
    println!("    Virieux & Operto (2009)   — FWI objective and chain rule");
    println!();
    println!("  To use a different CT:");
    println!("    set KWAVERS_SEISMIC_CT_PATH=path\\to\\ct_dicom_or_nifti");
    println!("    cargo run --example seismic_imaging_demo");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn superior_transducer_section_stays_outside_skull_and_cpml() {
        let cx = NX as f64 / 2.0;
        let cz = NZ as f64 / 2.0;
        for &(ix, iz) in &ACTIVE_TRANSDUCER_POSITIONS {
            let r = ((ix as f64 - cx).powi(2) + (iz as f64 - cz).powi(2)).sqrt();
            assert!(
                r > R_HEAD,
                "element ({ix},{iz}) must be outside skull radius {R_HEAD}, got {r}"
            );
            assert!(
                (10..54).contains(&ix) && (10..54).contains(&iz),
                "element ({ix},{iz}) must stay inside CPML-free physical domain"
            );
            assert!(
                iz < NZ / 2,
                "element ({ix},{iz}) must lie on the superior arc"
            );
        }
    }

    #[test]
    fn receiver_mask_excludes_only_transmitting_element() {
        for &source_index in &TRANSMIT_ELEMENT_INDICES {
            let mask = build_receiver_mask(source_index);
            let active = mask.iter().filter(|&&v| v).count();
            assert_eq!(active, N_RECEIVERS);
            let (sx, sz) = ACTIVE_TRANSDUCER_POSITIONS[source_index];
            assert!(!mask[[sx, 0, sz]]);
            for (idx, &(ix, iz)) in ACTIVE_TRANSDUCER_POSITIONS.iter().enumerate() {
                assert_eq!(mask[[ix, 0, iz]], idx != source_index);
            }
        }
    }

    #[test]
    fn brain_support_fills_non_bone_region_between_skull_edges() {
        let mut hu = Array3::<f64>::from_elem((NX, NY, NZ), HU_WATER);
        let row = NZ / 2;
        hu[[20, 0, row]] = 500.0;
        hu[[44, 0, row]] = 500.0;
        hu[[32, 0, row]] = -1000.0;
        hu[[20, 1, row]] = 500.0;
        hu[[44, 1, row]] = 500.0;

        let mask = brain_support_from_hu(&hu);

        assert!(mask[[32, row]]);
        assert!(!mask[[20, row]]);
        assert!(!mask[[44, row]]);
        assert!(!mask[[10, row]]);
        assert_eq!((21..44).filter(|&ix| mask[[ix, row]]).count(), 23);
    }
}
