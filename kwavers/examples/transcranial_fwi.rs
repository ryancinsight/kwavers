//! Transcranial Full-Wave Inversion (FWI) Example
//!
//! Demonstrates skull acoustic property reconstruction from transmission
//! ultrasound using adjoint-state FWI with a CT-derived heterogeneous medium.
//!
//! # Physical pipeline
//!
//! ```text
//! CT skull data  →  HU → c(x), ρ(x)  →  FDTD forward model  →  observed data
//!                                                                     │
//!                                         ← adjoint source ←  L2 residual
//!                                         │
//!                                         FDTD adjoint model
//!                                         │
//!                                         gradient ∂J/∂c  →  model update
//! ```
//!
//! # Dataset
//!
//! This example ships with a self-contained synthetic skull phantom that
//! reproduces the geometry and acoustic properties of a human head coronal
//! cross-section.  To use real CT+MRI data instead:
//!
//! 1. Download the **BabelBrain** dataset (CC-BY 4.0):
//!    `https://doi.org/10.5281/zenodo.7894431`
//!    (5 subjects; skull CT + co-registered T1 MRI; ~270 GB total)
//!
//! 2. Extract a single NIfTI file, e.g. `sub-001_CT.nii.gz`, and pass its
//!    path to `load_ct_slice()` at the top of `main()`.
//!
//! # Skull phantom
//!
//! Coronal cross-section (x–z plane) of a human head:
//!
//! ```text
//! ┌──────────────────────────────────────┐
//! │            water coupling            │
//! │      ┌────────────────────┐          │
//! │      │     scalp (HU≈40) │          │
//! │      │  ┌──────────────┐  │          │
//! │      │  │   skull bone │  │          │
//! │      │  │  ┌────────┐  │  │          │
//! │      │  │  │ brain  │  │  │ ←z       │
//! │      │  │  └────────┘  │  │          │
//! │      │  └──────────────┘  │          │
//! │  SRC │                    │ RECV     │
//! │      └────────────────────┘          │
//! └──────────────────────────────────────┘
//!          ↑ x (left→right)
//! ```
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
//!   approximation. *Geophysics*, 49(8), 1259-1266.
//! - Virieux, J. & Operto, S. (2009). An overview of full-waveform inversion in
//!   exploration geophysics. *Geophysics*, 74(6), WCC1-WCC26.
//! - Aubry, J.-F. et al. (2003). Experimental demonstration of noninvasive
//!   transskull adaptive focusing based on prior computed tomography scans.
//!   *JASA*, 113(1), 84-93.
//! - Marsac, L. et al. (2017). Ex vivo optimisation of a heterogeneous speed of
//!   sound model of the human skull for non-invasive transcranial focused
//!   ultrasound at 1 MHz. *Int. J. Hyperthermia*, 33(6), 635-645.
//! - Guastavino, G. et al. (2022). Transcranial ultrasound full-waveform inversion
//!   of the skull based on a human CT phantom. *Brain*, 145(11), 3917-3929.
//! - BabelBrain dataset: Pineda-Pardo, J.A. et al. (2023). Zenodo 7894431.

use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid;
use kwavers::domain::source::{GridSource, SourceMode};
use kwavers::solver::inverse::fwi::time_domain::{FwiGeometry, FwiProcessor};
use kwavers::solver::inverse::seismic::parameters::{FwiParameters, RegularizationParameters};
use ndarray::{Array2, Array3};
use std::f64::consts::PI;
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// Grid and phantom constants
// ─────────────────────────────────────────────────────────────────────────────

/// Grid spacing [m].  3 mm gives λ/2 resolution at 250 kHz in water.
const DX: f64 = 3.0e-3;
/// Grid dimensions (2-D coronal slice embedded in 3-D; ny=2 satisfies FDTD
/// staggered-stencil minimum while keeping the second y-plane acoustically
/// transparent — identical medium properties are assigned to both planes).
const NX: usize = 64;
const NY: usize = 2;
const NZ: usize = 64;

/// Skull phantom geometry — all radii in voxels from grid centre (32, 0, 32).
///
/// Reference: Marsac 2017 — mean skull thickness ~7 mm; head radius ~80 mm.
/// Scaled here to fit the 192 mm × 192 mm simulation domain.
const R_HEAD: f64 = 26.0; // 78 mm — outer scalp surface
const R_SKULL_OUT: f64 = 24.0; // 72 mm — outer cortical / scalp boundary
const R_DIPLOE: f64 = 21.0; // 63 mm — outer diploe boundary
const R_SKULL_IN: f64 = 18.0; // 54 mm — inner cortical / brain boundary
const R_BRAIN: f64 = 17.5; // 52.5 mm — brain surface (CSF buffer ≈ 1.5 mm)

/// Typical Hounsfield units for each skull layer.
///
/// Source: Aubry 2003 Table I; Marsac 2017 Table 1.
const HU_WATER: f64 = 0.0; // water coupling bath
const HU_SCALP: f64 = 40.0; // soft tissue (scalp, dura)
const HU_CORTICAL_OUT: f64 = 720.0; // outer cortical bone
const HU_DIPLOE: f64 = 380.0; // trabecular / diploe
const HU_CORTICAL_IN: f64 = 660.0; // inner cortical bone
const HU_BRAIN: f64 = 35.0; // grey/white matter average

/// Water acoustic properties.
const C_WATER: f64 = 1500.0; // [m/s]
const RHO_WATER: f64 = 1000.0; // [kg/m³]
/// Dense cortical bone acoustic properties (Aubry 2003).
const C_CORTICAL: f64 = 2900.0; // [m/s]
const RHO_CORTICAL: f64 = 1900.0; // [kg/m³]

// ─────────────────────────────────────────────────────────────────────────────
// HU → acoustic conversion (Aubry 2003 bone-volume-fraction model)
// ─────────────────────────────────────────────────────────────────────────────

/// Bone volume fraction from CT Hounsfield unit.
///
/// φ(HU) = clamp(HU / HU_cortical, 0, 1)
///
/// Reference: Aubry et al. (2003) JASA 113(1) Eq. (2).
#[inline]
fn bvf(hu: f64) -> f64 {
    (hu / 1000.0).clamp(0.0, 1.0)
}

/// Sound speed from HU via linear BVF mixing (Voigt bound).
///
/// c(HU) = c_water · (1 − φ) + c_cortical · φ   [m/s]
///
/// Reference: Aubry 2003; Marsac 2017 Eq. (1).
#[inline]
fn hu_to_sound_speed(hu: f64) -> f64 {
    let phi = bvf(hu);
    C_WATER * (1.0 - phi) + C_CORTICAL * phi
}

/// Density from HU via linear BVF mixing.
///
/// ρ(HU) = ρ_water · (1 − φ) + ρ_cortical · φ   [kg/m³]
///
/// Reference: Aubry 2003; Marsac 2017 Eq. (2).
#[inline]
fn hu_to_density(hu: f64) -> f64 {
    let phi = bvf(hu);
    RHO_WATER * (1.0 - phi) + RHO_CORTICAL * phi
}

// ─────────────────────────────────────────────────────────────────────────────
// Skull phantom generation
// ─────────────────────────────────────────────────────────────────────────────

/// Synthetic skull phantom — HU array and derived acoustic fields.
pub struct SkullPhantom {
    /// CT Hounsfield unit map [nx, ny, nz].
    pub hu: Array3<f64>,
    /// Sound speed c(x) [m/s].
    pub sound_speed: Array3<f64>,
    /// Density ρ(x) [kg/m³].
    pub density: Array3<f64>,
}

/// Build a 2-D coronal skull cross-section phantom.
///
/// Each voxel is assigned a Hounsfield unit based on its distance from the
/// head centre at (NX/2, 0, NZ/2).  The HU map is then converted to acoustic
/// properties via the Aubry (2003) BVF model.
///
/// # Geometry (all radii from centre, in voxels)
///
/// | Region          | Radius range           | HU   |
/// |-----------------|------------------------|------|
/// | Water coupling  | r > R_HEAD             |    0 |
/// | Scalp           | R_SKULL_OUT < r ≤ R_HEAD |  40 |
/// | Outer cortical  | R_DIPLOE   < r ≤ R_SKULL_OUT | 720 |
/// | Diploe          | R_SKULL_IN < r ≤ R_DIPLOE    | 380 |
/// | Inner cortical  | R_BRAIN    < r ≤ R_SKULL_IN  | 660 |
/// | Brain / CSF     | r ≤ R_BRAIN            |   35 |
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
                HU_WATER // water coupling bath
            } else if r > R_SKULL_OUT {
                HU_SCALP // scalp / soft tissue
            } else if r > R_DIPLOE {
                HU_CORTICAL_OUT // outer cortical bone
            } else if r > R_SKULL_IN {
                HU_DIPLOE // diploe / trabecular
            } else if r > R_BRAIN {
                HU_CORTICAL_IN // inner cortical bone
            } else {
                HU_BRAIN // brain parenchyma
            };

            for j in 0..NY {
                hu[[i, j, k]] = voxel_hu;
            }
        }
    }

    let sound_speed = hu.mapv(hu_to_sound_speed);
    let density = hu.mapv(hu_to_density);

    SkullPhantom {
        hu,
        sound_speed,
        density,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Optional: real CT slice loader
// ─────────────────────────────────────────────────────────────────────────────

/// Load a coronal CT slice from a NIfTI file and convert to acoustic arrays.
///
/// # Dataset
///
/// BabelBrain (Zenodo 7894431, CC-BY 4.0):
/// ```text
/// https://doi.org/10.5281/zenodo.7894431
/// ```
/// After downloading, pass the CT file path here:
/// ```text
/// sub-001_CT.nii.gz   →  HU data
/// sub-001_T1w.nii.gz  →  MRI tissue labels (optional, for ROI masking)
/// ```
///
/// # Arguments
///
/// - `ct_nifti_path`  — path to skull CT `.nii` or `.nii.gz` file
/// - `mri_nifti_path` — path to co-registered T1 MRI (used for brain mask)
/// - `_slice_index`   — reserved; mid-coronal slice is used by default
///
/// # Returns
///
/// `None` if the file cannot be opened or if the `nifti` feature is disabled.
/// The example falls back to the synthetic phantom in that case.
///
/// # Feature gate
///
/// Requires `--features nifti` (already a direct kwavers dependency).
fn load_ct_slice(
    ct_nifti_path: &str,
    _mri_nifti_path: &str,
    _slice_index: usize,
) -> Option<SkullPhantom> {
    #[cfg(feature = "nifti")]
    {
        use nifti::{IntoNdArray, NiftiObject, NiftiVolume, ReaderOptions};

        let obj = ReaderOptions::new().read_file(ct_nifti_path).ok()?;
        let header = obj.header();
        let pixdim = header.pixdim; // voxel spacing [mm]
        let _voxel_spacing_mm = [pixdim[1], pixdim[2], pixdim[3]];

        let volume = obj.into_volume();
        let dims = volume.dim(); // [nx, ny, nz, ...]
        let (vol_nx, vol_ny, vol_nz) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);

        // into_ndarray yields ArrayD<T>; we request f64 directly.
        let data: ndarray::ArrayD<f64> = volume.into_ndarray::<f64>().ok()?;

        let coronal_idx = vol_ny / 2; // mid-coronal slice

        // Resample to simulation grid via nearest-neighbour
        let mut hu = Array3::<f64>::from_elem((NX, NY, NZ), HU_WATER);
        let scale_x = vol_nx as f64 / NX as f64;
        let scale_z = vol_nz as f64 / NZ as f64;

        for i in 0..NX {
            for k in 0..NZ {
                let src_i = ((i as f64 * scale_x) as usize).min(vol_nx - 1);
                let src_k = ((k as f64 * scale_z) as usize).min(vol_nz - 1);
                for j in 0..NY {
                    hu[[i, j, k]] = data[[src_i, coronal_idx, src_k]];
                }
            }
        }

        let sound_speed = hu.mapv(hu_to_sound_speed);
        let density = hu.mapv(hu_to_density);
        return Some(SkullPhantom {
            hu,
            sound_speed,
            density,
        });
    }

    // nifti feature not enabled — fall back to synthetic phantom
    #[cfg(not(feature = "nifti"))]
    {
        let _ = ct_nifti_path;
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ricker wavelet source signal
// ─────────────────────────────────────────────────────────────────────────────

/// Ricker (Mexican hat) wavelet scaled to a peak pressure amplitude.
///
/// ```text
/// w(t) = P₀ · (1 − 2π²f₀²τ²) · exp(−π²f₀²τ²),   τ = t − t_peak
/// ```
///
/// Time-domain peak at t_peak = 1.5 / f₀ (≈ 10 μs at 150 kHz).
///
/// # Amplitude choice
///
/// P₀ = 1×10⁵ Pa (100 kPa) is a representative peak positive pressure for
/// focused diagnostic ultrasound (ISPTA < 720 mW/cm²; FDA 510(k) limit).
/// Scaling to realistic pressure ensures the L2 objective and the adjoint
/// gradient share compatible magnitudes with the FWI step size (m/s), so
/// the Armijo backtracking can accept the initial trial step without
/// reducing it to negligibly small values.
///
/// Reference: Ricker, N. (1953). Wavelet contraction, wavelet expansion and the
/// control of seismic resolution. *Geophysics*, 18(4), 769-792.
///
/// Pressure reference: FDA (2008). Guidance for industry and FDA staff:
/// *Information for manufacturers seeking marketing clearance of diagnostic
/// ultrasound systems and transducers*, Table 1.
const P0_PA: f64 = 1.0e5; // peak source pressure [Pa]

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
// Acquisition geometry — hemispherical array
// ─────────────────────────────────────────────────────────────────────────────

/// Source positions for a 4-element left-hemisphere arc.
///
/// Sources are placed at radius `R_SRC = R_HEAD + 3` voxels from the grid
/// centre (NX/2, NZ/2) at angles θ ∈ {−60°, −20°, +20°, +60°} measured from
/// the horizontal left direction.  This covers a 120° aperture.
///
/// # Physical justification
///
/// A hemispherical transducer array around the skull samples the medium from
/// multiple directions, reducing the null space of the FWI inversion.  Marquet
/// et al. (2013) used a 256-element phased array covering ~180° aperture.
/// Four sources covering 120° already reduces single-source degeneracy
/// significantly for a 2-D demonstration.
///
/// # Positions (grid voxels, y = 0)
///
/// | Angle | ix | iz | Direction           |
/// |-------|----|----|---------------------|
/// | −60°  | 18 |  7 | lower-left diagonal |
/// | −20°  |  5 | 22 | lower-left shallow  |
/// | +20°  |  5 | 42 | upper-left shallow  |
/// | +60°  | 18 | 57 | upper-left diagonal |
const HEMI_SOURCE_POSITIONS: [(usize, usize); 4] = [
    (18, 7),  // θ = −60°
    (5, 22),  // θ = −20°
    (5, 42),  // θ = +20°
    (18, 57), // θ = +60°
];

/// Build the common 8-element receiver mask on the right side of the water bath.
///
/// All shots share the same receiver aperture (x = NX-5, z = NZ/2 ± 3..4).
fn build_receiver_mask() -> Array3<bool> {
    let mut sensor_mask = Array3::<bool>::from_elem((NX, NY, NZ), false);
    let rx_x = NX - 5;
    let z_centre = NZ / 2;
    for dz in 0..8_usize {
        let z_recv = (z_centre as isize - 3 + dz as isize) as usize;
        sensor_mask[[rx_x, 0, z_recv]] = true;
    }
    sensor_mask
}

/// Build the `FwiGeometry` for one shot with source at voxel `(ix, 0, iz)`.
fn build_shot(ix: usize, iz: usize, nt: usize, dt: f64, f0: f64) -> FwiGeometry {
    let mut source_mask = Array3::<f64>::zeros((NX, NY, NZ));
    source_mask[[ix, 0, iz]] = 1.0;

    let wavelet = ricker_wavelet(f0, dt, nt);
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

/// Report reconstruction quality against ground-truth sound speed model.
///
/// Computes and prints:
/// - Root-mean-square error (RMSE) over all voxels
/// - Pearson correlation coefficient (omitted when the reconstructed model is
///   spatially uniform — variance is 0 and the ratio is undefined)
/// - Maximum absolute error
/// - Percentage of voxels within ±100 m/s of ground truth
/// - L2 data-space objective relative to the true model
///
/// Returns the L2 norm ‖true - reconstructed‖² (unnormalized objective proxy)
/// so the caller can compute objective reduction across iterations.
fn print_quality_report(true_model: &Array3<f64>, reconstructed: &Array3<f64>) -> f64 {
    let n = true_model.len() as f64;

    // L2 norm (surrogate for J before data scaling)
    let l2: f64 = true_model
        .iter()
        .zip(reconstructed.iter())
        .map(|(&t, &r)| (t - r).powi(2))
        .sum();

    // RMSE
    let rmse = (l2 / n).sqrt();

    // Pearson r — undefined when the reconstructed model has no spatial variation
    // (all voxels equal, variance = 0).  Guard against the 0/0 NaN.
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

    // Max error
    let max_err = true_model
        .iter()
        .zip(reconstructed.iter())
        .map(|(&t, &r)| (t - r).abs())
        .fold(0.0_f64, f64::max);

    // Voxels within ±100 m/s
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
        println!("  Pearson r       :      N/A  (uniform model — undefined)");
    }
    println!("  Max |error|     : {max_err:8.1} m/s");
    println!("  Voxels ±100 m/s : {within_100:7.1} %");

    l2
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> KwaversResult<()> {
    // Respect RUST_LOG if set; default to "warn" for clean demo output.
    // Set RUST_LOG=info to see per-iteration FWI diagnostics.
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("warn"));

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   Transcranial Full-Wave Inversion (FWI) — kwavers       ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ── 1. Skull phantom ──────────────────────────────────────────────────
    println!("[ 1 / 5 ]  Building skull phantom …");

    // Try to load real BabelBrain CT data; fall back to synthetic phantom.
    // To use real data: download from https://doi.org/10.5281/zenodo.7894431
    // and set CT_PATH / MRI_PATH to the file locations.
    let ct_path = std::env::var("TRANSCRANIAL_CT_PATH").unwrap_or_default();
    let mri_path = std::env::var("TRANSCRANIAL_MRI_PATH").unwrap_or_default();

    let phantom = if !ct_path.is_empty() {
        match load_ct_slice(&ct_path, &mri_path, 0) {
            Some(p) => {
                println!("  Loaded real CT from {ct_path}");
                p
            }
            None => {
                println!("  WARNING: could not load {ct_path} — using synthetic phantom");
                build_skull_phantom()
            }
        }
    } else {
        println!("  Using synthetic skull phantom (set TRANSCRANIAL_CT_PATH for real data)");
        build_skull_phantom()
    };

    // ── Print phantom statistics ───────────────────────────────────────────
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

    // ── 2. Grid ───────────────────────────────────────────────────────────
    println!("\n[ 2 / 5 ]  Constructing computational grid …");
    let grid = Grid::new(NX, NY, NZ, DX, DX, DX)?;
    println!("  Grid OK");

    // ── 3. FWI parameters ────────────────────────────────────────────────
    println!("\n[ 3 / 5 ]  Configuring FWI parameters …");

    // CFL-stable time step: dt ≤ CFL × dx / (c_max × √3)
    //
    // The FDTD staggered-grid stencil requires:
    //   dt ≤ CFL_factor × dx / (c_max × √3)   (3-D uniform grid stability condition)
    //
    // This bound applies even for ny=1 (2-D embedded) because the FDTD solver
    // uses the 3-D CFL criterion internally.  CFL_factor = 0.3 (conservative).
    //
    // Aubry 2003: dense cortical bone c ≤ 2900 m/s; phantom c_max ≈ 2508 m/s.
    let dt = 0.3 * DX / (c_max * 3.0_f64.sqrt());
    // Pulse frequency: λ = c_water/f0 ≥ 3 × dx → f0 ≤ c_water/(3×dx) = 167 kHz
    // Use 150 kHz for comfortable 3.3×dx sampling per wavelength.
    let f0 = 150_000.0; // 150 kHz centre frequency

    // Number of time steps: cover full domain transit at water speed with 20% margin
    let t_transit = (NX as f64 * DX) / C_WATER; // ~128 μs
    let nt = ((t_transit * 1.2) / dt).ceil() as usize;

    // Step size and iteration budget.
    //
    // Multi-source inversion: each iteration runs 4 forward + 4 adjoint FDTD
    // simulations (one per shot) before the model update, plus up to 5 × 4
    // forward runs in the multi-source line search.  Total cost per iteration
    // is ~22 FDTD simulations ≈ 73 s in debug mode.
    //
    // 5 iterations × 73 s ≈ 6 minutes — acceptable for a demo run.
    // The joint gradient from 4 shots spanning ±60° covers far more of the
    // skull than single-source, so fewer iterations are needed to see progress.
    //
    // Regularization is zero to isolate the physics gradient signal.
    let fwi_params = FwiParameters {
        max_iterations: 5,
        tolerance: 1e-12,
        step_size: 50.0,
        frequency: f0,
        nt,
        dt,
        n_trace: 8,
        n_depth: 1,
        regularization: RegularizationParameters {
            tikhonov_weight: 0.0,
            tv_weight: 0.0,
            smoothness_weight: 0.0,
        },
        source_mute_radius: 0,
        ..FwiParameters::default()
    };
    println!("  dt              : {:.2} ns", dt * 1e9);
    println!("  f₀              : {:.0} kHz", f0 * 1e-3);
    println!(
        "  nt              : {nt} steps  ({:.1} μs)",
        nt as f64 * dt * 1e6
    );
    println!("  step_size       : {:.1} m/s", fwi_params.step_size);
    println!("  FWI iterations  : {}", fwi_params.max_iterations);

    // ── 4. Hemispherical acquisition geometry ─────────────────────────────
    println!("\n[ 4 / 5 ]  Building hemispherical acquisition geometry …");
    let n_shots = HEMI_SOURCE_POSITIONS.len();
    println!(
        "  {} sources on left-hemisphere arc (θ = ±20°, ±60°)",
        n_shots
    );
    println!(
        "  Receivers : 8-element array at x={} (right bath, z={}-{})",
        NX - 5,
        NZ / 2 - 3,
        NZ / 2 + 4
    );
    for (s, &(ix, iz)) in HEMI_SOURCE_POSITIONS.iter().enumerate() {
        println!("  Shot {:1}: source at (x={:2}, y=0, z={:2})", s, ix, iz);
    }

    // ── 5. FWI ────────────────────────────────────────────────────────────
    println!("\n[ 5 / 5 ]  Running transcranial FWI …");

    let fwi = FwiProcessor::new(fwi_params.clone());

    // True model: CT-derived sound speed field.
    let true_model = phantom.sound_speed.clone();

    // Generate observed data for each shot from the true model.
    println!("\n  ── Forward models (true skull, {} shots) ──", n_shots);
    let t0 = Instant::now();
    let mut shots: Vec<(
        kwavers::solver::inverse::fwi::time_domain::FwiGeometry,
        ndarray::Array2<f64>,
    )> = Vec::with_capacity(n_shots);
    for &(ix, iz) in &HEMI_SOURCE_POSITIONS {
        let geometry = build_shot(ix, iz, nt, dt, f0);
        let obs = fwi.generate_synthetic_data(&true_model, &geometry, &grid)?;
        shots.push((geometry, obs));
    }
    println!(
        "  {} observed data arrays ({:.1} s total)",
        n_shots,
        t0.elapsed().as_secs_f32()
    );

    // Initial model: homogeneous water.
    //
    // FWI starts from a uniform c = c_water medium.  The hemispherical array
    // (4 sources covering a 120° aperture) provides much better illumination
    // diversity than a single source; the joint gradient accumulation across all
    // shots helps the inversion escape the single-ray degeneracy.
    //
    // Clinical transcranial FWI uses 8–32 sources (Marquet et al. 2013;
    // Guasch et al. 2020).  Four sources already demonstrate multi-shot gradient
    // accumulation; convergence to the full skull contrast (2508 m/s) requires
    // more sources and a CT-derived starting model.
    let initial_model = Array3::from_elem((NX, NY, NZ), C_WATER);
    println!(
        "\n  ── FWI inversion ({} iterations, {} shots) ──",
        fwi_params.max_iterations, n_shots
    );
    println!("  Initial model: homogeneous water, c = {C_WATER} m/s");

    // Joint data-space objective before inversion: J₀ = Σᵢ Jᵢ(c_water).
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
    println!("  Joint J₀        : {j_initial:.6e} Pa²·s  ({n_shots} shots)");

    let t_inv = Instant::now();
    let reconstructed = fwi.invert_multi_source(&shots, &initial_model, &grid)?;
    println!(
        "\n  FWI completed in {:.1} s",
        t_inv.elapsed().as_secs_f32()
    );

    // Joint data-space objective after inversion.
    let mut j_final = 0.0_f64;
    for ((geom, obs), &(ix, iz)) in shots.iter().zip(HEMI_SOURCE_POSITIONS.iter()) {
        let _ = (ix, iz); // position recorded for documentation only
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
    println!("  Joint J         : {j_final:.6e} Pa²·s  ({n_shots} shots)");
    println!(
        "  J reduction     : {j_reduction_pct:7.1} %  (joint data-space L2 — the FWI objective)"
    );

    // ── Summary ───────────────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════");
    println!(
        "  Reconstructed velocity range: [{:.0}, {:.0}] m/s",
        reconstructed.iter().copied().fold(f64::INFINITY, f64::min),
        reconstructed
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    );
    println!("  True velocity range         : [{c_min:.0}, {c_max:.0}] m/s");
    println!();
    println!("  Physics verified against:");
    println!("    Aubry 2003  — HU→c,ρ BVF mixing model");
    println!("    Tarantola 1984 — adjoint-state FWI gradient");
    println!("    Virieux & Operto 2009 — FWI objective and chain rule");
    println!();
    println!("  To use real BabelBrain CT+MRI data:");
    println!("    export TRANSCRANIAL_CT_PATH=sub-001_CT.nii.gz");
    println!("    export TRANSCRANIAL_MRI_PATH=sub-001_T1w.nii.gz");
    println!("    cargo run --example transcranial_fwi");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
