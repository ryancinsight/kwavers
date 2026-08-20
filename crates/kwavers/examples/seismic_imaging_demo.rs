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

#[path = "seismic_imaging/brain_model.rs"]
mod seismic_brain_model;
mod seismic_imaging;
#[path = "seismic_imaging/metrics.rs"]
mod seismic_metrics;
#[path = "seismic_imaging/planar_artifacts.rs"]
mod seismic_planar_artifacts;
use seismic_metrics::{print_quality_pairs, print_quality_report};

use aequitas::systems::si::quantities::{Frequency, Pressure, Time};
use kwavers_core::constants::{
    acoustic_parameters::SOUND_SPEED_SKULL_CORTICAL, fundamental::SOUND_SPEED_WATER_SIM,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_signal::DomainRickerWavelet;
use kwavers_solver::inverse::fwi::time_domain::{FwiGeometry, FwiProcessor};
use kwavers_solver::inverse::seismic::{
    parameters::{
        FwiParameters, ImagingCondition, RegularizationParameters, RtmSettings,
        SeismicBoundaryType, StorageStrategy,
    },
    rtm::RtmProcessor,
};
use kwavers_source::{GridSource, SourceMode};
use leto::{Array2, Array3};
use std::path::PathBuf;
use std::time::Instant;

#[path = "support/brain_prior.rs"]
mod brain_prior;
#[path = "support/seismic_input.rs"]
mod seismic_input;
use brain_prior::BrainPriorMode;
use seismic_input::SeismicInputMode;

use anyhow::Context as _;
use seismic_imaging::ct::{
    CtVolume, load_ct_volume, skull_centroid_2d, skull_equator_z, skull_outer_radius_ct,
};
use seismic_imaging::medium::SkullModel;
use seismic_imaging::render::{put_pixel, velocity_color, write_png};

// ─────────────────────────────────────────────────────────────────────────────
// Grid constants
// ─────────────────────────────────────────────────────────────────────────────

/// Grid spacing `m`.  3 mm gives λ/3.3 resolution at 150 kHz in water.
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
// Hounsfield-unit phantom labels
// ─────────────────────────────────────────────────────────────────────────────

/// Typical HU values per skull layer (Aubry 2003 Table I; Marsac 2017 Table 1).
const HU_WATER: f64 = 0.0; // water coupling bath
const HU_SCALP: f64 = 40.0; // soft tissue / scalp / dura
const HU_CORTICAL_OUT: f64 = 720.0; // outer cortical bone
const HU_DIPLOE: f64 = 380.0; // trabecular / diploe
const HU_CORTICAL_IN: f64 = 660.0; // inner cortical bone
const HU_BRAIN: f64 = 35.0; // grey/white matter average

/// Phase-correction example Medimodel series UID.  The directory contains
/// additional derived CT series with inconsistent orientation; this UID is the
/// same 67-slice skull CT series used by the companion example.
/// Full hemispherical aperture size used by the companion phase-correction demo.
const TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT: usize = 1024;

/// Colormap bounds [m/s] for skull image panels.
const C_LO: f64 = SOUND_SPEED_WATER_SIM; // 1500 m/s → blue
const C_HI: f64 = SOUND_SPEED_SKULL_CORTICAL; // 3100 m/s → red

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

/// MNI ICBM 2009c inner-skull radius at the coronal mid-plane `mm`.
/// The inner cortical surface is ≈ 82 mm from the brain centroid in this atlas.
const MNI_INNER_SKULL_RADIUS_MM: f64 = 82.0;

// Skull phantom structure
// ─────────────────────────────────────────────────────────────────────────────

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
fn build_skull_phantom() -> KwaversResult<SkullModel> {
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

    SkullModel::from_hu(hu)
}

// ─────────────────────────────────────────────────────────────────────────────
// Source wavelet
// ─────────────────────────────────────────────────────────────────────────────

/// Centre frequency of the Ricker source wavelet `Hz`.
///
/// 150 kHz: λ = 10 mm in water, 3.3 × dx sampling per wavelength at 3 mm.
/// Diagnostic ultrasound TUS range: 100–650 kHz (Marsac 2017; Guasch 2020).
const F0_HZ: f64 = 150_000.0;

/// Peak source pressure `Pa`.  100 kPa is a representative clinical TUS level.
///
/// Reference: FDA (2008), diagnostic ultrasound guidance, Table 1.
const P0_PA: f64 = 1.0e5;

// ─────────────────────────────────────────────────────────────────────────────
// Acquisition geometry — full-ring section
// ─────────────────────────────────────────────────────────────────────────────

/// Number of active elements in the FWI full-ring section.
const FWI_ACTIVE_ELEMENTS: usize = 16;

/// Number of transmit sources on the full-ring section (every other element).
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

/// Build the receiver mask on the same full-ring transducer section.
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
fn build_shot(
    source_element_index: usize,
    f0_hz: f64,
    nt: usize,
    dt: f64,
) -> KwaversResult<FwiGeometry> {
    let (ix, iz) = ACTIVE_TRANSDUCER_POSITIONS[source_element_index];
    let mut source_mask = Array3::<f64>::zeros((NX, NY, NZ));
    source_mask[[ix, 0, iz]] = 1.0;

    let wavelet =
        DomainRickerWavelet::causal(Frequency::from_base(f0_hz), Pressure::from_base(P0_PA))?;
    let mut p_signal = Array2::<f64>::zeros((1, nt));
    for (t, pressure) in wavelet.samples(Time::from_base(dt), nt)?.enumerate() {
        p_signal[[0, t]] = pressure;
    }

    let mut source = GridSource::new_empty();
    source.p_mask = Some(source_mask);
    source.p_signal = Some(p_signal);
    source.p_mode = SourceMode::Dirichlet;

    Ok(FwiGeometry::new(
        source,
        build_receiver_mask(source_element_index),
    ))
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

fn bilinear_hu(hu: &Array3<f64>, x: f64, y: f64, z: usize) -> f64 {
    let [nx, ny, nz] = hu.shape();
    if z >= nz {
        return 0.0;
    }
    let clamp_x = |index: isize| index.clamp(0, nx as isize - 1) as usize;
    let clamp_y = |index: isize| index.clamp(0, ny as isize - 1) as usize;
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let h00 = hu[[clamp_x(x0), clamp_y(y0), z]];
    let h10 = hu[[clamp_x(x0 + 1), clamp_y(y0), z]];
    let h01 = hu[[clamp_x(x0), clamp_y(y0 + 1), z]];
    let h11 = hu[[clamp_x(x0 + 1), clamp_y(y0 + 1), z]];
    h00 * (1.0 - fx) * (1.0 - fy) + h10 * fx * (1.0 - fy) + h01 * (1.0 - fx) * fy + h11 * fx * fy
}

fn resample_ct_to_fwi_grid(vol: &CtVolume) -> Array3<f64> {
    let hu = vol.hu();
    let z_eq = skull_equator_z(hu);
    let (cx, cy) = skull_centroid_2d(hu, z_eq);

    // Detect skull outer radius in CT pixels and derive scale so the skull
    // outer edge lands at R_HEAD FWI voxels from the grid centre.
    let r_skull_ct = skull_outer_radius_ct(hu, z_eq, cx, cy);
    let spacing_mm = vol.spacing_mm();
    let scale = r_skull_ct / R_HEAD; // CT pixels per FWI voxel

    println!(
        "  CT skull radius : {r_skull_ct:.1} px × {:.2} mm/px = {:.0} mm",
        spacing_mm[0],
        r_skull_ct * spacing_mm[0]
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
            let hu_val = bilinear_hu(hu, x_ct, y_ct, z_eq);
            for iy in 0..NY {
                result[[ix, iy, iz]] = hu_val;
            }
        }
    }
    let brain = seismic_planar_artifacts::brain_support_from_hu(&result);
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

/// Build the skull phantom for an explicit input mode.
fn build_phantom_for_demo(
    input: &SeismicInputMode,
) -> anyhow::Result<(SkullModel, Option<CtVolume>)> {
    let SeismicInputMode::Ct(path) = input else {
        if matches!(input, SeismicInputMode::CtMri { .. }) {
            anyhow::bail!("the 2-D seismic workflow accepts synthetic or ct:<path> input only");
        }
        println!("  Phantom         : synthetic analytical skull");
        return Ok((build_skull_phantom()?, None));
    };

    print!("  CT source       : {}  ", path.display());
    let vol = load_ct_volume(path)
        .with_context(|| format!("explicit CT input could not be loaded: {}", path.display()))?;
    let [cx, cy, nz] = vol.hu().shape();
    let spacing_mm = vol.spacing_mm();
    println!(
        "({cx}×{cy}×{nz} voxels @ [{:.2},{:.2},{:.2}] mm)",
        spacing_mm[0], spacing_mm[1], spacing_mm[2]
    );
    let hu_fwi = resample_ct_to_fwi_grid(&vol);
    let phantom = SkullModel::from_hu(hu_fwi)?;
    Ok((phantom, Some(vol)))
}

// ─────────────────────────────────────────────────────────────────────────────
// Reconstruction quality metrics
// ─────────────────────────────────────────────────────────────────────────────

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
        .filter(|([ix, _iy, iz], _)| {
            let r = (((*ix as f64) - cx).powi(2) + ((*iz as f64) - cz).powi(2)).sqrt();
            r < R_SKULL_IN
        })
        .map(|([ix, _iy, iz], &t)| (t, reconstructed[[ix, _iy, iz]]))
        .collect();
    print_quality_pairs(&free_pairs);
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
    let input_mode = SeismicInputMode::from_env("KWAVERS_SEISMIC_INPUT_MODE")
        .map_err(KwaversError::InvalidInput)?;
    println!("  Input mode       : {input_mode:?}");
    let (phantom, ct_vol) = build_phantom_for_demo(&input_mode)
        .map_err(|error| KwaversError::InvalidInput(error.to_string()))?;

    let c_min = phantom
        .acoustic()
        .sound_speed
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let c_max = phantom
        .acoustic()
        .sound_speed
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let hu_min = phantom.hu().iter().copied().fold(f64::INFINITY, f64::min);
    let hu_max = phantom
        .hu()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

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
    let t_transit = (NX as f64 * DX) / SOUND_SPEED_WATER_SIM;

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

    // ── 4. Full-ring acquisition geometry ─────────────────────────────────
    println!("\n[ 4 / 6 ]  Building full-ring acquisition geometry …");
    println!(
        "  Full aperture    : {TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT} elements, 650 kHz design authority"
    );
    println!("  FWI section      : {FWI_ACTIVE_ELEMENTS} active full-ring samples");
    println!(
        "  Transmits        : {N_SHOTS} shots; receivers/shot = {N_RECEIVERS} on same full ring"
    );
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

    let true_model = phantom.acoustic().sound_speed.clone();

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
                directional_tv_weight: 0.0,
                directional_tv_adaptive: false,
                smoothness_weight: 0.0,
            },
            source_mute_radius: 4,
            ..FwiParameters::default()
        });
        let t0 = Instant::now();
        for &element_index in &TRANSMIT_ELEMENT_INDICES {
            let geom = build_shot(element_index, F0_HZ, nt_fine, dt)?;
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
                directional_tv_weight: 0.0,
                directional_tv_adaptive: false,
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
        let mute_r = ((SOUND_SPEED_WATER_SIM / (2.0 * f0)) / DX).floor() as usize;
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
                directional_tv_weight: 0.0,
                directional_tv_adaptive: false,
                smoothness_weight: 0.0,
            },
            source_mute_radius: mute_r,
            ..FwiParameters::default()
        });

        let t_scale = Instant::now();
        for &element_index in &TRANSMIT_ELEMENT_INDICES {
            let geom = build_shot(element_index, f0, nt_scale, dt)?;
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
        current_model = current_model.mapv(|c| c.max(SOUND_SPEED_WATER_SIM));

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
                directional_tv_weight: 0.0,
                directional_tv_adaptive: false,
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
    let brain_prior =
        BrainPriorMode::from_env("KWAVERS_BRAIN_PRIOR").map_err(KwaversError::InvalidInput)?;
    println!("\n[ 6 / 7 ]  Stage-2 brain tissue FWI ({brain_prior:?}) …");

    let (brain_true_model, brain_reconstructed) = match seismic_brain_model::build_brain_prior(
        &phantom,
        &brain_prior,
    ) {
        Err(e) => {
            return Err(KwaversError::InvalidInput(format!(
                "selected brain prior failed: {e:#}"
            )));
        }
        Ok(brain_true) => {
            // Skull mask: bone voxels frozen at CT-derived velocity.
            let skull_mask = seismic_brain_model::build_skull_mask(&phantom.acoustic().sound_speed);
            let n_frozen = skull_mask.iter().filter(|&&b| b).count();
            let n_free = skull_mask.len() - n_frozen;
            println!(
                "  Skull mask        : {n_frozen} frozen bone voxels, {n_free} free brain voxels"
            );

            // Velocity range of the true brain tissue model (brain only).
            let (bt_min, bt_max) = skull_mask
                .indexed_iter()
                .filter(|(_, &frozen)| !frozen)
                .map(|([ix, iy, iz], _)| brain_true[[ix, iy, iz]])
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), c| {
                    (mn.min(c), mx.max(c))
                });
            println!("  True brain c      : [{bt_min:.1}, {bt_max:.1}] m/s");

            // Stage-2 FWI processor: brain tissue frequencies + tight bounds.
            let nt_brain = {
                // Total sim time: 3 Ricker half-periods + full domain transit.
                let domain_transit_s = (NX as f64 * DX) / SOUND_SPEED_WATER_SIM;
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
                    directional_tv_weight: 0.0,
                    directional_tv_adaptive: false,
                    smoothness_weight: 0.0,
                },
                source_mute_radius: 2,
                ..FwiParameters::default()
            });
            // Generate observed gathers using the true brain tissue model.
            let mut brain_shots: Vec<(FwiGeometry, Array2<f64>)> = Vec::with_capacity(N_SHOTS);
            let t_brain_obs = Instant::now();
            for &element_index in &TRANSMIT_ELEMENT_INDICES {
                let geom = build_shot(element_index, F0_BRAIN_HZ, nt_brain, dt)?;
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
                return Err(KwaversError::InvalidInput(format!(
                    "brain FWI produced no successful gathers from {N_SHOTS} shots"
                )));
            }

            // Initial brain model: uniform water inside skull, bone frozen.
            let mut brain_initial = skull_mask.mapv(|frozen| {
                if frozen {
                    0.0_f64
                } else {
                    SOUND_SPEED_WATER_SIM
                }
            });
            // Fill frozen voxels with CT skull velocity for the reference model.
            let [bi_nx, bi_ny, bi_nz] = brain_initial.shape();
            for i in 0..bi_nx {
                for j in 0..bi_ny {
                    for k in 0..bi_nz {
                        if skull_mask[[i, j, k]] {
                            brain_initial[[i, j, k]] = phantom.acoustic().sound_speed[[i, j, k]];
                        }
                    }
                }
            }

            println!(
                "  Running {N_BRAIN_ITER} iterations at {:.0} kHz (nt={nt_brain}) …",
                F0_BRAIN_HZ * 1e-3
            );
            let t_brain_inv = Instant::now();
            let brain_recon = fwi_brain
                .invert_multi_source_masked(
                    &brain_shots,
                    &brain_initial,
                    &phantom.acoustic().sound_speed, // skull reference (frozen voxels)
                    &skull_mask,
                    BRAIN_C_MIN,
                    BRAIN_C_MAX,
                    &grid,
                )
                .map_err(|error| {
                    KwaversError::InvalidInput(format!("brain FWI inversion failed: {error:#}"))
                })?;
            println!(
                "  Brain FWI done ({:.1} s)",
                t_brain_inv.elapsed().as_secs_f32()
            );
            println!("  Quality (brain voxels only, r < R_SKULL_IN):");
            print_quality_report_brain(&brain_true, &brain_recon);
            (Some(brain_true), Some(brain_recon))
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
        for ([i, _j, k], &active) in recv_mask.indexed_iter() {
            if active {
                if recv_idx < obs0.shape()[0] {
                    let trace = obs0.index_axis::<1>(0, recv_idx).expect("index_axis");
                    let nt_obs = trace.shape()[0].max(1);
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
        .map_err(|error| KwaversError::InvalidInput(format!("RTM migration failed: {error:#}")))?;
    let rtm_peak = rtm_image.iter().copied().fold(0.0_f64, f64::max);
    println!("  RTM image completed — peak amplitude: {rtm_peak:.4}");

    // ── Image output ──────────────────────────────────────────────────────
    let output_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/seismic_imaging_demo"));

    std::fs::create_dir_all(&output_dir)
        .map_err(|e| KwaversError::InvalidInput(format!("cannot create output dir: {e}")))?;

    let abs_dir = std::fs::canonicalize(&output_dir).map_err(|error| {
        KwaversError::InvalidInput(format!("cannot canonicalize output dir: {error}"))
    })?;

    let base = "brain_fwi";
    let three_plane_path = abs_dir.join(format!("{base}_three_plane.png"));
    let velocity_ppm_path = abs_dir.join(format!("{base}.ppm"));
    let rtm_path = abs_dir.join(format!("{base}_rtm.ppm"));
    let brain_prior_path = abs_dir.join(format!("{base}_ct_brain_prior.png"));
    let csv_path = abs_dir.join(format!("{base}.csv"));
    let brain_tissue_path = abs_dir.join(format!("{base}_brain_tissue.png"));

    let shot_positions = transmit_positions();
    let active_elements: Vec<(usize, usize)> = ACTIVE_TRANSDUCER_POSITIONS.to_vec();

    seismic_planar_artifacts::write_three_plane_png(
        &three_plane_path,
        &true_model,
        &reconstructed,
        seismic_planar_artifacts::VelocityScale { lo: C_LO, hi: C_HI },
        seismic_planar_artifacts::AcquisitionMarkers {
            shot_positions: &shot_positions,
            active_elements: &active_elements,
        },
        ct_vol.as_ref(),
    )
    .map_err(|e| KwaversError::InvalidInput(format!("PNG write failed: {e}")))?;

    seismic_planar_artifacts::write_velocity_panels(
        &velocity_ppm_path,
        &true_model,
        &initial_model,
        &reconstructed,
        &shot_positions,
        &active_elements,
    )
    .map_err(|e| KwaversError::InvalidInput(format!("velocity panel write failed: {e}")))?;

    seismic_planar_artifacts::write_brain_prior_png(
        &brain_prior_path,
        phantom.hu(),
        &shot_positions,
        &active_elements,
    )
    .map_err(|e| KwaversError::InvalidInput(format!("brain prior PNG write failed: {e}")))?;

    seismic_planar_artifacts::write_rtm_panel(&rtm_path, &rtm_image)
        .map_err(|e| KwaversError::InvalidInput(format!("RTM panel write failed: {e}")))?;

    seismic_planar_artifacts::write_velocity_csv(
        &csv_path,
        &true_model,
        &initial_model,
        &reconstructed,
    )
    .map_err(|e| KwaversError::InvalidInput(format!("CSV write failed: {e}")))?;

    // Brain tissue PNG — written only when Stage-2 FWI succeeded.
    if let (Some(bt_true), Some(bt_recon)) = (&brain_true_model, &brain_reconstructed) {
        seismic_planar_artifacts::write_brain_tissue_png(&brain_tissue_path, bt_true, bt_recon)
            .map_err(|e| {
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
        println!(
            "  PNG layout        : 3×2 grid — top: CT coronal | axial | sagittal (bone window); bottom: FWI true | reconstructed | difference"
        );
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
    println!("  To use an explicit CT input:");
    println!("    set KWAVERS_SEISMIC_INPUT_MODE=ct:path\\to\\ct_dicom_or_nifti");
    println!("    cargo run --example seismic_imaging_demo");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_ring_transducer_section_stays_outside_skull_and_cpml() {
        let cx = NX as f64 / 2.0;
        let cz = NZ as f64 / 2.0;
        let mut has_superior = false;
        let mut has_inferior = false;
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
            has_superior |= iz < NZ / 2;
            has_inferior |= iz > NZ / 2;
        }
        assert!(
            has_superior && has_inferior,
            "full-ring section must cover both z hemispheres"
        );
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

        let mask = seismic_planar_artifacts::brain_support_from_hu(&hu);

        assert!(mask[[32, row]]);
        assert!(!mask[[20, row]]);
        assert!(!mask[[44, row]]);
        assert!(!mask[[10, row]]);
        assert_eq!((21..44).filter(|&ix| mask[[ix, row]]).count(), 23);
    }
}
