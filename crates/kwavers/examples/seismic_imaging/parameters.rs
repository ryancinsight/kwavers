//! Grid, phantom, FWI, and rendering parameters for the seismic imaging demo.

use kwavers_core::constants::{
    acoustic_parameters::SOUND_SPEED_SKULL_CORTICAL, fundamental::SOUND_SPEED_WATER_SIM,
};

// Grid constants

/// Grid spacing `m`.  3 mm gives λ/3.3 resolution at 150 kHz in water.
///
/// Reference: Marsac 2017 — mean skull thickness ≈ 7 mm → at least 2 voxels
/// through bone at 3 mm spacing.
pub(crate) const DX: f64 = 3.0e-3;

/// Grid dimensions.  NY = 2 satisfies the FDTD staggered-stencil minimum while
/// keeping the second y-plane acoustically transparent (identical medium).
pub(crate) const NX: usize = 64; // lateral  192 mm
pub(crate) const NY: usize = 2; // quasi-2-D embedding
pub(crate) const NZ: usize = 64; // depth    192 mm

// Skull phantom geometry — radii in voxels from centre (32, 32)
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

pub(crate) const R_HEAD: f64 = 18.0; // 54 mm — outer scalp surface
pub(crate) const R_SKULL_OUT: f64 = 16.0; // 48 mm — outer cortical / scalp boundary
pub(crate) const R_DIPLOE: f64 = 14.0; // 42 mm — outer diploe boundary
pub(crate) const R_SKULL_IN: f64 = 12.0; // 36 mm — inner cortical / brain boundary
pub(crate) const R_BRAIN: f64 = 11.0; // 33 mm — brain surface (CSF buffer ≈ 3 mm)

// Hounsfield-unit phantom labels

/// Typical HU values per skull layer (Aubry 2003 Table I; Marsac 2017 Table 1).
pub(crate) const HU_WATER: f64 = 0.0; // water coupling bath
pub(crate) const HU_SCALP: f64 = 40.0; // soft tissue / scalp / dura
pub(crate) const HU_CORTICAL_OUT: f64 = 720.0; // outer cortical bone
pub(crate) const HU_DIPLOE: f64 = 380.0; // trabecular / diploe
pub(crate) const HU_CORTICAL_IN: f64 = 660.0; // inner cortical bone
pub(crate) const HU_BRAIN: f64 = 35.0; // grey/white matter average

/// Phase-correction example Medimodel series UID.  The directory contains
/// additional derived CT series with inconsistent orientation; this UID is the
/// same 67-slice skull CT series used by the companion example.
/// Full hemispherical aperture size used by the companion phase-correction demo.
pub(crate) const TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT: usize = 1024;

/// Colormap bounds [m/s] for skull image panels.
pub(crate) const C_LO: f64 = SOUND_SPEED_WATER_SIM; // 1500 m/s → blue
pub(crate) const C_HI: f64 = SOUND_SPEED_SKULL_CORTICAL; // 3100 m/s → red

// Stage-2 brain tissue FWI constants (Guasch 2020 §Methods)

/// Brain tissue sound speeds from Duck (1990) "Physical Properties of Tissue".
pub(crate) const C_GRAY: f64 = 1541.0; // gray matter [m/s]
pub(crate) const C_WHITE: f64 = 1520.0; // white matter [m/s]
pub(crate) const C_CSF: f64 = 1505.0; // cerebrospinal fluid [m/s]

/// Velocity bounds for brain tissue FWI (excludes bone which is frozen).
pub(crate) const BRAIN_C_MIN: f64 = 1480.0; // m/s
pub(crate) const BRAIN_C_MAX: f64 = 1560.0; // m/s

/// Velocity threshold for classifying a voxel as bone (frozen in Stage 2).
/// Corresponds to BVF > 0.143, approximately HU ≈ 143 → cortical bone onset.
pub(crate) const BONE_VELOCITY_THRESHOLD: f64 = 1714.0; // m/s

/// Peak frequency for brain tissue FWI.
/// At 400 kHz: λ_brain ≈ 3.8 mm; tissue velocity errors < 3% → no cycle-skipping.
pub(crate) const F0_BRAIN_HZ: f64 = 400_000.0; // Hz

/// Number of FWI iterations for brain tissue Stage 2.
pub(crate) const N_BRAIN_ITER: usize = 20;

/// Step size for brain tissue FWI (smaller than skull FWI — brain Δc is tiny).
pub(crate) const STEP_SIZE_BRAIN: f64 = 30.0; // m/s per normalised gradient step

/// MNI ICBM 2009c inner-skull radius at the coronal mid-plane `mm`.
/// The inner cortical surface is ≈ 82 mm from the brain centroid in this atlas.
pub(crate) const MNI_INNER_SKULL_RADIUS_MM: f64 = 82.0;

// Gaussian blur for CT-derived initial model

/// FWI gradient descent step size [m/s].
pub(crate) const STEP_SIZE: f64 = 50.0;

/// Pixel size per model panel [px].
pub(crate) const PANEL: usize = 320;

/// Colorbar height below each panel [px].
pub(crate) const COLORBAR_H: usize = 20;
