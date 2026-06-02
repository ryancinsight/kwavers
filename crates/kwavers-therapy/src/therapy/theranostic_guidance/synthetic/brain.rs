//! Synthetic brain CT phantom for calvarium focused-bowl placement verification.
//!
//! ## Physical specification
//!
//! Grid: 128 × 128 × 128 voxels at 1.5 mm isotropic → 192 × 192 × 192 mm.
//!
//! The head is constructed from a pair of half-ellipsoids sharing the lateral
//! (x, y) radii but using different z-radii on each side of the equatorial
//! plane (z = 64):
//!
//! - **Inferior half** (z < 64): larger z-radius (60 voxels = 90 mm) simulates
//!   the larger jaw/neck cross-section.
//! - **Superior half** (z ≥ 64): smaller z-radius (48 voxels = 72 mm) simulates
//!   the tapered skull dome.
//!
//! This asymmetry ensures `plan_transcranial_focused_bowl_placement` correctly
//! identifies z = 0 as inferior and z = 127 as superior, placing bowl elements on the
//! calvarium (z > peak_z = 64).
//!
//! ## Layer structure
//!
//! | Layer   | Voxel depth from outer surface | HU  | Classification     |
//! |---------|-------------------------------|-----|--------------------|
//! | Scalp   | 0–2 voxels (0–3 mm)           |  35 | soft tissue        |
//! | Skull   | 2–6 voxels (3–9 mm)           | 700 | cortical bone      |
//! | Brain   | interior                      |  40 | soft tissue        |
//!
//! The scalp and brain are both below the default skull_hu_threshold of 300 HU,
//! matching the separation used by `plan_transcranial_focused_bowl_placement`.

use ndarray::Array3;

const NX: usize = 128;
const NY: usize = 128;
const NZ: usize = 128;
const SPACING_MM: f64 = 1.5;

const HU_AIR: f64 = -1000.0;
const HU_SCALP: f64 = 35.0; // soft tissue; < skull_hu_threshold
const HU_SKULL: f64 = 700.0; // cortical bone; > skull_hu_threshold
                             // Brain interior layers (all < skull_hu_threshold = 300)
const HU_GRAY_MATTER: f64 = 37.0; // cortical and deep gray matter (c ≈ 1560 m/s)
const HU_WHITE_MATTER: f64 = 28.0; // deep white matter tracts (c ≈ 1580 m/s)
const HU_CSF: f64 = 10.0; // cerebrospinal fluid in lateral ventricles (c ≈ 1515 m/s)
const HU_THALAMUS: f64 = 38.0; // thalamic nuclei — primary histotripsy target

/// Asymmetric half-ellipsoid radial parameter.
///
/// `rz_inf` applies when `iz < cz` (inferior half),
/// `rz_sup` applies when `iz ≥ cz` (superior half).
/// Returns ≤ 1.0 when the voxel is inside the half-ellipsoid.
#[inline]
#[allow(clippy::too_many_arguments)]
fn head_r(
    ix: usize,
    iy: usize,
    iz: usize,
    cx: f64,
    cy: f64,
    cz: f64,
    rx: f64,
    ry: f64,
    rz_inf: f64,
    rz_sup: f64,
) -> f64 {
    let dz = iz as f64 - cz;
    let rz = if dz < 0.0 { rz_inf } else { rz_sup };
    let dx = (ix as f64 - cx) / rx;
    let dy = (iy as f64 - cy) / ry;
    let dzn = dz / rz;
    dx * dx + dy * dy + dzn * dzn
}

/// True if voxel `(ix, iy, iz)` is inside a lateral ventricle.
///
/// The lateral ventricles are modelled as two mirror-symmetric ellipsoidal CSF
/// spaces flanking the midline, located superior and slightly posterior to the
/// geometric brain centre.  Their primary acoustic effect is to form a CSF
/// channel that reduces skull-to-target attenuation in transcranial sonication.
///
/// Geometry (voxels, relative to brain centre at `(cx, cy, cz)`):
/// - Body centres: `(cx ± 8, cy, cz + 2)`
/// - Semi-axes: `(a_lat=5, a_ant=4, a_sup=8)` voxels
///
/// # Reference
/// Drury et al. (1996) MNI152 ventricle atlas; 1.5 mm isotropic spacing.
#[inline]
fn in_lateral_ventricle(ix: usize, iy: usize, iz: usize, cx: f64, cy: f64, cz: f64) -> bool {
    let dz = (iz as f64 - (cz + 2.0)) / 8.0;
    let dy = (iy as f64 - cy) / 4.0;
    let dx_l = (ix as f64 - (cx - 8.0)) / 5.0;
    let dx_r = (ix as f64 - (cx + 8.0)) / 5.0;
    dx_l * dx_l + dy * dy + dz * dz <= 1.0 || dx_r * dx_r + dy * dy + dz * dz <= 1.0
}

/// True if voxel `(ix, iy, iz)` is inside the bilateral thalamic nuclei.
///
/// The thalami are modelled as two ovoid deep-gray-matter structures at the
/// geometric centre of the brain, slightly inferior and lateral to the midline.
/// They are the standard stereotaxic target for deep-brain histotripsy
/// (essential-tremor, Parkinson's disease).
///
/// Geometry (voxels): centres at `(cx ± 5, cy, cz − 4)`, semi-axes `(4, 4, 5)`.
///
/// # Reference
/// Behrens et al. (2003), *Nat. Neurosci.* 6, 750 (thalamic parcellation atlas).
#[inline]
fn in_thalamus(ix: usize, iy: usize, iz: usize, cx: f64, cy: f64, cz: f64) -> bool {
    let dz = (iz as f64 - (cz - 4.0)) / 5.0;
    let dy = (iy as f64 - cy) / 4.0;
    let dx_l = (ix as f64 - (cx - 5.0)) / 4.0;
    let dx_r = (ix as f64 - (cx + 5.0)) / 4.0;
    dx_l * dx_l + dy * dy + dz * dz <= 1.0 || dx_r * dx_r + dy * dy + dz * dz <= 1.0
}

/// Synthetic brain CT phantom.
///
/// Returns `(ct_hu, spacing_mm)`.
/// - `ct_hu`: shape `[NX, NY, NZ]` with values in Hounsfield units.
/// - `spacing_mm`: isotropic 1.5 mm.
///
/// ## Anatomical layers (outward to inward)
///
/// | Region              | HU   | Acoustic property         |
/// |---------------------|------|---------------------------|
/// | Air                 | −1000| opaque                    |
/// | Scalp               | 35   | c ≈ 1550 m/s              |
/// | Cortical skull      | 700  | c ≈ 2900 m/s, α high      |
/// | Cortical gray matter| 37   | c ≈ 1560 m/s              |
/// | White matter        | 28   | c ≈ 1580 m/s              |
/// | Lateral ventricles  | 10   | c ≈ 1515 m/s (CSF)        |
/// | Thalamic nuclei     | 38   | c ≈ 1560 m/s (target)     |
pub fn synthetic_brain_phantom() -> (Array3<f64>, [f64; 3]) {
    let (cx, cy, cz) = (64.0_f64, 64.0, 64.0);

    // Outer: scalp surface. Lateral radii (rx, ry) are shared across all shells.
    let (rx, ry) = (50.0_f64, 58.0);
    let (rz_inf, rz_sup) = (60.0_f64, 48.0);

    // Middle: skull outer surface (scalp = 2 voxels thick).
    let (mx, my) = (rx - 2.0, ry - 2.0);
    let (mz_inf, mz_sup) = (rz_inf - 2.0, rz_sup - 2.0);

    // Inner: skull inner surface (skull shell = 4 voxels thick, ~6 mm).
    let (inner_rx, inner_ry) = (mx - 4.0, my - 4.0);
    let (inner_rz_inf, inner_rz_sup) = (mz_inf - 4.0, mz_sup - 4.0);

    let mut ct = Array3::<f64>::from_elem((NX, NY, NZ), HU_AIR);

    for x in 0..NX {
        for y in 0..NY {
            for z in 0..NZ {
                let r_outer = head_r(x, y, z, cx, cy, cz, rx, ry, rz_inf, rz_sup);
                if r_outer > 1.0 {
                    continue; // air; already initialised
                }
                let r_mid = head_r(x, y, z, cx, cy, cz, mx, my, mz_inf, mz_sup);
                let r_inner = head_r(
                    x,
                    y,
                    z,
                    cx,
                    cy,
                    cz,
                    inner_rx,
                    inner_ry,
                    inner_rz_inf,
                    inner_rz_sup,
                );

                ct[[x, y, z]] = if r_inner <= 1.0 {
                    // Brain interior: check anatomical sub-structures first.
                    if in_lateral_ventricle(x, y, z, cx, cy, cz) {
                        HU_CSF // lateral ventricles (CSF)
                    } else if in_thalamus(x, y, z, cx, cy, cz) {
                        HU_THALAMUS // thalamic nuclei — histotripsy target
                    } else if r_inner > 0.92 {
                        HU_GRAY_MATTER // cortical mantle (~3 mm at 1.5 mm/voxel)
                    } else {
                        HU_WHITE_MATTER // deep white matter
                    }
                } else if r_mid <= 1.0 {
                    HU_SKULL
                } else {
                    HU_SCALP
                };
            }
        }
    }
    (ct, [SPACING_MM; 3])
}
