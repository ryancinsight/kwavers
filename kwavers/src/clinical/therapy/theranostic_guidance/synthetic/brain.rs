//! Synthetic brain CT phantom for calvarium helmet placement verification.
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
//! This asymmetry ensures `plan_brain_helmet_placement` correctly identifies
//! z = 0 as inferior and z = 127 as superior, placing helmet elements on the
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
//! matching the separation used by `plan_brain_helmet_placement`.

use ndarray::Array3;

const NX: usize = 128;
const NY: usize = 128;
const NZ: usize = 128;
const SPACING_MM: f64 = 1.5;

const HU_AIR: f64 = -1000.0;
const HU_SCALP: f64 = 35.0; // soft tissue; < skull_hu_threshold
const HU_SKULL: f64 = 700.0; // cortical bone; > skull_hu_threshold
const HU_BRAIN: f64 = 40.0; // soft tissue; < skull_hu_threshold

/// Asymmetric half-ellipsoid radial parameter.
///
/// `rz_inf` applies when `iz < cz` (inferior half),
/// `rz_sup` applies when `iz ≥ cz` (superior half).
/// Returns ≤ 1.0 when the voxel is inside the half-ellipsoid.
#[inline]
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

/// Synthetic brain CT phantom.
///
/// Returns `(ct_hu, spacing_mm)`.
/// - `ct_hu`: shape `[NX, NY, NZ]` with values in Hounsfield units.
/// - `spacing_mm`: isotropic 1.5 mm.
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
                let r_inner = head_r(x, y, z, cx, cy, cz, inner_rx, inner_ry, inner_rz_inf, inner_rz_sup);

                ct[[x, y, z]] = if r_inner <= 1.0 {
                    HU_BRAIN
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
