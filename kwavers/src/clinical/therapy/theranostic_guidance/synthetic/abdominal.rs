//! Synthetic abdominal CT phantoms for geometry verification.
//!
//! ## Physical specification
//!
//! ### Liver phantom — 224 × 192 × 320 mm at 2 mm isotropic
//! - Body ellipsoid: semi-axes (88 × 76 × 144 mm), HU 35 (soft tissue).
//! - Liver: offset 10 mm anterior and 10 mm right of body centroid,
//!   semi-axes (36 × 32 × 44 mm), HU 55 (hepatic parenchyma).
//! - Tumour: 10 mm sphere inside liver, HU 70, label 2.
//!
//! ### Kidney phantom — 150 × 135 × 210 mm at 1.5 mm isotropic
//! - Body ellipsoid: semi-axes (57 × 49.5 × 90 mm), HU 35.
//! - Right kidney: 20 mm posterior and 15 mm lateral from centroid,
//!   semi-axes (16.5 × 13.5 × 24 mm), HU 45 (renal cortex).
//! - Tumour: 6 mm sphere inside kidney, HU 70, label 2.
//!
//! ## Placement invariants
//!
//! All organ centres are inside the body ellipsoid.  The flood-fill
//! exterior-air algorithm requires only that the body is surrounded by air
//! (HU −1000) on all six faces, which is guaranteed by the phantom dimensions.
//! `nearest_exterior_skin_point` will place the bowl contact on the
//! anterior-lateral skin face, outside the body, as required.

use ndarray::Array3;

// ─── Liver phantom dimensions ─────────────────────────────────────────────────

const LNX: usize = 112; // AP axis
const LNY: usize = 96; // RL axis
const LNZ: usize = 160; // SI axis
const L_SPACING: f64 = 2.0; // mm

// ─── Kidney phantom dimensions ────────────────────────────────────────────────

const KNX: usize = 100; // AP axis
const KNY: usize = 90; // RL axis
const KNZ: usize = 140; // SI axis
const K_SPACING: f64 = 1.5; // mm

// ─── HU constants ─────────────────────────────────────────────────────────────

const HU_AIR: f64 = -1000.0;
const HU_BODY: f64 = 35.0;
const HU_LIVER: f64 = 55.0;
const HU_KIDNEY: f64 = 45.0;
const HU_TARGET: f64 = 70.0;

// ─── Geometry helpers ─────────────────────────────────────────────────────────

/// Ellipsoid radial parameter; inside when ≤ 1.
#[inline]
fn ell(
    ix: usize,
    iy: usize,
    iz: usize,
    cx: f64,
    cy: f64,
    cz: f64,
    rx: f64,
    ry: f64,
    rz: f64,
) -> f64 {
    let dx = (ix as f64 - cx) / rx;
    let dy = (iy as f64 - cy) / ry;
    let dz = (iz as f64 - cz) / rz;
    dx * dx + dy * dy + dz * dz
}

/// Sphere radial parameter; inside when ≤ 1.
#[inline]
fn sph(ix: usize, iy: usize, iz: usize, cx: f64, cy: f64, cz: f64, r: f64) -> f64 {
    ell(ix, iy, iz, cx, cy, cz, r, r, r)
}

// ─── Public constructors ──────────────────────────────────────────────────────

/// Synthetic liver abdominal CT phantom.
///
/// Returns `(ct_hu, label, spacing_mm)`.
/// - `ct_hu`: `[LNX, LNY, LNZ]`, values in Hounsfield units.
/// - `label`: `[LNX, LNY, LNZ]`, 0 = background, 1 = liver, 2 = tumour.
/// - `spacing_mm`: isotropic 2 mm.
pub fn synthetic_abdominal_liver_phantom() -> (Array3<f64>, Array3<i16>, [f64; 3]) {
    // Body centroid.
    let (cx, cy, cz) = (56.0_f64, 48.0, 80.0);
    // Body semi-axes [voxels].
    let (bx, by, bz) = (44.0_f64, 38.0, 72.0);
    // Liver centre: +5 voxels AP (anterior), +5 voxels RL (right).
    let (ox, oy, oz) = (cx + 5.0, cy + 5.0, cz);
    let (orx, ory, orz) = (18.0_f64, 16.0, 22.0);
    // Tumour centre: additional +5 AP, +3 RL inside liver.
    let (tx, ty, tz) = (ox + 5.0, oy + 3.0, oz + 2.0);
    let tr = 5.0_f64;

    let mut ct = Array3::<f64>::from_elem((LNX, LNY, LNZ), HU_AIR);
    let mut label = Array3::<i16>::zeros((LNX, LNY, LNZ));

    for x in 0..LNX {
        for y in 0..LNY {
            for z in 0..LNZ {
                if ell(x, y, z, cx, cy, cz, bx, by, bz) > 1.0 {
                    continue;
                }
                if ell(x, y, z, ox, oy, oz, orx, ory, orz) <= 1.0 {
                    if sph(x, y, z, tx, ty, tz, tr) <= 1.0 {
                        ct[[x, y, z]] = HU_TARGET;
                        label[[x, y, z]] = 2;
                    } else {
                        ct[[x, y, z]] = HU_LIVER;
                        label[[x, y, z]] = 1;
                    }
                } else {
                    ct[[x, y, z]] = HU_BODY;
                }
            }
        }
    }
    (ct, label, [L_SPACING; 3])
}

/// Synthetic kidney abdominal CT phantom.
///
/// Returns `(ct_hu, label, spacing_mm)`.
/// - `ct_hu`: `[KNX, KNY, KNZ]`, values in Hounsfield units.
/// - `label`: `[KNX, KNY, KNZ]`, 0 = background, 1 = kidney, 2 = tumour.
/// - `spacing_mm`: isotropic 1.5 mm.
pub fn synthetic_abdominal_kidney_phantom() -> (Array3<f64>, Array3<i16>, [f64; 3]) {
    let (cx, cy, cz) = (50.0_f64, 45.0, 70.0);
    let (bx, by, bz) = (38.0_f64, 33.0, 60.0);
    // Right kidney: +10 AP (anterior), −12 RL (right = posterior-lateral).
    let (ox, oy, oz) = (cx + 10.0, cy - 12.0, cz + 2.0);
    let (orx, ory, orz) = (11.0_f64, 9.0, 16.0);
    let (tx, ty, tz) = (ox + 3.0, oy + 2.0, oz);
    let tr = 4.0_f64;

    let mut ct = Array3::<f64>::from_elem((KNX, KNY, KNZ), HU_AIR);
    let mut label = Array3::<i16>::zeros((KNX, KNY, KNZ));

    for x in 0..KNX {
        for y in 0..KNY {
            for z in 0..KNZ {
                if ell(x, y, z, cx, cy, cz, bx, by, bz) > 1.0 {
                    continue;
                }
                if ell(x, y, z, ox, oy, oz, orx, ory, orz) <= 1.0 {
                    if sph(x, y, z, tx, ty, tz, tr) <= 1.0 {
                        ct[[x, y, z]] = HU_TARGET;
                        label[[x, y, z]] = 2;
                    } else {
                        ct[[x, y, z]] = HU_KIDNEY;
                        label[[x, y, z]] = 1;
                    }
                } else {
                    ct[[x, y, z]] = HU_BODY;
                }
            }
        }
    }
    (ct, label, [K_SPACING; 3])
}
