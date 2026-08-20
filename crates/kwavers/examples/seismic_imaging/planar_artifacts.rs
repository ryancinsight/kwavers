//! Planar seismic artifact rendering.

use super::{
    Array2, Array3, BONE_VELOCITY_THRESHOLD, BRAIN_C_MAX, BRAIN_C_MIN, C_HI, C_LO, COLORBAR_H,
    CtVolume, DX, NX, NZ, PANEL, R_SKULL_IN, put_pixel, velocity_color, write_png,
};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

// Image output
// ─────────────────────────────────────────────────────────────────────────────

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
#[derive(Clone, Copy)]
pub(super) struct VelocityScale {
    pub(super) lo: f64,
    pub(super) hi: f64,
}

fn draw_velocity_panel(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    x_offset: usize,
    y_offset: usize,
    model: &Array3<f64>,
    scale: VelocityScale,
) {
    for py in 0..PANEL {
        for px in 0..PANEL {
            let ix = (px * NX / PANEL).min(NX - 1);
            let iz = (py * NZ / PANEL).min(NZ - 1);
            let color = velocity_color(model[[ix, 0, iz]], scale.lo, scale.hi);
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
#[derive(Clone, Copy)]
pub(super) struct AcquisitionMarkers<'a> {
    pub(super) shot_positions: &'a [(usize, usize)],
    pub(super) active_elements: &'a [(usize, usize)],
}

pub(super) fn write_three_plane_png(
    path: &Path,
    true_model: &Array3<f64>,
    reconstructed: &Array3<f64>,
    velocity_scale: VelocityScale,
    acquisition: AcquisitionMarkers<'_>,
    ct_vol: Option<&CtVolume>,
) -> io::Result<()> {
    let img_w = 3 * PANEL;
    let c_lo = velocity_scale.lo;
    let c_hi = velocity_scale.hi;

    if let Some(vol) = ct_vol {
        // ── 3×2 grid: CT triplanar (top) + FWI reconstruction (bottom) ───
        // Row heights: PANEL + COLORBAR_H for CT, then PANEL + COLORBAR_H for FWI.
        let img_h = 2 * (PANEL + COLORBAR_H);
        let mut rgb = vec![0_u8; img_w * img_h * 3];

        let [nx_ct, ny_ct, nz_ct] = vol.hu().shape();
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
                    ct_bone_color(vol.hu()[[ix, cy_ct, iz]]),
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
                    ct_bone_color(vol.hu()[[ix, iy, cz_ct]]),
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
                    ct_bone_color(vol.hu()[[cx_ct, iy, iz]]),
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
        draw_velocity_panel(
            &mut rgb,
            img_w,
            img_h,
            0,
            fwi_y0,
            true_model,
            velocity_scale,
        );
        draw_acquisition_markers(
            &mut rgb,
            img_w,
            img_h,
            0,
            fwi_y0,
            acquisition.shot_positions,
            acquisition.active_elements,
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
            velocity_scale,
        );
        draw_acquisition_markers(
            &mut rgb,
            img_w,
            img_h,
            PANEL,
            fwi_y0,
            acquisition.shot_positions,
            acquisition.active_elements,
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

        draw_velocity_panel(&mut rgb, img_w, img_h, 0, 0, true_model, velocity_scale);
        draw_acquisition_markers(
            &mut rgb,
            img_w,
            img_h,
            0,
            0,
            acquisition.shot_positions,
            acquisition.active_elements,
        );
        draw_colorbar(&mut rgb, img_w, img_h, 0, 0, c_lo, c_hi);

        draw_velocity_panel(
            &mut rgb,
            img_w,
            img_h,
            PANEL,
            0,
            reconstructed,
            velocity_scale,
        );
        draw_acquisition_markers(
            &mut rgb,
            img_w,
            img_h,
            PANEL,
            0,
            acquisition.shot_positions,
            acquisition.active_elements,
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
pub(super) fn write_velocity_panels(
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
    let velocity_scale = VelocityScale { lo: C_LO, hi: C_HI };

    draw_velocity_panel(&mut rgb, img_w, img_h, 0, 0, true_model, velocity_scale);
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

    draw_velocity_panel(
        &mut rgb,
        img_w,
        img_h,
        PANEL,
        0,
        initial_model,
        velocity_scale,
    );
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
        velocity_scale,
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
pub(super) fn brain_support_from_hu(hu: &Array3<f64>) -> Array2<bool> {
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
pub(super) fn write_brain_prior_png(
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
pub(super) fn write_rtm_panel(path: &Path, rtm_image: &Array3<f64>) -> std::io::Result<()> {
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
pub(super) fn write_brain_tissue_png(
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
        .filter(|([ix, _, iz], _)| is_brain(*ix, *iz))
        .map(|([ix, _, iz], &t)| (reconstructed[[ix, 0, iz]] - t).abs())
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
pub(super) fn write_velocity_csv(
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

// ─────────────────────────────────────────────────────────────────────────────
