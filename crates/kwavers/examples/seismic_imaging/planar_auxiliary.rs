//! Auxiliary planar artifacts: priors, RTM, brain tissue, and CSV profiles.

use leto::Array3;

use super::seismic_planar_artifacts::{diverging_color, draw_acquisition_markers};
use super::{
    put_pixel, velocity_color, write_png, BONE_VELOCITY_THRESHOLD, BRAIN_C_MAX, BRAIN_C_MIN, DX,
    NX, NZ, PANEL, R_SKULL_IN,
};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

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
    let brain = super::seismic_brain_model::brain_support_from_hu(hu);

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
pub(super) fn write_rtm_panel(path: &Path, rtm_image: &Array3<f64>) -> io::Result<()> {
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
pub(super) fn write_brain_tissue_png(
    path: &Path,
    true_model: &Array3<f64>,
    reconstructed: &Array3<f64>,
) -> io::Result<()> {
    let img_w = 3 * PANEL;
    let img_h = PANEL + super::COLORBAR_H;
    let mut rgb = vec![0_u8; img_w * img_h * 3];

    let cx = (NX / 2) as f64;
    let cz = (NZ / 2) as f64;
    let is_brain = |ix: usize, iz: usize| -> bool {
        let r = ((ix as f64 - cx).powi(2) + (iz as f64 - cz).powi(2)).sqrt();
        r < R_SKULL_IN
    };
    let frozen_color = |c_ref: f64| -> [u8; 3] {
        if c_ref >= BONE_VELOCITY_THRESHOLD {
            [200, 200, 200]
        } else if c_ref >= 1502.0 {
            [140, 140, 140]
        } else {
            [40, 40, 40]
        }
    };

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
        let color = velocity_color(
            BRAIN_C_MIN + t * (BRAIN_C_MAX - BRAIN_C_MIN),
            BRAIN_C_MIN,
            BRAIN_C_MAX,
        );
        for dy in 0..super::COLORBAR_H {
            put_pixel(&mut rgb, img_w, img_h, px, PANEL + dy, color);
        }
    }

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
        let color = velocity_color(
            BRAIN_C_MIN + t * (BRAIN_C_MAX - BRAIN_C_MIN),
            BRAIN_C_MIN,
            BRAIN_C_MAX,
        );
        for dy in 0..super::COLORBAR_H {
            put_pixel(&mut rgb, img_w, img_h, PANEL + px, PANEL + dy, color);
        }
    }

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
                diverging_color(
                    reconstructed[[ix, 0, iz]] - true_model[[ix, 0, iz]],
                    max_diff,
                )
            } else {
                frozen_color(true_model[[ix, 0, iz]])
            };
            put_pixel(&mut rgb, img_w, img_h, 2 * PANEL + px, py, color);
        }
    }
    for px in 0..PANEL {
        let signed = (2.0 * px as f64 / (PANEL - 1) as f64 - 1.0) * max_diff;
        let color = diverging_color(signed, max_diff);
        for dy in 0..super::COLORBAR_H {
            put_pixel(&mut rgb, img_w, img_h, 2 * PANEL + px, PANEL + dy, color);
        }
    }

    write_png(path, &rgb, img_w, img_h)
}

/// Write the central-column (x = NX/2) velocity profiles to CSV.
pub(super) fn write_velocity_csv(
    path: &Path,
    true_model: &Array3<f64>,
    initial_model: &Array3<f64>,
    reconstructed: &Array3<f64>,
) -> io::Result<()> {
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
