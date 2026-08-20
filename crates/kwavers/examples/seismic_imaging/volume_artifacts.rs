//! Orthogonal volume-rendering artifact for the 3-D seismic example.

use super::seismic_imaging::render::{put_pixel, velocity_color, write_png};
use super::{Array3, COLORBAR_H, NX, NY, NZ, PANEL};
use std::path::Path;

/// Write axial, coronal, and sagittal velocity slices with a shared colorbar.
pub(super) fn write_orthogonal_slices_png(
    path: &Path,
    vol: &Array3<f64>,
    c_lo: f64,
    c_hi: f64,
) -> std::io::Result<()> {
    let img_w = 3 * PANEL;
    let img_h = PANEL + COLORBAR_H;
    let mut rgb = vec![0_u8; img_w * img_h * 3];

    let mid_y = NY / 2;
    let mid_z = NZ / 2;
    let mid_x = NX / 2;

    // Panel 0 (axial): y = NY/2, ix = column, iz = row.
    for py in 0..PANEL {
        for px in 0..PANEL {
            let ix = (px * NX / PANEL).min(NX - 1);
            let iz = (py * NZ / PANEL).min(NZ - 1);
            let color = velocity_color(vol[[ix, mid_y, iz]], c_lo, c_hi);
            put_pixel(&mut rgb, img_w, img_h, px, py, color);
        }
    }

    // Panel 1 (coronal): z = NZ/2, ix = column, iy = row.
    for py in 0..PANEL {
        for px in 0..PANEL {
            let ix = (px * NX / PANEL).min(NX - 1);
            let iy = (py * NY / PANEL).min(NY - 1);
            let color = velocity_color(vol[[ix, iy, mid_z]], c_lo, c_hi);
            put_pixel(&mut rgb, img_w, img_h, PANEL + px, py, color);
        }
    }

    // Panel 2 (sagittal): x = NX/2, iy = column, iz = row.
    for py in 0..PANEL {
        for px in 0..PANEL {
            let iy = (px * NY / PANEL).min(NY - 1);
            let iz = (py * NZ / PANEL).min(NZ - 1);
            let color = velocity_color(vol[[mid_x, iy, iz]], c_lo, c_hi);
            put_pixel(&mut rgb, img_w, img_h, 2 * PANEL + px, py, color);
        }
    }

    // Colorbar row below all three panels.
    for panel_col in 0..3 {
        for px in 0..PANEL {
            let t = px as f64 / (PANEL - 1) as f64;
            let c = c_lo + t * (c_hi - c_lo);
            let color = velocity_color(c, c_lo, c_hi);
            for dy in 0..COLORBAR_H {
                put_pixel(
                    &mut rgb,
                    img_w,
                    img_h,
                    panel_col * PANEL + px,
                    PANEL + dy,
                    color,
                );
            }
        }
    }

    write_png(path, &rgb, img_w, img_h)
}
