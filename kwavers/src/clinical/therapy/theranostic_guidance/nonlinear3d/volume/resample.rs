//! Lattice resampling kernels for nonlinear 3-D CT volume preparation.

use ndarray::Array3;

use super::bbox::BBox;

pub(super) fn resample_scalar(input: &Array3<f64>, bbox: BBox, n: usize) -> Array3<f64> {
    Array3::from_shape_fn((n, n, n), |(ix, iy, iz)| {
        let x = map_coord(ix, n, bbox.x0, bbox.x1);
        let y = map_coord(iy, n, bbox.y0, bbox.y1);
        let z = map_coord(iz, n, bbox.z0, bbox.z1);
        trilinear(input, x, y, z)
    })
}

pub(super) fn resample_labels(input: &Array3<i16>, bbox: BBox, n: usize) -> Array3<i16> {
    Array3::from_shape_fn((n, n, n), |(ix, iy, iz)| {
        let xr = map_range(ix, n, bbox.x0, bbox.x1);
        let yr = map_range(iy, n, bbox.y0, bbox.y1);
        let zr = map_range(iz, n, bbox.z0, bbox.z1);
        let mut value = 0i16;
        for x in xr.0..=xr.1 {
            for y in yr.0..=yr.1 {
                for z in zr.0..=zr.1 {
                    value = value.max(input[[x, y, z]]);
                }
            }
        }
        value
    })
}

pub(super) fn isotropic_spacing_m(bbox: BBox, spacing_mm: [f64; 3], n: usize) -> f64 {
    let sx = (bbox.x1 - bbox.x0 + 1) as f64 * spacing_mm[0] * 1.0e-3;
    let sy = (bbox.y1 - bbox.y0 + 1) as f64 * spacing_mm[1] * 1.0e-3;
    let sz = (bbox.z1 - bbox.z0 + 1) as f64 * spacing_mm[2] * 1.0e-3;
    sx.max(sy).max(sz) / n as f64
}

fn map_coord(idx: usize, n: usize, min: usize, max: usize) -> f64 {
    if n <= 1 {
        min as f64
    } else {
        min as f64 + idx as f64 * (max - min) as f64 / (n - 1) as f64
    }
}

fn map_range(idx: usize, n: usize, min: usize, max: usize) -> (usize, usize) {
    let width = max - min + 1;
    let start = min + idx * width / n;
    let end = min + (((idx + 1) * width).saturating_sub(1)) / n;
    (start.min(max), end.min(max))
}

fn trilinear(input: &Array3<f64>, x: f64, y: f64, z: f64) -> f64 {
    let dims = input.dim();
    let x0 = x.floor().clamp(0.0, (dims.0 - 1) as f64) as usize;
    let y0 = y.floor().clamp(0.0, (dims.1 - 1) as f64) as usize;
    let z0 = z.floor().clamp(0.0, (dims.2 - 1) as f64) as usize;
    let x1 = (x0 + 1).min(dims.0 - 1);
    let y1 = (y0 + 1).min(dims.1 - 1);
    let z1 = (z0 + 1).min(dims.2 - 1);
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;
    let tz = z - z0 as f64;
    let c00 = lerp(input[[x0, y0, z0]], input[[x1, y0, z0]], tx);
    let c10 = lerp(input[[x0, y1, z0]], input[[x1, y1, z0]], tx);
    let c01 = lerp(input[[x0, y0, z1]], input[[x1, y0, z1]], tx);
    let c11 = lerp(input[[x0, y1, z1]], input[[x1, y1, z1]], tx);
    lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}
