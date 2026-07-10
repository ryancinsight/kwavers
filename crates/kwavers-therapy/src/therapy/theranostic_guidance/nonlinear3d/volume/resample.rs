//! Lattice resampling kernels for nonlinear 3-D CT volume preparation.

use leto::Array3;

use kwavers_math::numerics::operators::interpolation::trilinear_index_space;

use crate::therapy::theranostic_guidance::geometry::IndexBounds3;

pub(super) fn resample_scalar(input: &Array3<f64>, bbox: IndexBounds3, n: usize) -> Array3<f64> {
    Array3::from_shape_fn((n, n, n), |[ix, iy, iz]| {
        let x = map_coord(ix, n, bbox.x0, bbox.x1);
        let y = map_coord(iy, n, bbox.y0, bbox.y1);
        let z = map_coord(iz, n, bbox.z0, bbox.z1);
        trilinear_index_space(input, x, y, z)
    })
}

pub(super) fn resample_labels(input: &Array3<i16>, bbox: IndexBounds3, n: usize) -> Array3<i16> {
    Array3::from_shape_fn((n, n, n), |[ix, iy, iz]| {
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

pub(super) fn isotropic_spacing_m(bbox: IndexBounds3, spacing_mm: [f64; 3], n: usize) -> f64 {
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
