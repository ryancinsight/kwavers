//! Pressure exposure and display-normalization helpers for clinical workflows.

use crate::solver::inverse::same_aperture::{PlanarPoint, C_REF_M_S};
use ndarray::Array2;

use super::config::TheranosticInverseConfig;
use super::geometry::DeviceLayout;
use super::medium::PreparedTheranosticSlice;

pub fn exposure_map(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    config: &TheranosticInverseConfig,
) -> Array2<f64> {
    let (nx, ny) = prepared.ct_hu.dim();
    let cx = (nx - 1) as f64 * 0.5;
    let cy = (ny - 1) as f64 * 0.5;
    let frequency = config.frequencies_hz.last().copied().unwrap_or(500_000.0);
    let k = std::f64::consts::TAU * frequency / C_REF_M_S;
    let mut field = Array2::<f64>::zeros((nx, ny));
    for ix in 0..nx {
        for iy in 0..ny {
            if !prepared.body_mask[[ix, iy]] {
                continue;
            }
            let point = PlanarPoint {
                x_m: (ix as f64 - cx) * prepared.spacing_m,
                y_m: (iy as f64 - cy) * prepared.spacing_m,
            };
            let mut pressure = 0.0;
            for source in &layout.therapy_elements {
                let d = distance(point, *source).max(prepared.spacing_m);
                let df = distance(layout.focus_m, *source).max(prepared.spacing_m);
                pressure += (k * (d - df)).cos() / d.sqrt();
            }
            field[[ix, iy]] = pressure.abs();
        }
    }
    let normalized = normalize_positive(&field, &prepared.body_mask);
    normalized.mapv(|value| value * config.source_pressure_pa)
}

pub fn normalize_positive(image: &Array2<f64>, mask: &Array2<bool>) -> Array2<f64> {
    let mut max_value = 0.0;
    for (value, active) in image.iter().zip(mask.iter()) {
        if *active {
            max_value = f64::max(max_value, value.abs());
        }
    }
    if max_value <= 0.0 {
        return Array2::<f64>::zeros(image.dim());
    }
    Array2::from_shape_fn(image.dim(), |idx| {
        if mask[idx] {
            (image[idx] / max_value).clamp(0.0, 1.0)
        } else {
            0.0
        }
    })
}

fn distance(a: PlanarPoint, b: PlanarPoint) -> f64 {
    (a.x_m - b.x_m).hypot(a.y_m - b.y_m)
}
