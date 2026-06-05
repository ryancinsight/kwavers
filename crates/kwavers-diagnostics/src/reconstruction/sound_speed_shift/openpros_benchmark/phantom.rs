//! Deterministic reduced prostate SOS phantom.

use ndarray::Array2;

use super::types::OpenProsShiftBenchmarkConfig;

pub(super) fn active_mask(config: &OpenProsShiftBenchmarkConfig) -> Array2<bool> {
    Array2::from_elem(config.shape(), true)
}

pub(super) fn shift_phantom(config: &OpenProsShiftBenchmarkConfig) -> Array2<f64> {
    let shape = config.shape();
    Array2::from_shape_fn(shape, |(ix, iy)| {
        let axial = normalized(ix, shape.0);
        let lateral = normalized(iy, shape.1);
        let gland = ellipse(axial + 0.08, lateral, 0.34, 0.52);
        let bladder = ellipse(axial - 0.46, lateral, 0.20, 0.46);
        let lesion = ellipse(axial + 0.02, lateral - 0.22, 0.10, 0.14);
        let bone = axial > 0.54 && lateral.abs() > 0.42;
        let heterogeneity = if gland {
            4.0 * (std::f64::consts::PI * (2.0 * axial + lateral)).sin()
        } else {
            0.0
        };

        let mut shift = 0.0;
        if gland {
            shift += 28.0 + heterogeneity;
        }
        if bladder {
            shift -= 34.0;
        }
        if lesion {
            shift += 54.0;
        }
        if bone {
            shift += 120.0;
        }
        shift
    })
}

fn normalized(index: usize, count: usize) -> f64 {
    if count <= 1 {
        0.0
    } else {
        2.0 * index as f64 / (count - 1) as f64 - 1.0
    }
}

fn ellipse(axial: f64, lateral: f64, axial_radius: f64, lateral_radius: f64) -> bool {
    (axial / axial_radius).powi(2) + (lateral / lateral_radius).powi(2) <= 1.0
}
