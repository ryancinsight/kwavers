//! Display-normalization helpers for clinical theranostic workflows.

use leto::Array2;

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
