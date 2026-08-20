//! CT-derived initial-model construction for the planar inversion workflow.

use leto::Array3;

use super::{NX, NY, NZ};

/// Blur a model separably in the x–z plane while preserving every y slice.
///
/// The one-dimensional kernel is truncated at `3σ` and normalized before the
/// x and z passes. Boundary samples use clamped reflection, so the output
/// remains defined at the model edges without allocating a padded volume.
///
/// The CT-derived prior supplies the inversion with the skull location before
/// FWI sharpens its interfaces. Guasch (2020), npj Digital Medicine,
/// §Methods, describes this initial-model strategy for transcranial FWI.
pub(super) fn gaussian_blur_xz(model: &Array3<f64>, sigma: f64) -> Array3<f64> {
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel_size = 2 * radius + 1;

    let raw: Vec<f64> = (0..kernel_size)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let ksum: f64 = raw.iter().sum();
    let kernel: Vec<f64> = raw.iter().map(|&k| k / ksum).collect();

    let mut tmp = Array3::<f64>::zeros((NX, NY, NZ));
    for j in 0..NY {
        for k in 0..NZ {
            for i in 0..NX {
                let mut value = 0.0;
                for (kernel_index, &weight) in kernel.iter().enumerate() {
                    let source = (i as isize + kernel_index as isize - radius as isize)
                        .clamp(0, NX as isize - 1) as usize;
                    value += weight * model[[source, j, k]];
                }
                tmp[[i, j, k]] = value;
            }
        }
    }

    let mut result = Array3::<f64>::zeros((NX, NY, NZ));
    for j in 0..NY {
        for i in 0..NX {
            for k in 0..NZ {
                let mut value = 0.0;
                for (kernel_index, &weight) in kernel.iter().enumerate() {
                    let source = (k as isize + kernel_index as isize - radius as isize)
                        .clamp(0, NZ as isize - 1) as usize;
                    value += weight * tmp[[i, j, source]];
                }
                result[[i, j, k]] = value;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::gaussian_blur_xz;
    use super::{NX, NY, NZ};
    use leto::Array3;

    #[test]
    fn blur_preserves_a_constant_model_within_roundoff() {
        let value = 1525.0;
        let model = Array3::from_elem((NX, NY, NZ), value);
        let blurred = gaussian_blur_xz(&model, 3.0);
        let bound = 64.0 * f64::EPSILON * value.abs();

        assert!(
            blurred
                .iter()
                .all(|&sample| (sample - value).abs() <= bound)
        );
    }

    #[test]
    fn blur_spreads_an_impulse_without_cross_slice_leakage() {
        let mut model = Array3::zeros((NX, NY, NZ));
        let center = (NX / 2, 0, NZ / 2);
        model[[center.0, center.1, center.2]] = 1.0;

        let blurred = gaussian_blur_xz(&model, 3.0);

        assert!(blurred[[center.0, center.1, center.2]] > 0.0);
        assert!(blurred[[center.0 - 1, center.1, center.2]] > 0.0);
        assert!(blurred[[center.0, center.1, center.2 - 1]] > 0.0);
        assert_eq!(blurred[[center.0, 1, center.2]], 0.0);
    }
}
