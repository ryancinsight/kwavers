//! Separable 3-D Gaussian initial-model construction.

use super::{Array3, NX, NY, NZ};
use moirai_parallel::{map_collect_index_with, Adaptive};

/// Blur a volume sequentially along x, y, and z with clamped boundaries.
///
/// Each output voxel is computed by one provider-owned Moirai map operation per
/// axis. The normalized kernel is truncated at `3σ`.
pub(super) fn gaussian_blur_3d(model: &Array3<f64>, sigma: f64) -> Array3<f64> {
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
    let cell_count = NX * NY * NZ;

    let tmp_x_values = map_collect_index_with::<Adaptive, _, _>(cell_count, |idx| {
        let ix = idx / (NY * NZ);
        let rem = idx % (NY * NZ);
        let iy = rem / NZ;
        let iz = rem % NZ;
        let mut acc = 0.0_f64;
        for (ki, &kw) in kernel.iter().enumerate() {
            let si =
                (ix as isize + ki as isize - radius as isize).clamp(0, NX as isize - 1) as usize;
            acc += kw * model[[si, iy, iz]];
        }
        acc
    });
    let tmp_x = Array3::<f64>::from_shape_vec((NX, NY, NZ), tmp_x_values)
        .expect("invariant: flat Moirai x-pass preserves model shape length");

    let tmp_y_values = map_collect_index_with::<Adaptive, _, _>(cell_count, |idx| {
        let ix = idx / (NY * NZ);
        let rem = idx % (NY * NZ);
        let iy = rem / NZ;
        let iz = rem % NZ;
        let mut acc = 0.0_f64;
        for (ki, &kw) in kernel.iter().enumerate() {
            let sj =
                (iy as isize + ki as isize - radius as isize).clamp(0, NY as isize - 1) as usize;
            acc += kw * tmp_x[[ix, sj, iz]];
        }
        acc
    });
    let tmp_y = Array3::<f64>::from_shape_vec((NX, NY, NZ), tmp_y_values)
        .expect("invariant: flat Moirai y-pass preserves model shape length");

    let out_values = map_collect_index_with::<Adaptive, _, _>(cell_count, |idx| {
        let ix = idx / (NY * NZ);
        let rem = idx % (NY * NZ);
        let iy = rem / NZ;
        let iz = rem % NZ;
        let mut acc = 0.0_f64;
        for (ki, &kw) in kernel.iter().enumerate() {
            let sk =
                (iz as isize + ki as isize - radius as isize).clamp(0, NZ as isize - 1) as usize;
            acc += kw * tmp_y[[ix, iy, sk]];
        }
        acc
    });

    Array3::<f64>::from_shape_vec((NX, NY, NZ), out_values)
        .expect("invariant: flat Moirai z-pass preserves model shape length")
}

#[cfg(test)]
mod tests {
    use super::gaussian_blur_3d;
    use super::{NX, NY, NZ};
    use leto::Array3;

    #[test]
    fn blur_preserves_a_constant_volume_within_roundoff() {
        let value = 1525.0;
        let volume = Array3::from_elem((NX, NY, NZ), value);
        let blurred = gaussian_blur_3d(&volume, 3.0);
        let bound = 64.0 * f64::EPSILON * value.abs();

        assert!(blurred
            .iter()
            .all(|&sample| (sample - value).abs() <= bound));
    }

    #[test]
    fn blur_spreads_an_impulse_without_cross_axis_leakage() {
        let mut volume = Array3::zeros((NX, NY, NZ));
        let center = (NX / 2, NY / 2, NZ / 2);
        volume[[center.0, center.1, center.2]] = 1.0;

        let blurred = gaussian_blur_3d(&volume, 3.0);

        assert!(blurred[[center.0, center.1, center.2]] > 0.0);
        assert!(blurred[[center.0 - 1, center.1, center.2]] > 0.0);
        assert!(blurred[[center.0, center.1, center.2 - 1]] > 0.0);
        assert!(blurred[[center.0, center.1 - 1, center.2]] > 0.0);
    }
}
