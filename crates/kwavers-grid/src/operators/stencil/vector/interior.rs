//! Interior Cartesian derivatives. The output halo is already zero on newly
//! allocated arrays and must be explicitly cleared on reused buffers.
use super::super::centered_first_derivative_sum;
use super::VectorDerivative;
use eunomia::FloatElement;

pub(super) fn write<
    O: VectorDerivative<N>,
    T: FloatElement,
    const N: usize,
    const CLEAR_HALO: bool,
>(
    [nx, ny, nz]: [usize; 3],
    weights: &[T],
    scale: [T; 3],
    mut output: [&mut [T]; N],
    sample: impl Fn(usize, [usize; 3]) -> T,
) {
    let r = weights.len();
    let zero = T::from_f64(0.0);
    if [nx, ny, nz].iter().any(|&n| n <= 2 * r) {
        if CLEAR_HALO {
            for dst in output {
                dst.fill(zero);
            }
        }
        return;
    }
    for i in 0..nx {
        for j in 0..ny {
            let base = (i * ny + j) * nz;
            if i < r || i >= nx - r || j < r || j >= ny - r {
                if CLEAR_HALO {
                    for dst in &mut output {
                        dst[base..base + nz].fill(zero);
                    }
                }
                continue;
            }
            if CLEAR_HALO {
                for dst in &mut output {
                    dst[base..base + r].fill(zero);
                    dst[base + nz - r..base + nz].fill(zero);
                }
            }
            for k in r..nz - r {
                let value = O::evaluate(|component, direction| {
                    let c = component.index();
                    let axis = direction.index();
                    centered_first_derivative_sum(weights, |d| {
                        let mut hi = [i, j, k];
                        let mut lo = hi;
                        hi[axis] += d;
                        lo[axis] -= d;
                        sample(c, hi) - sample(c, lo)
                    }) * scale[axis]
                });
                for (dst, v) in output.iter_mut().zip(value) {
                    dst[base + k] = v;
                }
            }
        }
    }
}
