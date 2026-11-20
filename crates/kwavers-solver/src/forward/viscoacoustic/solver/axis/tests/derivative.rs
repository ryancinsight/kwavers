use super::super::super::{fft_wavenumbers, ViscoacousticMemorySolver};
use kwavers_math::fft::{
    fft_3d_axis_complex_inplace, ifft_3d_axis_complex_inplace, Complex64, Fft3d,
};
use leto::Array3;

#[test]
fn odd_grid_retains_the_highest_positive_wavenumber() {
    let spacing = 2.5e-4;
    let fundamental = std::f64::consts::TAU / (5.0 * spacing);
    let k = fft_wavenumbers(5, spacing);
    assert_eq!(
        k,
        [
            0.0,
            fundamental,
            2.0 * fundamental,
            -2.0 * fundamental,
            -fundamental
        ]
    );
    assert_eq!(fft_wavenumbers(1, spacing), [0.0]);
    let even = fft_wavenumbers(4, spacing);
    assert_eq!(even[2], 0.0, "even Nyquist is its own conjugate");
    assert_eq!(even[1], -even[3]);
}

fn reference_axis_derivative(
    fft: &Fft3d,
    k: &[f64],
    axis: usize,
    field: &Array3<f64>,
    scratch: &mut Array3<Complex64>,
    out: &mut Array3<f64>,
) {
    let [nx, ny, nz] = field.shape();
    for (dst, &src) in scratch
        .as_slice_mut()
        .expect("test scratch is dense")
        .iter_mut()
        .zip(field.as_slice().expect("test field is dense"))
    {
        *dst = Complex64::new(src, 0.0);
    }
    fft_3d_axis_complex_inplace(fft, scratch, axis);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mode = [x, y, z][axis];
                scratch[[x, y, z]] *= Complex64::new(0.0, k[mode]);
            }
        }
    }
    ifft_3d_axis_complex_inplace(fft, scratch, axis);
    for (dst, src) in out
        .as_slice_mut()
        .expect("test output is dense")
        .iter_mut()
        .zip(scratch.as_slice().expect("test scratch is dense"))
    {
        *dst = src.re;
    }
}

fn assert_complex_bits_eq(actual: &Array3<Complex64>, expected: &Array3<Complex64>) {
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }
}

fn check_singleton_axis(shape: [usize; 3], axis: usize) {
    let [nx, ny, nz] = shape;
    let mut solver = ViscoacousticMemorySolver::new(
        crate::forward::viscoacoustic::ViscoacousticGrid::new(
            nx, ny, nz, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-8,
        ),
        1_000.0,
        2.25e9,
        &[],
    )
    .expect("test solver parameters are valid");
    let fft = solver.fft.clone();
    // The solver omits the wavenumber vector of an inactive axis, so the
    // test builds the (single-entry) vector it exercises directly.
    let axis_len = [nx, ny, nz][axis];
    assert_eq!(axis_len, 1, "check_singleton_axis targets a singleton axis");
    let k = fft_wavenumbers(axis_len, 1.0e-4);

    for seed in 0..3 {
        let field = Array3::from_shape_fn((nx, ny, nz), |[x, y, z]| {
            ((17 * x + 11 * y + 5 * z + seed) as f64 * 0.37).sin()
        });
        let mut reference_scratch = Array3::zeros((nx, ny, nz));
        let mut expected = Array3::from_elem((nx, ny, nz), f64::NAN);
        reference_axis_derivative(
            &fft,
            &k,
            axis,
            &field,
            &mut reference_scratch,
            &mut expected,
        );

        solver.cbuf.fill(Complex64::new(seed as f64 + 1.0, -3.0));
        let scratch_before = solver.cbuf.clone();
        let mut actual = Array3::from_elem((nx, ny, nz), f64::NAN);
        ViscoacousticMemorySolver::axis_derivative(
            &fft,
            &k,
            axis,
            &field,
            &mut solver.cbuf,
            &mut actual,
        );

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            // Exact zero is independent of the sign introduced by
            // i·0 applied to a negative real sample in the FFT oracle.
            assert_eq!(actual.to_bits(), (0.0 + expected).to_bits());
            assert_eq!(actual.to_bits(), 0.0_f64.to_bits());
        }
        assert_complex_bits_eq(&solver.cbuf, &scratch_before);
    }
}

/// On an active axis each element is scaled by the same `i·k[mode]` the
/// per-element reference applies, so the derivative and the spectrum left
/// in the scratch match it to the bit on every axis, for even and odd
/// extents, and for a volume large enough that the passes spread over
/// tasks.
#[test]
fn active_axes_match_fft_reference_to_the_bit() {
    for shape in [[8, 6, 5], [7, 4, 3], [37, 29, 31]] {
        let [nx, ny, nz] = shape;
        let mut solver = ViscoacousticMemorySolver::new(
            crate::forward::viscoacoustic::ViscoacousticGrid::new(
                nx, ny, nz, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-8,
            ),
            1_000.0,
            2.25e9,
            &[],
        )
        .expect("test solver parameters are valid");
        let fft = solver.fft.clone();
        for axis in 0..3 {
            let k = fft_wavenumbers(shape[axis], 1.0e-4);
            let field = Array3::from_shape_fn((nx, ny, nz), |[x, y, z]| {
                ((17 * x + 11 * y + 5 * z + axis) as f64 * 0.37).sin()
            });
            let mut reference_scratch = Array3::zeros((nx, ny, nz));
            let mut expected = Array3::from_elem((nx, ny, nz), f64::NAN);
            reference_axis_derivative(
                &fft,
                &k,
                axis,
                &field,
                &mut reference_scratch,
                &mut expected,
            );

            let mut actual = Array3::from_elem((nx, ny, nz), f64::NAN);
            ViscoacousticMemorySolver::axis_derivative(
                &fft,
                &k,
                axis,
                &field,
                &mut solver.cbuf,
                &mut actual,
            );

            assert!(
                expected.iter().any(|value| *value != 0.0),
                "shape {shape:?} axis {axis}: the reference derivative is non-trivial"
            );
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "shape {shape:?} axis {axis}"
                );
            }
            assert_complex_bits_eq(&solver.cbuf, &reference_scratch);
        }
    }
}

#[test]
fn singleton_axes_match_fft_reference_without_touching_scratch() {
    check_singleton_axis([8, 1, 1], 1);
    check_singleton_axis([8, 1, 1], 2);
    check_singleton_axis([4, 3, 1], 2);
}

#[test]
fn singleton_axis_isolates_non_finite_samples_without_touching_scratch() {
    let mut solver = ViscoacousticMemorySolver::new(
        crate::forward::viscoacoustic::ViscoacousticGrid::new(
            4, 1, 1, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-8,
        ),
        1_000.0,
        2.25e9,
        &[],
    )
    .expect("test solver parameters are valid");
    let field = Array3::from_shape_fn((4, 1, 1), |[x, _, _]| match x {
        0 => f64::NAN,
        1 => f64::INFINITY,
        2 => f64::NEG_INFINITY,
        _ => -0.0,
    });
    solver.cbuf.fill(Complex64::new(7.0, -11.0));
    let scratch_before = solver.cbuf.clone();
    let mut derivative = Array3::from_elem((4, 1, 1), f64::NAN);

    ViscoacousticMemorySolver::axis_derivative(
        &solver.fft,
        &fft_wavenumbers(1, 1.0e-4),
        1,
        &field,
        &mut solver.cbuf,
        &mut derivative,
    );

    for value in &derivative {
        assert_eq!(value.to_bits(), 0.0_f64.to_bits());
    }
    assert_complex_bits_eq(&solver.cbuf, &scratch_before);
}
