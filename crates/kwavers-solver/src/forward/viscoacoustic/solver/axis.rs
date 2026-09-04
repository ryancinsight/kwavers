//! Allocation-free pseudospectral derivatives for viscoacoustic stepping.

use super::ViscoacousticMemorySolver;
use kwavers_math::fft::{
    fft_3d_axis_complex_inplace, ifft_3d_axis_complex_inplace, Complex64, Fft3d,
};
use leto::Array3;

/// Build a full-storage reference solver at exactly the given grid.
///
/// The production constructors derive the storage mask from the grid and omit
/// inactive axes; the differential tests need the former six-array layout, so
/// this injects the all-active mask at the same grid shape through the same
/// assembler. The step code is identical under both layouts — an inactive-axis
/// update is the exact positive-zero identity — which is what makes the two
/// bitwise comparable. Test-local by construction, not a public surface.
#[cfg(test)]
pub(super) fn reference_solver(
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    dt: f64,
) -> ViscoacousticMemorySolver {
    let shape = (nx, ny, nz);
    let inv_rho = Array3::from_elem(shape, 1.0 / 1_000.0);
    let m_inf = Array3::from_elem(shape, 2.25e9);
    let arm_fields: Vec<super::Arm> = [(1.5e8, 3.2e-7), (8.0e7, 8.0e-8)]
        .iter()
        .map(|&(dm, tau)| {
            super::build_arm(
                &Array3::from_elem(shape, dm),
                &Array3::from_elem(shape, tau),
                dt,
            )
        })
        .collect();
    ViscoacousticMemorySolver::assemble(
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        dt,
        inv_rho,
        m_inf,
        arm_fields,
        super::ActiveAxes {
            x: true,
            y: true,
            z: true,
        },
    )
}

impl ViscoacousticMemorySolver {
    /// Compute `∂field/∂xₐ → out` with a retained per-axis FFT plan and scratch.
    ///
    /// A singleton axis has no spatial variation, so its derivative is exactly
    /// positive zero. That path clears `out` without reading `field`, touching
    /// the complex scratch, or dispatching either transform. For finite fields
    /// this is bitwise equal to the former FFT route; non-finite samples cannot
    /// contaminate a derivative along an inactive axis.
    pub(super) fn axis_derivative(
        fft: &Fft3d,
        k: &[f64],
        axis: usize,
        field: &Array3<f64>,
        cbuf: &mut Array3<Complex64>,
        out: &mut Array3<f64>,
    ) {
        let [nx, ny, nz] = cbuf.shape();
        assert_eq!(
            field.shape(),
            [nx, ny, nz],
            "invariant: viscoacoustic FFT scratch shape matches input field"
        );
        assert_eq!(
            out.shape(),
            [nx, ny, nz],
            "invariant: viscoacoustic derivative output shape matches input field"
        );
        let axis_len = match axis {
            0 => nx,
            1 => ny,
            2 => nz,
            _ => unreachable!("invariant: derivative axis is 0, 1, or 2"),
        };
        assert_eq!(
            k.len(),
            axis_len,
            "invariant: derivative wavenumbers match the selected axis"
        );
        if axis_len == 1 {
            out.fill(0.0);
            return;
        }

        if let (Some(dst), Some(src)) = (cbuf.as_slice_mut(), field.as_slice()) {
            for (dst, &src) in dst.iter_mut().zip(src) {
                *dst = Complex64::new(src, 0.0);
            }
        } else {
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        cbuf[[x, y, z]] = Complex64::new(field[[x, y, z]], 0.0);
                    }
                }
            }
        }
        fft_3d_axis_complex_inplace(fft, cbuf, axis);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let mode = match axis {
                        0 => x,
                        1 => y,
                        2 => z,
                        _ => unreachable!("invariant: derivative axis is 0, 1, or 2"),
                    };
                    cbuf[[x, y, z]] *= Complex64::new(0.0, k[mode]);
                }
            }
        }
        ifft_3d_axis_complex_inplace(fft, cbuf, axis);
        if let (Some(dst), Some(src)) = (out.as_slice_mut(), cbuf.as_slice()) {
            for (dst, src) in dst.iter_mut().zip(src) {
                *dst = src.re;
            }
        } else {
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        out[[x, y, z]] = cbuf[[x, y, z]].re;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            nx,
            ny,
            nz,
            1.0e-4,
            1.0e-4,
            1.0e-4,
            1.0e-8,
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
        let k = super::super::fft_wavenumbers(axis_len, 1.0e-4);

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
                assert_eq!(actual.to_bits(), expected.to_bits());
                assert_eq!(actual.to_bits(), 0.0_f64.to_bits());
            }
            assert_complex_bits_eq(&solver.cbuf, &scratch_before);
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
            4,
            1,
            1,
            1.0e-4,
            1.0e-4,
            1.0e-4,
            1.0e-8,
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
            &super::super::fft_wavenumbers(1, 1.0e-4),
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

    /// An inactive axis owns no storage: the state arrays of every singleton
    /// axis are the empty `(0,0,0)` staging and the wavenumber vector is
    /// empty, while every active axis and every grid-shaped field is intact.
    #[test]
    fn storage_omits_inactive_axes() {
        let solver_1d = ViscoacousticMemorySolver::new(
            8,
            1,
            1,
            1.0e-4,
            1.0e-4,
            1.0e-4,
            1.0e-8,
            1_000.0,
            2.25e9,
            &[],
        )
        .expect("valid 1-D solver");
        assert_eq!(solver_1d.vx.shape(), [8, 1, 1]);
        assert_eq!(solver_1d.gx.shape(), [8, 1, 1]);
        assert_eq!(solver_1d.gy.shape(), [8, 1, 1]);
        assert!(solver_1d.vy.is_empty() && solver_1d.vz.is_empty());
        assert!(solver_1d.gz.is_empty());
        assert!(solver_1d.ky.is_empty() && solver_1d.kz.is_empty());
        assert_eq!(solver_1d.kx.len(), 8);

        let solver_2d = ViscoacousticMemorySolver::new(
            8,
            4,
            1,
            1.0e-4,
            1.0e-4,
            1.0e-4,
            1.0e-8,
            1_000.0,
            2.25e9,
            &[],
        )
        .expect("valid 2-D solver");
        assert_eq!(solver_2d.vy.shape(), [8, 4, 1]);
        assert!(solver_2d.vz.is_empty() && solver_2d.gz.is_empty());
        assert!(solver_2d.kz.is_empty());

        let solver_3d = ViscoacousticMemorySolver::new(
            4,
            4,
            4,
            1.0e-4,
            1.0e-4,
            1.0e-4,
            1.0e-8,
            1_000.0,
            2.25e9,
            &[],
        )
        .expect("valid 3-D solver");
        for field in [&solver_3d.vx, &solver_3d.vy, &solver_3d.vz] {
            assert_eq!(field.shape(), [4, 4, 4]);
        }
        assert_eq!(
            (solver_3d.kx.len(), solver_3d.ky.len(), solver_3d.kz.len()),
            (4, 4, 4)
        );
    }

    /// The active-axis mask of the canonical grids matches the grid shape
    /// exactly, including every singleton permutation.
    #[test]
    fn active_axes_mask_matches_grid() {
        let cases: [([usize; 3], [bool; 3]); 8] = [
            ([8, 1, 1], [true, false, false]),
            ([1, 8, 1], [false, true, false]),
            ([1, 1, 8], [false, false, true]),
            ([8, 8, 1], [true, true, false]),
            ([8, 1, 8], [true, false, true]),
            ([1, 8, 8], [false, true, true]),
            ([8, 8, 8], [true, true, true]),
            ([1, 1, 1], [false, false, false]),
        ];
        for (shape, mask) in cases {
            let solver = ViscoacousticMemorySolver::new(
                shape[0],
                shape[1],
                shape[2],
                1.0e-4,
                1.0e-4,
                1.0e-4,
                1.0e-8,
                1_000.0,
                2.25e9,
                &[],
            )
            .expect("valid solver for canonical shape");
            assert_eq!(solver.axes.x, mask[0], "x at {shape:?}");
            assert_eq!(solver.axes.y, mask[1], "y at {shape:?}");
            assert_eq!(solver.axes.z, mask[2], "z at {shape:?}");
        }
    }

    /// Stepping an all-singleton grid exercises the no-active-axis path:
    /// the divergence is the exact zero fill and the pressure stays
    /// bitwise-constant (no source, no damping).
    #[test]
    fn step_without_active_axes_keeps_pressure_bitwise_constant() {
        let mut solver = ViscoacousticMemorySolver::new(
            1,
            1,
            1,
            1.0e-4,
            1.0e-4,
            1.0e-4,
            1.0e-8,
            1_000.0,
            2.25e9,
            &[(1.5e8, 3.2e-7)],
        )
        .expect("valid point solver");
        let initial = Array3::from_elem((1, 1, 1), 3.5);
        solver.set_pressure(&initial).expect("matching shape");
        let p0 = solver.pressure()[[0, 0, 0]].to_bits();
        let e0 = solver.energy();
        for _ in 0..16 {
            solver.step();
        }
        assert_eq!(solver.pressure()[[0, 0, 0]].to_bits(), p0);
        assert_eq!(solver.energy().to_bits(), e0.to_bits());
    }

    /// The eight active-axis masks: each candidate is built at the grid that
    /// realizes the mask, against a full-storage reference at the **same**
    /// grid with the mask injected all-active. The step code is identical
    /// under both layouts — an inactive-axis update is the exact positive-zero
    /// identity (zero velocity, zero derivative, `+0` accumulation) — so every
    /// sample of pressure and the conserved energy must agree bitwise.
    const MASK_CASES: [[usize; 3]; 8] = [
        [16, 1, 1],
        [1, 16, 1],
        [1, 1, 16],
        [8, 8, 1],
        [8, 1, 8],
        [1, 8, 8],
        [8, 8, 8],
        [1, 1, 1],
    ];

    /// Seed standing along every active axis (constant across the singleton
    /// axes) so each active velocity component and each active derivative is
    /// exercised, and both solvers see the identical field.
    fn seed(shape: [usize; 3]) -> Array3<f64> {
        let [nx, ny, nz] = shape;
        let span = nx.max(ny).max(nz);
        Array3::from_shape_fn((nx, ny, nz), |[x, y, z]| {
            let phase = if nx > 1 { x as f64 } else { 0.0 }
                + if ny > 1 { y as f64 } else { 0.0 }
                + if nz > 1 { z as f64 } else { 0.0 };
            (std::f64::consts::TAU * 3.0 * phase / span as f64).cos()
        })
    }

    #[test]
    fn every_active_axis_mask_matches_the_full_storage_reference() {
        const DT: f64 = 1.0e-8;
        const WARMUP: usize = 32;
        const SPAN: usize = 48;
        const SAMPLE_POINTS: [(usize, usize, usize); 4] =
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 2, 2)];

        for shape in &MASK_CASES {
            let [nx, ny, nz] = *shape;
            let (dx, dy, dz) = (1.0e-4_f64, 1.0e-4, 1.0e-4);
            let mut candidate = ViscoacousticMemorySolver::new(
                nx,
                ny,
                nz,
                dx,
                dy,
                dz,
                DT,
                1_000.0,
                2.25e9,
                &[(1.5e8, 3.2e-7), (8.0e7, 8.0e-8)],
            )
            .expect("candidate solver parameters are valid");
            let mut reference = reference_solver(nx, ny, nz, dx, dy, dz, DT);

            let seed = seed(*shape);
            candidate.set_pressure(&seed).expect("candidate seed");
            reference.set_pressure(&seed).expect("reference seed");
            let thickness = nx.max(ny).max(nz) / 4;
            candidate.enable_absorbing_layer(thickness, 2.0e6);
            reference.enable_absorbing_layer(thickness, 2.0e6);

            // Fixed interior sample points, clamped to the grid.
            let points: Vec<[usize; 3]> = SAMPLE_POINTS
                .iter()
                .map(|&(i, j, k)| [i.min(nx - 1), j.min(ny - 1), k.min(nz - 1)])
                .collect();
            let samples = |s: &ViscoacousticMemorySolver| -> Vec<u64> {
                points
                    .iter()
                    .map(|&[i, j, k]| s.pressure()[[i, j, k]].to_bits())
                    .chain(std::iter::once(s.energy().to_bits()))
                    .collect()
            };

            for _ in 0..WARMUP {
                candidate.step();
                reference.step();
            }
            assert_eq!(
                samples(&candidate),
                samples(&reference),
                "pressure/energy diverges from the full-storage reference at {shape:?} after warmup"
            );

            let first_pass: Vec<Vec<u64>> = (0..SPAN)
                .map(|_| {
                    candidate.step();
                    samples(&candidate)
                })
                .collect();
            for _ in 0..SPAN {
                reference.step();
            }
            assert_eq!(
                samples(&candidate),
                samples(&reference),
                "pressure/energy diverges from the full-storage reference at {shape:?} after the first pass"
            );

            // Reset plus repeated stepping reproduces its own trace bitwise.
            candidate.set_pressure(&seed).expect("reset seed");
            for _ in 0..WARMUP {
                candidate.step();
            }
            let second_pass: Vec<Vec<u64>> = (0..SPAN)
                .map(|_| {
                    candidate.step();
                    samples(&candidate)
                })
                .collect();
            assert_eq!(
                first_pass, second_pass,
                "reset-plus-repeat is not bitwise-stable at {shape:?}"
            );
        }
    }
}
