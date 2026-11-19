use super::super::super::{build_arm, ActiveAxes, Arm, Coeff, ViscoacousticMemorySolver};
use leto::Array3;

/// Build a full-storage reference solver at exactly the given grid.
///
/// The production constructors derive the storage mask from the grid and omit
/// inactive axes; the differential tests need the former six-array layout, so
/// this injects the all-active mask at the same grid shape through the same
/// assembler. The step code is identical under both layouts — an inactive-axis
/// update is the exact positive-zero identity — which is what makes the two
/// bitwise comparable. Test-local by construction, not a public surface.
fn reference_solver(
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
    let arm_fields: Vec<Arm> = [(1.5e8, 3.2e-7), (8.0e7, 8.0e-8)]
        .iter()
        .map(|&(dm, tau)| {
            build_arm(
                &Array3::from_elem(shape, dm),
                &Array3::from_elem(shape, tau),
                dt,
            )
        })
        .collect();
    ViscoacousticMemorySolver::assemble(
        crate::forward::viscoacoustic::ViscoacousticGrid::new(nx, ny, nz, dx, dy, dz, dt),
        Coeff::Field(inv_rho),
        Coeff::Field(m_inf),
        arm_fields,
        ActiveAxes {
            x: true,
            y: true,
            z: true,
        },
    )
}

/// An inactive axis owns no storage: the state arrays of every singleton
/// axis are the empty `(0,0,0)` staging and the wavenumber vector is
/// empty, while every active axis and every grid-shaped field is intact.
#[test]
fn storage_omits_inactive_axes() {
    let solver_1d = ViscoacousticMemorySolver::new(
        crate::forward::viscoacoustic::ViscoacousticGrid::new(
            8, 1, 1, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-8,
        ),
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
        crate::forward::viscoacoustic::ViscoacousticGrid::new(
            8, 4, 1, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-8,
        ),
        1_000.0,
        2.25e9,
        &[],
    )
    .expect("valid 2-D solver");
    assert_eq!(solver_2d.vy.shape(), [8, 4, 1]);
    assert!(solver_2d.vz.is_empty() && solver_2d.gz.is_empty());
    assert!(solver_2d.kz.is_empty());

    let solver_3d = ViscoacousticMemorySolver::new(
        crate::forward::viscoacoustic::ViscoacousticGrid::new(
            4, 4, 4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-8,
        ),
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
            crate::forward::viscoacoustic::ViscoacousticGrid::new(
                shape[0], shape[1], shape[2], 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-8,
            ),
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
        crate::forward::viscoacoustic::ViscoacousticGrid::new(
            1, 1, 1, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-8,
        ),
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
    const SAMPLE_POINTS: [(usize, usize, usize); 4] = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 2, 2)];

    for shape in &MASK_CASES {
        let [nx, ny, nz] = *shape;
        let (dx, dy, dz) = (1.0e-4_f64, 1.0e-4, 1.0e-4);
        let mut candidate = ViscoacousticMemorySolver::new(
            crate::forward::viscoacoustic::ViscoacousticGrid::new(nx, ny, nz, dx, dy, dz, DT),
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
        candidate
            .enable_absorbing_layer(thickness, 2.0e6)
            .expect("absorbing layer parameters are valid");
        reference
            .enable_absorbing_layer(thickness, 2.0e6)
            .expect("absorbing layer parameters are valid");

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
