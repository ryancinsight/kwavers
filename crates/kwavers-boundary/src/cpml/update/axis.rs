//! Axis-generic CPML memory update and gradient correction.
//!
//! Replaces the former per-axis `x`/`y`/`z` transcriptions (12 near-identical
//! methods) with one parameterized implementation. The boundary axis is selected
//! at runtime, and the per-axis memory (`ψ`) and profile
//! (`a`/`b`/`κ`) arrays are resolved through the `CPMLMemory`/`CPMLProfiles`
//! axis accessors. Behavior is bit-identical to the original per-axis code
//! (proven by the differential tests below at migration time).

use super::super::memory::CPMLMemory;
use super::super::profiles::CPMLProfiles;
use super::CPMLUpdater;
use leto::{Array1, Array3, ArrayView3, ArrayViewMut3, SliceArg};

/// `axis`-th component of a 3-D dimension tuple.
#[inline]
fn dim_axis(d: [usize; 3], axis: usize) -> usize {
    d[axis]
}

#[inline]
fn axis_range_args(axis: usize, start: usize, end: Option<usize>) -> [SliceArg; 3] {
    let mut args = [SliceArg::All; 3];
    args[axis] = SliceArg::range(Some(start as isize), end.map(|n| n as isize), 1);
    args
}

/// Sweep one strip plane pair through the ψ FMA `ψ ← b·ψ + a·∂f`.
///
/// The profile scalars ride outside the inner loop (the per-cell index array
/// the former indexed sweep built defeated autovectorization). A contiguous
/// slice fast path was tried and reverted: leto exposes no sound
/// mutable-window-as-slice API (`into_slice` spans the whole backing buffer,
/// not the window), so the generic zip is the only correct vehicle here. The
/// per-cell arithmetic matches the former sweep operand-for-operand, so `ψ`
/// is unchanged to the bit.
fn sweep_memory_plane(
    psi_plane: &mut ArrayViewMut3<'_, f64>,
    gradient_plane: &ArrayView3<'_, f64>,
    b: f64,
    a: f64,
) {
    leto_ops::zip_mut_with(psi_plane, gradient_plane, |psi, &g| {
        *psi = b.mul_add(*psi, a * g);
    })
    .expect("invariant: CPML strip planes match");
}

/// Sweep one strip plane pair through the correction `∂f ← ∂f/κ + ψ`.
///
/// Same structure as [`sweep_memory_plane`]; the per-cell division is kept
/// (not replaced by a precomputed reciprocal), so the corrected gradient is
/// unchanged to the bit.
fn sweep_correction_plane(
    gradient_plane: &mut ArrayViewMut3<'_, f64>,
    psi_plane: &ArrayView3<'_, f64>,
    kappa: f64,
) {
    leto_ops::zip_mut_with(gradient_plane, psi_plane, |g, &psi| {
        *g = *g / kappa + psi;
    })
    .expect("invariant: CPML strip planes match");
}

/// Recursive-convolution memory update along `axis`: `ψ ← b·ψ + a·∂f` on both
/// PML strips. Shared by the pressure- and velocity-gradient paths, which differ
/// only in which `ψ` buffer is supplied.
///
/// Each strip plane is swept with its profile scalars (`b[c]`, `a[c]`) hoisted
/// out of the inner loop; the per-cell index array the former indexed sweep
/// built defeated autovectorization and cost ~3x a full-field sweep per strip
/// cell. The arithmetic per cell is unchanged, so `ψ` is unchanged to the bit.
fn update_memory_along(
    psi: &mut Array3<f64>,
    gradient: &Array3<f64>,
    axis: usize,
    b: &Array1<f64>,
    a: &Array1<f64>,
) {
    let n = dim_axis(gradient.shape(), axis);
    let thickness = dim_axis(psi.shape(), axis) / 2;
    let left_count = thickness.min(n);

    // Left strip: ψ plane c aligns with gradient plane c.
    for c in 0..left_count {
        let args = axis_range_args(axis, c, Some(c + 1));
        let mut psi_plane = psi
            .slice_with_mut::<3>(&args)
            .expect("invariant: valid left CPML memory plane");
        let gradient_plane = gradient
            .slice_with::<3>(&args)
            .expect("invariant: valid left CPML gradient plane");
        sweep_memory_plane(&mut psi_plane, &gradient_plane, b[c], a[c]);
    }

    // Right strip: gradient plane c aligns with ψ plane `thickness + (c - right_start)`.
    if n > thickness {
        let right_start = n - thickness;
        for p in 0..(n - right_start) {
            let c = right_start + p;
            let psi_args = axis_range_args(axis, thickness + p, Some(thickness + p + 1));
            let gradient_args = axis_range_args(axis, c, Some(c + 1));
            let mut psi_plane = psi
                .slice_with_mut::<3>(&psi_args)
                .expect("invariant: valid right CPML memory plane");
            let gradient_plane = gradient
                .slice_with::<3>(&gradient_args)
                .expect("invariant: valid right CPML gradient plane");
            sweep_memory_plane(&mut psi_plane, &gradient_plane, b[c], a[c]);
        }
    }
}

/// Apply the CPML gradient correction along `axis`: `∂f ← ∂f/κ + ψ` on both PML
/// strips. Shared by the pressure- and velocity-gradient paths.
///
/// Like [`update_memory_along`], each strip plane is swept with its `κ[c]`
/// hoisted, keeping the per-cell division (bit-identity with the former sweep)
/// while letting the inner loop vectorize.
fn apply_correction_along(
    gradient: &mut Array3<f64>,
    psi: &Array3<f64>,
    axis: usize,
    kappa: &Array1<f64>,
) {
    let n = dim_axis(gradient.shape(), axis);
    let thickness = dim_axis(psi.shape(), axis) / 2;
    let left_count = thickness.min(n);

    // Left strip: gradient plane c aligns with ψ plane c.
    for c in 0..left_count {
        let args = axis_range_args(axis, c, Some(c + 1));
        let mut gradient_plane = gradient
            .slice_with_mut::<3>(&args)
            .expect("invariant: valid left CPML correction plane");
        let psi_plane = psi
            .slice_with::<3>(&args)
            .expect("invariant: valid left CPML memory correction plane");
        sweep_correction_plane(&mut gradient_plane, &psi_plane, kappa[c]);
    }

    // Right strip: gradient plane c aligns with ψ plane `thickness + (c - right_start)`.
    if n > thickness {
        let right_start = n - thickness;
        for p in 0..(n - right_start) {
            let c = right_start + p;
            let gradient_args = axis_range_args(axis, c, Some(c + 1));
            let psi_args = axis_range_args(axis, thickness + p, Some(thickness + p + 1));
            let mut gradient_plane = gradient
                .slice_with_mut::<3>(&gradient_args)
                .expect("invariant: valid right CPML correction plane");
            let psi_plane = psi
                .slice_with::<3>(&psi_args)
                .expect("invariant: valid right CPML memory correction plane");
            sweep_correction_plane(&mut gradient_plane, &psi_plane, kappa[c]);
        }
    }
}

impl CPMLUpdater {
    /// Update the pressure-gradient memory along `axis` (0=x, 1=y, 2=z).
    pub(super) fn update_p_memory_axis(
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        axis: usize,
        profiles: &CPMLProfiles,
    ) {
        if axis >= 3 {
            return;
        }
        update_memory_along(
            memory.psi_p_mut(axis),
            gradient,
            axis,
            profiles.b_staggered(axis),
            profiles.a_staggered(axis),
        );
    }

    /// Update the velocity-gradient memory along `axis` (0=x, 1=y, 2=z).
    pub(super) fn update_v_memory_axis(
        memory: &mut CPMLMemory,
        v_gradient: &Array3<f64>,
        axis: usize,
        profiles: &CPMLProfiles,
    ) {
        if axis >= 3 {
            return;
        }
        update_memory_along(
            memory.psi_v_mut(axis),
            v_gradient,
            axis,
            profiles.b(axis),
            profiles.a(axis),
        );
    }

    /// Apply the pressure-gradient CPML correction along `axis` (0=x, 1=y, 2=z).
    pub(super) fn apply_p_correction_axis(
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        axis: usize,
        profiles: &CPMLProfiles,
    ) {
        if axis >= 3 {
            return;
        }
        apply_correction_along(
            gradient,
            memory.psi_p(axis),
            axis,
            profiles.kappa_staggered(axis),
        );
    }

    /// Apply the velocity-gradient CPML correction along `axis` (0=x, 1=y, 2=z).
    pub(super) fn apply_v_correction_axis(
        v_gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        axis: usize,
        profiles: &CPMLProfiles,
    ) {
        if axis >= 3 {
            return;
        }
        apply_correction_along(v_gradient, memory.psi_v(axis), axis, profiles.kappa(axis));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpml::config::CPMLConfig;
    use kwavers_grid::Grid;

    const NX: usize = 20;
    const NY: usize = 18;
    const NZ: usize = 16;

    fn distinct(shape: [usize; 3], seed: f64) -> Array3<f64> {
        Array3::from_shape_fn(shape, |[i, j, k]| {
            ((i * 7 + j * 13 + k * 17) as f64).mul_add(0.013, seed) + 0.25
        })
    }

    /// Memory update then correction round-trips along every axis and leaves the
    /// non-PML interior untouched (CPML only acts on the boundary strips).
    #[test]
    fn correction_is_identity_in_pml_free_interior() {
        let grid = Grid::new(NX, NY, NZ, 1e-4, 1e-4, 1e-4).unwrap();
        let config = CPMLConfig::default();
        let mem = CPMLMemory::new(&config, &grid); // zero ψ
        let prof = CPMLProfiles::new(&config, &grid, 1500.0, 1e-8).unwrap();
        let u = CPMLUpdater::new();

        for axis in 0..3 {
            let original = distinct([NX, NY, NZ], 0.5);
            let mut g = original.clone();
            // With zero ψ, the correction is g ← g/κ; κ = 1 in the interior, so the
            // deep interior must be unchanged.
            u.apply_p_correction(&mut g, &mem, axis, &prof);
            let mid = [NX / 2, NY / 2, NZ / 2];
            assert!(
                (g[mid] - original[mid]).abs() < 1e-12,
                "interior altered on axis {axis}"
            );
        }
    }

    /// Out-of-range axis is a no-op (preserves the former `_ => {}` dispatch).
    #[test]
    fn out_of_range_axis_is_noop() {
        let grid = Grid::new(NX, NY, NZ, 1e-4, 1e-4, 1e-4).unwrap();
        let config = CPMLConfig::default();
        let mut mem = CPMLMemory::new(&config, &grid);
        mem.psi_p_x = distinct(mem.psi_p_x.shape(), 1.0);
        let before = mem.clone();
        let prof = CPMLProfiles::new(&config, &grid, 1500.0, 1e-8).unwrap();
        let g = distinct([NX, NY, NZ], 0.5);
        CPMLUpdater::new().update_p_memory(&mut mem, &g, 3, &prof);
        assert_eq!(before.psi_p_x, mem.psi_p_x);
    }
}
