//! Acoustic energy conservation checks.

use kwavers_grid::Grid;
use leto::Array3;
use moirai_parallel::{reduce_index_with, Sequential};

/// Compute total acoustic energy and relative error against `initial_energy`.
///
/// # Panics
///
/// Panics when any supplied field has a shape different from `pressure`; equal
/// shapes are required for the pointwise energy reduction.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn validate_energy_conservation(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    density: &Array3<f64>,
    sound_speed: &Array3<f64>,
    initial_energy: f64,
    grid: &Grid,
) -> f64 {
    let shape = pressure.shape();
    assert_eq!(
        velocity_x.shape(),
        shape,
        "invariant: acoustic energy velocity_x shape mismatch"
    );
    assert_eq!(
        velocity_y.shape(),
        shape,
        "invariant: acoustic energy velocity_y shape mismatch"
    );
    assert_eq!(
        velocity_z.shape(),
        shape,
        "invariant: acoustic energy velocity_z shape mismatch"
    );
    assert_eq!(
        density.shape(),
        shape,
        "invariant: acoustic energy density shape mismatch"
    );
    assert_eq!(
        sound_speed.shape(),
        shape,
        "invariant: acoustic energy sound_speed shape mismatch"
    );

    let dv = grid.dx * grid.dy * grid.dz;

    // Each array is checked for its own C-contiguity; the fast path pairs them
    // by raw flat index, which is only valid when every array shares the same
    // (row-major) layout. Mixed layouts across these six independently-supplied
    // arguments fall through to the logical `.iter()` pairing below, which is
    // correct regardless of each array's storage order.
    let total_energy = match (
        pressure.as_slice(),
        velocity_x.as_slice(),
        velocity_y.as_slice(),
        velocity_z.as_slice(),
        density.as_slice(),
        sound_speed.as_slice(),
    ) {
        (
            Some(pressure),
            Some(velocity_x),
            Some(velocity_y),
            Some(velocity_z),
            Some(density),
            Some(sound_speed),
        ) => reduce_index_with::<Sequential, _, _, _>(
            pressure.len(),
            0.0_f64,
            |idx| {
                acoustic_cell_energy(
                    pressure[idx],
                    [velocity_x[idx], velocity_y[idx], velocity_z[idx]],
                    density[idx],
                    sound_speed[idx],
                    dv,
                )
            },
            |left, right| left + right,
        ),
        _ => pressure
            .iter()
            .zip(velocity_x.iter())
            .zip(velocity_y.iter())
            .zip(velocity_z.iter())
            .zip(density.iter())
            .zip(sound_speed.iter())
            .fold(0.0_f64, |acc, (((((p, vx), vy), vz), rho), c)| {
                acc + acoustic_cell_energy(*p, [*vx, *vy, *vz], *rho, *c, dv)
            }),
    };

    (total_energy - initial_energy).abs() / initial_energy.max(1e-10)
}

#[inline]
fn acoustic_cell_energy(p: f64, velocity: [f64; 3], rho: f64, c: f64, dv: f64) -> f64 {
    if rho > 0.0 && c > 0.0 {
        let [vx, vy, vz] = velocity;
        let kinetic = 0.5 * rho * vz.mul_add(vz, vx.mul_add(vx, vy * vy));
        let potential = super::acoustic_potential_energy_density(p, rho, c);
        (kinetic + potential) * dv
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use kwavers_grid::Grid;
    use leto::Array3;

    type AcousticFields = (
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
    );

    fn small_grid() -> Grid {
        Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).unwrap()
    }

    fn make_fields(p: f64, v: f64, rho: f64, c: f64) -> AcousticFields {
        let s: [usize; 3] = [4, 4, 4];
        let pressure = Array3::from_elem(s, p);
        let velocity_x = Array3::from_elem(s, v);
        let velocity_y = Array3::zeros(s);
        let velocity_z = Array3::zeros(s);
        let density = Array3::from_elem(s, rho);
        let sound_speed = Array3::from_elem(s, c);
        (
            pressure,
            velocity_x,
            velocity_y,
            velocity_z,
            density,
            sound_speed,
        )
    }

    /// Builds an owned F-contiguous `Array3<f64>` directly through leto's
    /// public `Layout`/`VecStorage`/`Array` constructors: `as_slice()` (the
    /// C-contiguity check) returns `None` while `as_slice_memory_order()`
    /// would have returned `Some` — the exact layout a caller's
    /// `.transpose([2, 1, 0])` could hand this function.
    fn f_ordered_from_fn(shape: [usize; 3], f: impl Fn(usize, usize, usize) -> f64) -> Array3<f64> {
        let [nx, ny, nz] = shape;
        let layout = leto::Layout::f_contiguous(shape).expect("invariant: nonzero shape");
        // `VecStorage::generate` calls its `FnMut` sequentially for positions
        // 0..len, so a captured counter reconstructs the F-order flat index.
        let mut p = 0usize;
        let storage = leto::VecStorage::generate(nx * ny * nz, || {
            let i = p % nx;
            let j = (p / nx) % ny;
            let k = p / (nx * ny);
            p += 1;
            f(i, j, k)
        });
        leto::Array::new(layout, storage).expect("invariant: layout fits storage")
    }

    /// A field supplied with F-contiguous storage must be paired with the
    /// other five fields by *logical* index, not by raw memory-order flat
    /// index — that raw pairing is only valid when every one of the six
    /// independently-supplied fields shares the same layout, which nothing
    /// forces for a public five-argument-plus-pressure API.
    ///
    /// `pressure` and `density` both vary spatially and combine nonlinearly
    /// in `potential = p² / (2ρc²)`; velocities are zero so kinetic energy
    /// cannot mask a wrong pairing. Because `density` is F-contiguous while
    /// `pressure` stays C-contiguous, a raw-memory-order pairing (`p[idx]`
    /// against `density`'s F-order-flattened `idx`) combines mismatched
    /// physical cells — unlike a same-shaped sum of squares or a sum over a
    /// single array, a ratio of two independently-permuted arrays does not
    /// generally sum to the same total (e.g. `1/1 + 4/4 = 2` vs the swapped
    /// `1/4 + 4/1 = 4.25`), so a wrong pairing changes the total measurably.
    #[test]
    fn energy_conservation_pairs_a_transposed_field_by_logical_index() {
        let grid = small_grid();
        let shape = [4usize, 4, 4];
        let c = SOUND_SPEED_WATER_SIM;
        let p_at =
            |i: usize, j: usize, k: usize| 1.0 + i as f64 + 10.0 * j as f64 + 100.0 * k as f64;
        let rho_at = |i: usize, j: usize, k: usize| {
            DENSITY_WATER_NOMINAL * (1.0 + 0.01 * (i as f64 + 10.0 * j as f64 + 100.0 * k as f64))
        };

        let pressure = Array3::from_shape_fn(shape, |[i, j, k]| p_at(i, j, k));
        let density = f_ordered_from_fn(shape, rho_at);
        assert!(
            density.as_slice().is_none() && density.as_slice_memory_order().is_some(),
            "density must be dense in F order for this case to mean anything"
        );
        let velocity_x = Array3::zeros(shape);
        let velocity_y = Array3::zeros(shape);
        let velocity_z = Array3::zeros(shape);
        let sound_speed = Array3::from_elem(shape, c);

        let dv = grid.dx * grid.dy * grid.dz;
        let mut expected = 0.0_f64;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    expected += acoustic_cell_energy(
                        p_at(i, j, k),
                        [0.0, 0.0, 0.0],
                        rho_at(i, j, k),
                        c,
                        dv,
                    );
                }
            }
        }

        let error = validate_energy_conservation(
            &pressure,
            &velocity_x,
            &velocity_y,
            &velocity_z,
            &density,
            &sound_speed,
            expected,
            &grid,
        );
        assert!(
            error.abs() < 1e-9,
            "pairing by raw memory order instead of logical index would not equal \
             the direct logical-index sum (relative error={error:.3e})"
        );
    }

    /// When `initial_energy` equals the computed total, relative error = 0.
    #[test]
    fn energy_conservation_error_zero_when_initial_matches_computed() {
        let grid = small_grid();
        let (p, vx, vy, vz, rho, c) =
            make_fields(100.0, 0.1, DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM);

        // Compute what the function computes first, then pass it as initial
        // so the error is exactly 0.
        let dv = grid.dx * grid.dy * grid.dz;
        let p_val = 100.0_f64;
        let v_val = 0.1_f64;
        let rho_val = DENSITY_WATER_NOMINAL;
        let c_val = SOUND_SPEED_WATER_SIM;
        let kinetic = 0.5 * rho_val * v_val * v_val;
        let potential = super::super::acoustic_potential_energy_density(p_val, rho_val, c_val);
        let cell_energy = (kinetic + potential) * dv;
        let initial_energy = cell_energy * (4.0_f64).powi(3);

        let error =
            validate_energy_conservation(&p, &vx, &vy, &vz, &rho, &c, initial_energy, &grid);
        assert!(error.abs() < 1e-12, "error must be 0 (got {error:.3e})");
    }

    /// All-zero pressure/velocity: total energy = 0; error = initial/(max(initial, 1e-10)).
    /// With initial=0: denominator = 1e-10, numerator = 0 → error = 0.
    #[test]
    fn energy_conservation_zero_error_for_zero_fields_and_zero_initial() {
        let grid = small_grid();
        let (p, vx, vy, vz, rho, c) =
            make_fields(0.0, 0.0, DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM);
        let error = validate_energy_conservation(&p, &vx, &vy, &vz, &rho, &c, 0.0, &grid);
        assert_eq!(
            error, 0.0,
            "zero fields with zero initial energy must give 0 error"
        );
    }
}
