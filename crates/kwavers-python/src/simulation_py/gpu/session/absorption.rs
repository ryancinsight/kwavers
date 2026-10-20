//! Host power-law absorption operators of the GPU PSTD session.

use kwavers_gpu::pstd_gpu::{power_law_operators, PowerLawOperators};
use kwavers_grid::Grid;
use kwavers_solver::forward::pstd::config::PowerLawDispersion;
use numpy::ndarray::ArrayView3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The session's operators: all zero without an absorption field, otherwise
/// the construction the Rust GPU runner uses, so both entry points accept and
/// reject the same exponents and coefficients.
///
/// `absorption` is dB/(MHz^y cm) per voxel, shaped like `grid`; its logical
/// (C) order is the `[ix][iy][iz]` flat order of `c0_flat`, whatever the
/// array's memory layout.
///
/// # Errors
/// Raises `ValueError` when the field shape differs from the grid or the
/// builder rejects it: a coefficient that is not finite and non-negative, or,
/// where some voxel absorbs, an exponent that is non-finite or `y = 1` with
/// causal dispersion.
pub(super) fn build_absorption_kernels(
    grid: &Grid,
    absorption: Option<ArrayView3<'_, f64>>,
    c0_flat: &[f32],
    alpha_power: f64,
    dispersion: PowerLawDispersion,
) -> PyResult<PowerLawOperators> {
    let Some(field) = absorption else {
        return Ok(PowerLawOperators::lossless(c0_flat.len()));
    };
    let expected = (grid.nx, grid.ny, grid.nz);
    if field.dim() != expected {
        return Err(PyValueError::new_err(format!(
            "absorption shape {:?} does not match the grid {expected:?}",
            field.dim()
        )));
    }
    let coefficients: Vec<f64> = field.iter().copied().collect();
    power_law_operators(grid, alpha_power, dispersion, c0_flat, |flat| {
        coefficients[flat]
    })
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::build_absorption_kernels;
    use eunomia::FloatElement;
    use kwavers_gpu::pstd_gpu::{power_law_operators, PowerLawOperators};
    use kwavers_grid::Grid;
    use kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;
    use kwavers_solver::forward::pstd::config::PowerLawDispersion::{Causal, Omitted};
    use numpy::ndarray::Array3;
    use pyo3::exceptions::PyValueError;
    use pyo3::Python;

    const SHAPE: (usize, usize, usize) = (4, 3, 5);

    fn grid() -> Grid {
        Grid::new(SHAPE.0, SHAPE.1, SHAPE.2, 1.0e-4, 1.5e-4, 2.0e-4).expect("grid")
    }

    /// Heterogeneous coefficients, distinct per voxel so a transposed or
    /// mis-strided mapping changes the result.
    fn field() -> Array3<f64> {
        Array3::from_shape_fn(SHAPE, |(ix, iy, iz)| {
            0.25 + 0.1 * f64::from(u8::try_from(ix + 4 * iy + 12 * iz).expect("small index"))
        })
    }

    fn sound_speed() -> Vec<f32> {
        (0..SHAPE.0 * SHAPE.1 * SHAPE.2)
            .map(|flat| 1400.0 + f32::from(u8::try_from(flat).expect("small index")))
            .collect()
    }

    #[test]
    fn builder_equals_the_shared_construction_for_a_heterogeneous_field() {
        let (grid, field, c0) = (grid(), field(), sound_speed());
        let built =
            build_absorption_kernels(&grid, Some(field.view()), &c0, 1.5, Causal).expect("built");
        // The shared builder reads the field by flat C-order index.
        let flat: Vec<f64> = field.iter().copied().collect();
        let direct =
            power_law_operators(&grid, 1.5, Causal, &c0, |index| flat[index]).expect("direct");
        assert_eq!(built, direct);
        assert!(built.tau.windows(2).any(|pair| pair[0] != pair[1]));
    }

    /// A Fortran-ordered view of the same logical field maps to the same
    /// operators: the mapping follows logical indices, not memory.
    #[test]
    fn memory_layout_does_not_change_the_operators() {
        let (grid, field, c0) = (grid(), field(), sound_speed());
        let transposed_storage = field.t().as_standard_layout().into_owned();
        let fortran_view = transposed_storage.t();
        assert!(!fortran_view.is_standard_layout());
        assert_eq!(fortran_view, field);
        let c_order =
            build_absorption_kernels(&grid, Some(field.view()), &c0, 1.5, Causal).expect("c");
        let fortran =
            build_absorption_kernels(&grid, Some(fortran_view), &c0, 1.5, Causal).expect("f");
        assert_eq!(c_order, fortran);
    }

    /// τ = −2 α₀ c₀^(y−1) = −2 α₀ at y = 1, where c₀^0 = 1 exactly, and
    /// η = 0 without dispersion. α₀ is the shared dB-to-neper conversion of
    /// each voxel's own coefficient, so the f32 store is the same rounding of
    /// the same f64.
    #[test]
    fn unit_exponent_without_dispersion_has_the_closed_form_tau_and_zero_eta() {
        let (grid, field, c0) = (grid(), field(), sound_speed());
        let built =
            build_absorption_kernels(&grid, Some(field.view()), &c0, 1.0, Omitted).expect("built");
        assert!(built.eta.iter().all(|&eta| eta == 0.0));
        for (flat, (&tau, &coefficient)) in built.tau.iter().zip(field.iter()).enumerate() {
            let alpha_0 = power_law_db_cm_to_np_omega_m(coefficient, 1.0);
            assert!(alpha_0 > 0.0);
            assert_eq!(tau, f32::from_f64(-2.0 * alpha_0), "voxel {flat}");
        }
    }

    /// An all-zero field is lossless whatever the exponent, as for a medium
    /// in the Rust runner.
    #[test]
    fn an_all_zero_field_is_lossless_at_a_causal_unit_exponent() {
        let (grid, c0) = (grid(), sound_speed());
        let zeros = Array3::<f64>::zeros(SHAPE);
        let built = build_absorption_kernels(&grid, Some(zeros.view()), &c0, 1.0, Causal)
            .expect("lossless");
        assert_eq!(built, PowerLawOperators::lossless(c0.len()));
    }

    #[test]
    fn unit_exponent_with_causal_dispersion_is_a_value_error() {
        let (grid, field, c0) = (grid(), field(), sound_speed());
        let error = build_absorption_kernels(&grid, Some(field.view()), &c0, 1.0, Causal)
            .expect_err("tan(pi/2) is singular");
        Python::attach(|py| {
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(
                error.to_string().contains("PowerLawDispersion::Omitted"),
                "{error}"
            );
        });
    }

    #[test]
    fn a_mismatched_field_shape_is_a_value_error() {
        let (grid, c0) = (grid(), sound_speed());
        let wrong = Array3::<f64>::ones((SHAPE.0, SHAPE.2, SHAPE.1));
        let error = build_absorption_kernels(&grid, Some(wrong.view()), &c0, 1.5, Causal)
            .expect_err("shape differs");
        Python::attach(|py| {
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(
                error.to_string().contains("does not match the grid"),
                "{error}"
            );
        });
    }

    #[test]
    fn without_an_absorption_field_every_operator_is_zero() {
        let (grid, c0) = (grid(), sound_speed());
        let built = build_absorption_kernels(&grid, None, &c0, 1.0, Causal).expect("lossless");
        let total = c0.len();
        for operator in [&built.nabla1, &built.nabla2, &built.tau, &built.eta] {
            assert_eq!(operator.len(), total);
            assert!(operator.iter().all(|&value| value == 0.0));
        }
    }
}
