//! Dense CBS fixed-point volume solver.

use super::green::{
    apply_shifted_green_operator, shifted_outgoing_green, shifted_wavenumber, GreenOperatorKind,
};
use super::grid::GridSpec;
use super::potential::{convergence_epsilon, pointwise_preconditioner, shifted_potential};
use crate::core::error::{KwaversError, KwaversResult};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

/// CBS fixed-point solver settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CbsConfig {
    pub max_iterations: usize,
    pub relative_tolerance: f64,
}

impl Default for CbsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            relative_tolerance: 1.0e-10,
        }
    }
}

/// CBS solution and convergence diagnostics.
#[derive(Clone, Debug)]
pub struct CbsSolution {
    pub field: Vec<Complex64>,
    pub iterations: usize,
    pub relative_residual: f64,
    pub epsilon: f64,
}

/// Solve `(∇² + k(x)²)u = q` on a uniform grid with a dense CBS Green operator.
///
/// `source_density` is a cell-centered source density. A point source of unit
/// strength at one voxel is represented by `1 / dV` in that voxel.
///
/// # Errors
/// Returns an error if dimensions, source data, potential data, or config values
/// violate the discrete CBS contract.
pub fn solve_volume_field(
    grid: GridSpec,
    reference_wavenumber: f64,
    real_potential: &[f64],
    source_density: &[Complex64],
    config: CbsConfig,
) -> KwaversResult<CbsSolution> {
    solve_volume_field_with_operator(
        grid,
        reference_wavenumber,
        real_potential,
        source_density,
        config,
        GreenOperatorKind::DenseFreeSpace,
    )
}

/// Solve the shifted CBS field with an explicit Green operator discretization.
///
/// # Errors
/// Returns an error if dimensions, source data, potential data, or config values
/// violate the discrete CBS contract.
pub fn solve_volume_field_with_operator(
    grid: GridSpec,
    reference_wavenumber: f64,
    real_potential: &[f64],
    source_density: &[Complex64],
    config: CbsConfig,
    operator: GreenOperatorKind,
) -> KwaversResult<CbsSolution> {
    validate_inputs(
        grid,
        reference_wavenumber,
        real_potential,
        source_density,
        config,
    )?;
    validate_operator(grid, operator)?;
    let epsilon = convergence_epsilon(real_potential)?;
    let shifted = shifted_potential(real_potential, epsilon)?;
    let gamma = pointwise_preconditioner(&shifted, epsilon)?;
    let mut field = vec![Complex64::new(0.0, 0.0); grid.len()];
    let mut relative_residual = f64::INFINITY;

    for iteration in 1..=config.max_iterations {
        let rhs = source_density
            .iter()
            .zip(shifted.iter().zip(field.iter()))
            .map(|(&source, (&potential, &value))| source - potential * value)
            .collect::<Vec<_>>();
        let green_rhs =
            apply_shifted_green_operator(grid, reference_wavenumber, epsilon, &rhs, operator);
        let residual = field
            .iter()
            .zip(green_rhs.iter())
            .map(|(&value, &target)| value - target)
            .collect::<Vec<_>>();
        relative_residual = norm(&residual) / norm(&green_rhs).max(f64::EPSILON);
        for ((value, &preconditioner), &residual_value) in
            field.iter_mut().zip(gamma.iter()).zip(residual.iter())
        {
            *value += preconditioner * residual_value;
        }
        if relative_residual <= config.relative_tolerance {
            return Ok(CbsSolution {
                field,
                iterations: iteration,
                relative_residual,
                epsilon,
            });
        }
    }

    Ok(CbsSolution {
        field,
        iterations: config.max_iterations,
        relative_residual,
        epsilon,
    })
}

/// Solve the Euclidean adjoint of the dense CBS Lippmann-Schwinger system.
///
/// The forward fixed point solves
/// `(I + G_epsilon diag(V - i epsilon)) u = G_epsilon q`. This function solves
/// `(I + G_epsilon diag(V - i epsilon))^H lambda = r` for the receiver-adjoint
/// source `r`. The supplied `epsilon` must be the shift used by the paired
/// forward solve, making the returned field the exact discrete adjoint of that
/// linearized dense operator.
///
/// # Errors
/// Returns an error if the discrete system is invalid or singular.
pub fn solve_adjoint_volume_field(
    grid: GridSpec,
    reference_wavenumber: f64,
    real_potential: &[f64],
    adjoint_rhs: &[Complex64],
    epsilon: f64,
    config: CbsConfig,
) -> KwaversResult<CbsSolution> {
    solve_adjoint_volume_field_with_operator(
        grid,
        reference_wavenumber,
        real_potential,
        adjoint_rhs,
        epsilon,
        config,
        GreenOperatorKind::DenseFreeSpace,
    )
}

/// Solve the Euclidean adjoint for an explicit Green operator discretization.
///
/// # Errors
/// Returns an error if the discrete system is invalid, singular, or fails the
/// requested residual tolerance.
pub fn solve_adjoint_volume_field_with_operator(
    grid: GridSpec,
    reference_wavenumber: f64,
    real_potential: &[f64],
    adjoint_rhs: &[Complex64],
    epsilon: f64,
    config: CbsConfig,
    operator_kind: GreenOperatorKind,
) -> KwaversResult<CbsSolution> {
    validate_inputs(
        grid,
        reference_wavenumber,
        real_potential,
        adjoint_rhs,
        config,
    )?;
    validate_operator(grid, operator_kind)?;
    let shifted = shifted_potential(real_potential, epsilon)?;
    let operator =
        dense_operator_matrix(grid, reference_wavenumber, epsilon, &shifted, operator_kind);
    let adjoint_operator = operator.adjoint();
    let rhs = DVector::from_column_slice(adjoint_rhs);
    let solution = adjoint_operator.clone().lu().solve(&rhs).ok_or_else(|| {
        KwaversError::InvalidInput("dense CBS adjoint operator is singular".to_owned())
    })?;
    let residual = &adjoint_operator * &solution - rhs;
    let relative_residual = residual.norm() / adjoint_rhs_norm(adjoint_rhs).max(f64::EPSILON);
    if !relative_residual.is_finite() || relative_residual > config.relative_tolerance {
        return Err(KwaversError::InvalidInput(format!(
            "dense CBS adjoint residual {} exceeds tolerance {}",
            relative_residual, config.relative_tolerance
        )));
    }

    Ok(CbsSolution {
        field: solution.iter().copied().collect(),
        iterations: 1,
        relative_residual,
        epsilon,
    })
}

fn validate_inputs(
    grid: GridSpec,
    reference_wavenumber: f64,
    real_potential: &[f64],
    source_density: &[Complex64],
    config: CbsConfig,
) -> KwaversResult<()> {
    if !reference_wavenumber.is_finite() || reference_wavenumber <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "CBS reference wavenumber must be positive and finite, got {reference_wavenumber}"
        )));
    }
    if real_potential.len() != grid.len() || source_density.len() != grid.len() {
        return Err(KwaversError::DimensionMismatch(format!(
            "CBS vector length mismatch: grid {}, potential {}, source {}",
            grid.len(),
            real_potential.len(),
            source_density.len()
        )));
    }
    if config.max_iterations == 0 {
        return Err(KwaversError::InvalidInput(
            "CBS max_iterations must be nonzero".to_owned(),
        ));
    }
    if !config.relative_tolerance.is_finite() || config.relative_tolerance <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "CBS relative_tolerance must be positive and finite, got {}",
            config.relative_tolerance
        )));
    }
    if source_density
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(KwaversError::InvalidInput(
            "CBS source density must be finite".to_owned(),
        ));
    }
    Ok(())
}

fn validate_operator(grid: GridSpec, operator: GreenOperatorKind) -> KwaversResult<()> {
    if let GreenOperatorKind::SpectralPeriodic { absorbing_boundary } = operator {
        absorbing_boundary.validate_for_grid(grid)?;
    }
    Ok(())
}

fn dense_operator_matrix(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    shifted_potential: &[Complex64],
    operator: GreenOperatorKind,
) -> DMatrix<Complex64> {
    match operator {
        GreenOperatorKind::DenseFreeSpace => {
            dense_free_space_operator_matrix(grid, reference_wavenumber, epsilon, shifted_potential)
        }
        GreenOperatorKind::SpectralPeriodic { .. } => operator_matrix_by_columns(
            grid,
            reference_wavenumber,
            epsilon,
            shifted_potential,
            operator,
        ),
    }
}

fn dense_free_space_operator_matrix(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    shifted_potential: &[Complex64],
) -> DMatrix<Complex64> {
    let centers = grid.centers();
    let shifted_k = shifted_wavenumber(reference_wavenumber, epsilon);
    let min_distance = grid.min_distance_m();
    let cell_volume = grid.cell_volume_m3();
    DMatrix::from_fn(grid.len(), grid.len(), |row, column| {
        identity_entry(row, column)
            + shifted_outgoing_green(centers[column].1, centers[row].1, shifted_k, min_distance)
                * shifted_potential[column]
                * cell_volume
    })
}

fn operator_matrix_by_columns(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    shifted_potential: &[Complex64],
    operator: GreenOperatorKind,
) -> DMatrix<Complex64> {
    let mut matrix = DMatrix::zeros(grid.len(), grid.len());
    for column in 0..grid.len() {
        let mut basis = vec![Complex64::new(0.0, 0.0); grid.len()];
        basis[column] = shifted_potential[column];
        let green_column =
            apply_shifted_green_operator(grid, reference_wavenumber, epsilon, &basis, operator);
        for row in 0..grid.len() {
            matrix[(row, column)] = identity_entry(row, column) + green_column[row];
        }
    }
    matrix
}

#[inline]
fn identity_entry(row: usize, column: usize) -> Complex64 {
    if row == column {
        Complex64::new(1.0, 0.0)
    } else {
        Complex64::new(0.0, 0.0)
    }
}

fn norm(values: &[Complex64]) -> f64 {
    values
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn adjoint_rhs_norm(values: &[Complex64]) -> f64 {
    norm(values)
}
