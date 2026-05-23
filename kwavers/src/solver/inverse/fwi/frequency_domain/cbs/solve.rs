//! CBS fixed-point volume solver — forward and adjoint.
//!
//! # Forward CBS system
//!
//! Solves `(∇² + k(x)²) u = q` in the Lippmann-Schwinger form
//! `(I + G_ε diag(Ṽ)) u = G_ε q`
//! where `Ṽ = V − iε` is the shifted scattering potential and `G_ε` is the
//! shifted reference Green operator.  The Richardson fixed-point iterate is:
//! ```text
//! rhs_k       = q − Ṽ u_k
//! target_k    = G_ε(rhs_k)              [fixed-point image]
//! residual_k  = u_k − target_k          [= A u_k − G_ε q]
//! u_{k+1}    = u_k + γ · residual_k,   γ = iε / Ṽ
//! ```
//! Contraction is guaranteed when `ε ≥ ‖V‖_∞`
//! (Osnabrugge–Leedumrongwatthanakun–Vellekoop 2016, Theorem 1).
//!
//! # Adjoint CBS system (discrete Euclidean adjoint)
//!
//! Solves `A^H λ = r` where `A^H = I + diag(Ṽ^*) G_ε^H`.
//! The adjoint fixed-point iterate mirrors the forward exactly:
//! ```text
//! adj_green_k = G_ε^H(λ_k)
//! target_k    = r − Ṽ^* · adj_green_k   [fixed-point image]
//! residual_k  = λ_k − target_k          [= A^H λ_k − r]
//! λ_{k+1}    = λ_k + γ^H · residual_k, γ^H = conj(γ) = −iε / Ṽ^*
//! ```
//!
//! ## Convergence proof sketch
//!
//! The iteration matrix of the adjoint update is
//! `I + γ^H A^H = diag(V/(V + iε)) − iε G_ε^H`.
//! The diagonal factor satisfies `|V/(V + iε)| = |V|/√(V² + ε²) < 1` for
//! ε > 0, providing the same elementwise contraction guarantee as the forward
//! `|V/(V − iε)| < 1`.  The spectral radius of the full operator is therefore
//! < 1 under the same `ε ≥ ‖V‖_∞` condition.
//!
//! ## Operator routing
//!
//! | `GreenOperatorKind`        | Forward path      | Adjoint path              |
//! |----------------------------|-------------------|---------------------------|
//! | `DenseFreeSpace`           | O(N²) dense sum   | Exact dense LU (small N)  |
//! | `SpectralPeriodic`         | FFT fixed-point   | FFT adjoint fixed-point   |
//! | `SpectralPstdPeriodic`     | FFT fixed-point   | FFT adjoint fixed-point   |
//!
//! The dense LU path is kept only for `DenseFreeSpace`.  All spectral
//! operators use the iterative adjoint, which costs `O(max_iterations × N
//! log N)` and avoids the `O(N²log N)` matrix build plus `O(N³)` LU
//! factorization required for large spectral grids.

use super::green::{
    apply_shifted_green_adjoint_operator, apply_shifted_green_operator, shifted_outgoing_green,
    shifted_wavenumber, GreenOperatorKind,
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
/// `source_density` is a cell-centered source density.  A point source of unit
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
/// Routes to an exact dense LU solve for `DenseFreeSpace` (small grids only)
/// and to an iterative Richardson adjoint for all spectral operators.  The
/// iterative path costs `O(max_iterations × N log N)` — identical to the
/// forward solve — and avoids the `O(N² log N)` matrix build plus `O(N³)` LU
/// factorization that would be required for large spectral grids.
///
/// # Errors
/// Returns an error if the discrete system is invalid, or (for `DenseFreeSpace`)
/// if the dense operator is singular or the residual exceeds tolerance.
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
    match operator_kind {
        GreenOperatorKind::DenseFreeSpace => solve_adjoint_dense_free_space(
            grid,
            reference_wavenumber,
            epsilon,
            &shifted,
            adjoint_rhs,
            config,
        ),
        GreenOperatorKind::SpectralPeriodic { .. }
        | GreenOperatorKind::SpectralPstdPeriodic { .. } => solve_adjoint_spectral_iterative(
            grid,
            reference_wavenumber,
            epsilon,
            &shifted,
            adjoint_rhs,
            config,
            operator_kind,
        ),
    }
}

/// Exact dense LU adjoint solve for `DenseFreeSpace`.
///
/// Builds the full N×N operator matrix and solves `A^H λ = r` via LU
/// factorization.  Correct only for small grids (N ≲ 1000) where the O(N³)
/// cost is acceptable.
fn solve_adjoint_dense_free_space(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    shifted: &[Complex64],
    adjoint_rhs: &[Complex64],
    config: CbsConfig,
) -> KwaversResult<CbsSolution> {
    let operator = dense_free_space_operator_matrix(grid, reference_wavenumber, epsilon, shifted);
    let adjoint_operator = operator.adjoint();
    let rhs = DVector::from_column_slice(adjoint_rhs);
    let solution = adjoint_operator.clone().lu().solve(&rhs).ok_or_else(|| {
        KwaversError::InvalidInput("dense CBS adjoint operator is singular".to_owned())
    })?;
    let residual = &adjoint_operator * &solution - &rhs;
    let relative_residual = residual.norm() / norm(adjoint_rhs).max(f64::EPSILON);
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

/// Iterative Richardson adjoint solve for spectral CBS operators.
///
/// Solves `A^H λ = r` where `A^H = I + diag(Ṽ^*) G_ε^H` by the fixed-point
/// iterate that exactly mirrors the forward CBS solver:
/// ```text
/// adj_green_k = G_ε^H(λ_k)
/// target_k    = r − Ṽ^* · adj_green_k
/// residual_k  = λ_k − target_k           [= A^H λ_k − r]
/// λ_{k+1}    = λ_k + γ^H · residual_k,  γ^H = conj(γ) = −iε / Ṽ^*
/// ```
///
/// The iteration matrix is `I + γ^H A^H = diag(V/(V + iε)) − iε G_ε^H`.
/// Its diagonal factor `|V/(V + iε)| < 1` ensures contraction under the same
/// `ε ≥ ‖V‖_∞` condition that guarantees forward convergence.
///
/// Cost: `O(max_iterations × N log N)` — identical to the forward solve.
fn solve_adjoint_spectral_iterative(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    shifted: &[Complex64],
    adjoint_rhs: &[Complex64],
    config: CbsConfig,
    operator_kind: GreenOperatorKind,
) -> KwaversResult<CbsSolution> {
    // γ^H = conj(iε / Ṽ) = conj(iε) / conj(Ṽ) = −iε / Ṽ^*
    let adjoint_gamma: Vec<Complex64> = shifted
        .iter()
        .map(|&v| Complex64::new(0.0, -epsilon) / v.conj())
        .collect();
    let mut field = vec![Complex64::new(0.0, 0.0); grid.len()];
    let mut relative_residual = f64::INFINITY;

    for iteration in 1..=config.max_iterations {
        // G_ε^H(λ_k): Hermitian adjoint of the shifted Green operator applied to λ_k
        let adj_green = apply_shifted_green_adjoint_operator(
            grid,
            reference_wavenumber,
            epsilon,
            &field,
            operator_kind,
        );
        // target_k = r − Ṽ^* · G_ε^H(λ_k)   [adjoint fixed-point image]
        // residual_k = λ_k − target_k = A^H λ_k − r
        let residual: Vec<Complex64> = field
            .iter()
            .zip(shifted.iter().zip(adj_green.iter().zip(adjoint_rhs.iter())))
            .map(|(&lam, (&v, (&g_lam, &rhs)))| lam - (rhs - v.conj() * g_lam))
            .collect();
        // Convergence: ||residual|| / ||target|| mirrors the forward denominator
        let target_norm = norm_of_target(adjoint_rhs, &shifted, &adj_green);
        relative_residual = norm(&residual) / target_norm.max(f64::EPSILON);
        // λ_{k+1} = λ_k + γ^H · residual_k  (same sign as forward += γ · residual)
        for ((lam, &gam), &res) in field
            .iter_mut()
            .zip(adjoint_gamma.iter())
            .zip(residual.iter())
        {
            *lam += gam * res;
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

/// Compute `‖r − Ṽ^* · adj_green‖` — the adjoint-target norm used as the
/// convergence denominator, mirroring `norm(green_rhs)` in the forward solve.
fn norm_of_target(
    adjoint_rhs: &[Complex64],
    shifted: &[Complex64],
    adj_green: &[Complex64],
) -> f64 {
    adjoint_rhs
        .iter()
        .zip(shifted.iter().zip(adj_green.iter()))
        .map(|(&rhs, (&v, &g))| (rhs - v.conj() * g).norm_sqr())
        .sum::<f64>()
        .sqrt()
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
    match operator {
        GreenOperatorKind::DenseFreeSpace => {}
        GreenOperatorKind::SpectralPeriodic { absorbing_boundary } => {
            absorbing_boundary.validate_for_grid(grid)?;
        }
        GreenOperatorKind::SpectralPstdPeriodic {
            time_step_s,
            reference_sound_speed_m_s,
            absorbing_boundary,
            temporal_transfer,
        } => {
            if !time_step_s.is_finite() || time_step_s <= 0.0 {
                return Err(KwaversError::InvalidInput(format!(
                    "PSTD spectral Green time_step_s must be positive and finite, got {time_step_s}"
                )));
            }
            if !reference_sound_speed_m_s.is_finite() || reference_sound_speed_m_s <= 0.0 {
                return Err(KwaversError::InvalidInput(format!(
                    "PSTD spectral Green reference sound speed must be positive and finite, got {reference_sound_speed_m_s}"
                )));
            }
            if let Some(temporal_transfer) = temporal_transfer {
                temporal_transfer.validate()?;
            }
            absorbing_boundary.validate_for_grid(grid)?;
        }
    }
    Ok(())
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
