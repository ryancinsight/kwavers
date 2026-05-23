//! BLI projection helpers for CBS source injection and receiver sampling.

use super::green::GreenOperatorKind;
use super::grid::{bli_weights, BliConfig, GridSpec, GridWeight};
use super::temporal::pstd_source_kappa_symbol;
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::fft::{fft_3d_complex_into, ifft_3d_complex_inplace};
use crate::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use crate::solver::inverse::linear_born_inversion::ElementPosition;
use ndarray::Array3;
use num_complex::Complex64;
use std::f64::consts::TAU;

/// Project point-source strengths onto cell-centered source density.
///
/// A unit point source contributes `weight / dV` so that integration by the
/// Green operator recovers the source strength.
///
/// # Errors
/// Returns an error when a source point has no BLI support on the grid.
pub fn source_density_from_bli(
    grid: GridSpec,
    sources: &[ElementPosition],
) -> KwaversResult<Vec<Complex64>> {
    let mut source_density = vec![Complex64::new(0.0, 0.0); grid.len()];
    for &source in sources {
        let weights = nonempty_bli_weights(grid, source, "source")?;
        for contribution in weights {
            source_density[contribution.linear_index] +=
                Complex64::new(contribution.weight / grid.cell_volume_m3(), 0.0);
        }
    }
    Ok(source_density)
}

/// Project sources according to the selected Green-operator discretization.
///
/// Continuous Helmholtz operators use bandlimited interpolation. The PSTD
/// spectral operator uses the same on-grid source mask and k-space source
/// correction as the time-domain PSTD acquisition generator.
///
/// # Errors
/// Returns an error when source support or operator parameters are invalid.
pub fn source_density_for_operator(
    grid: GridSpec,
    sources: &[ElementPosition],
    operator: GreenOperatorKind,
) -> KwaversResult<Vec<Complex64>> {
    match operator {
        GreenOperatorKind::DenseFreeSpace | GreenOperatorKind::SpectralPeriodic { .. } => {
            source_density_from_bli(grid, sources)
        }
        GreenOperatorKind::SpectralPstdPeriodic {
            time_step_s,
            reference_sound_speed_m_s,
            ..
        } => source_density_from_pstd_grid_kappa(
            grid,
            sources,
            time_step_s,
            reference_sound_speed_m_s,
        ),
    }
}

fn source_density_from_pstd_grid_kappa(
    grid: GridSpec,
    sources: &[ElementPosition],
    time_step_s: f64,
    reference_sound_speed_m_s: f64,
) -> KwaversResult<Vec<Complex64>> {
    validate_pstd_source_parameters(time_step_s, reference_sound_speed_m_s)?;
    let mut mask = Array3::<Complex64>::zeros(grid.dimensions);
    for &source in sources {
        let index = exact_grid_index(grid, source, "source")?;
        mask[index] += Complex64::new(1.0 / grid.cell_volume_m3(), 0.0);
    }

    let mut spectrum = Array3::<Complex64>::zeros(grid.dimensions);
    fft_3d_complex_into(&mask, &mut spectrum);
    let (nx, ny, nz) = grid.dimensions;
    for ix in 0..nx {
        let kx = angular_mode(ix, nx, grid.spacing_m);
        for iy in 0..ny {
            let ky = angular_mode(iy, ny, grid.spacing_m);
            for iz in 0..nz {
                let kz = angular_mode(iz, nz, grid.spacing_m);
                let k = kx.mul_add(kx, ky.mul_add(ky, kz * kz)).sqrt();
                let source_kappa =
                    pstd_source_kappa_symbol(k, time_step_s, reference_sound_speed_m_s);
                spectrum[[ix, iy, iz]] *= Complex64::new(source_kappa, 0.0);
            }
        }
    }

    ifft_3d_complex_inplace(&mut spectrum);
    Ok(spectrum.iter().copied().collect())
}

/// Sample a cell-centered field at one receiver point through the BLI stencil.
///
/// # Errors
/// Returns an error when the receiver has no BLI support on the grid.
pub fn sample_field_with_bli(
    grid: GridSpec,
    field: &[Complex64],
    receiver: ElementPosition,
) -> KwaversResult<Complex64> {
    validate_field_len(grid, field)?;
    let weights = nonempty_bli_weights(grid, receiver, "receiver")?;
    Ok(weights
        .iter()
        .map(|contribution| field[contribution.linear_index] * contribution.weight)
        .sum())
}

/// Sample all ring-array receivers from one cell-centered field.
///
/// # Errors
/// Returns an error when the field length is invalid or a receiver is outside
/// the BLI support domain.
pub fn sample_array_with_bli(
    grid: GridSpec,
    field: &[Complex64],
    array: &MultiRowRingArray,
) -> KwaversResult<Vec<Complex64>> {
    array
        .elements()
        .iter()
        .map(|&receiver| sample_field_with_bli(grid, field, receiver))
        .collect()
}

/// Sample receivers according to the selected Green-operator discretization.
///
/// Continuous Helmholtz operators use BLI. The PSTD spectral operator samples
/// the same exact grid points recorded by the PSTD acquisition sensor mask.
///
/// # Errors
/// Returns an error when field length, receiver support, or PSTD grid
/// alignment is invalid.
pub fn sample_array_for_operator(
    grid: GridSpec,
    field: &[Complex64],
    array: &MultiRowRingArray,
    operator: GreenOperatorKind,
) -> KwaversResult<Vec<Complex64>> {
    match operator {
        GreenOperatorKind::DenseFreeSpace | GreenOperatorKind::SpectralPeriodic { .. } => {
            sample_array_with_bli(grid, field, array)
        }
        GreenOperatorKind::SpectralPstdPeriodic { .. } => sample_array_on_grid(grid, field, array),
    }
}

fn sample_array_on_grid(
    grid: GridSpec,
    field: &[Complex64],
    array: &MultiRowRingArray,
) -> KwaversResult<Vec<Complex64>> {
    validate_field_len(grid, field)?;
    array
        .elements()
        .iter()
        .map(|&receiver| {
            let (ix, iy, iz) = exact_grid_index(grid, receiver, "receiver")?;
            Ok(field[grid.linear_index(ix, iy, iz)])
        })
        .collect()
}

/// Apply the Euclidean adjoint of BLI receiver sampling.
///
/// If `p_i = sum_j w_ij u_j`, this returns `R^H r`, i.e.
/// `sum_i conj(w_ij) r_i`. BLI weights are real, so conjugation is structural.
///
/// # Errors
/// Returns an error when residual cardinality differs from receiver count or a
/// receiver has no BLI support on the grid.
pub fn receiver_adjoint_from_bli(
    grid: GridSpec,
    array: &MultiRowRingArray,
    residual: &[Complex64],
) -> KwaversResult<Vec<Complex64>> {
    if residual.len() != array.element_count() {
        return Err(KwaversError::DimensionMismatch(format!(
            "CBS receiver residual mismatch: residual {}, geometry {}",
            residual.len(),
            array.element_count()
        )));
    }

    let mut adjoint = vec![Complex64::new(0.0, 0.0); grid.len()];
    for (&receiver, &value) in array.elements().iter().zip(residual.iter()) {
        let weights = nonempty_bli_weights(grid, receiver, "receiver")?;
        for contribution in weights {
            adjoint[contribution.linear_index] += value * contribution.weight;
        }
    }
    Ok(adjoint)
}

/// Apply the Euclidean receiver adjoint for the selected operator.
///
/// Continuous Helmholtz operators use the BLI adjoint. The PSTD spectral
/// operator injects residuals into the exact grid cells recorded by the PSTD
/// sensor mask.
///
/// # Errors
/// Returns an error when residual cardinality or PSTD receiver alignment is
/// invalid.
pub fn receiver_adjoint_for_operator(
    grid: GridSpec,
    array: &MultiRowRingArray,
    residual: &[Complex64],
    operator: GreenOperatorKind,
) -> KwaversResult<Vec<Complex64>> {
    match operator {
        GreenOperatorKind::DenseFreeSpace | GreenOperatorKind::SpectralPeriodic { .. } => {
            receiver_adjoint_from_bli(grid, array, residual)
        }
        GreenOperatorKind::SpectralPstdPeriodic { .. } => {
            receiver_adjoint_on_grid(grid, array, residual)
        }
    }
}

fn receiver_adjoint_on_grid(
    grid: GridSpec,
    array: &MultiRowRingArray,
    residual: &[Complex64],
) -> KwaversResult<Vec<Complex64>> {
    if residual.len() != array.element_count() {
        return Err(KwaversError::DimensionMismatch(format!(
            "CBS receiver residual mismatch: residual {}, geometry {}",
            residual.len(),
            array.element_count()
        )));
    }

    let mut adjoint = vec![Complex64::new(0.0, 0.0); grid.len()];
    for (&receiver, &value) in array.elements().iter().zip(residual.iter()) {
        let (ix, iy, iz) = exact_grid_index(grid, receiver, "receiver")?;
        adjoint[grid.linear_index(ix, iy, iz)] += value;
    }
    Ok(adjoint)
}

fn nonempty_bli_weights(
    grid: GridSpec,
    point: ElementPosition,
    role: &str,
) -> KwaversResult<Vec<GridWeight>> {
    let weights = bli_weights(grid, point, BliConfig::default())?;
    if weights.is_empty() {
        return Err(KwaversError::InvalidInput(format!(
            "CBS {role} point {:?} lies outside the inversion grid {:?} with spacing {}",
            point, grid.dimensions, grid.spacing_m
        )));
    }
    Ok(weights)
}

fn validate_field_len(grid: GridSpec, field: &[Complex64]) -> KwaversResult<()> {
    if field.len() != grid.len() {
        return Err(KwaversError::DimensionMismatch(format!(
            "CBS field length mismatch: grid {}, field {}",
            grid.len(),
            field.len()
        )));
    }
    Ok(())
}

fn validate_pstd_source_parameters(
    time_step_s: f64,
    reference_sound_speed_m_s: f64,
) -> KwaversResult<()> {
    if !time_step_s.is_finite() || time_step_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "PSTD source projection time_step_s must be positive and finite, got {time_step_s}"
        )));
    }
    if !reference_sound_speed_m_s.is_finite() || reference_sound_speed_m_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "PSTD source projection reference sound speed must be positive and finite, got {reference_sound_speed_m_s}"
        )));
    }
    Ok(())
}

fn exact_grid_index(
    grid: GridSpec,
    point: ElementPosition,
    role: &str,
) -> KwaversResult<(usize, usize, usize)> {
    let nearest = [
        exact_axis_index(grid.dimensions.0, grid.spacing_m, point.x_m, role)?,
        exact_axis_index(grid.dimensions.1, grid.spacing_m, point.y_m, role)?,
        exact_axis_index(grid.dimensions.2, grid.spacing_m, point.z_m, role)?,
    ];
    Ok((nearest[0], nearest[1], nearest[2]))
}

fn exact_axis_index(n: usize, spacing_m: f64, value_m: f64, role: &str) -> KwaversResult<usize> {
    let raw = value_m / spacing_m + 0.5 * (n - 1) as f64;
    let rounded = raw.round();
    let tolerance = 1.0e-9;
    if (raw - rounded).abs() > tolerance || rounded < 0.0 || rounded > (n - 1) as f64 {
        return Err(KwaversError::InvalidInput(format!(
            "PSTD spectral {role} point coordinate {value_m} is not on the centered grid axis"
        )));
    }
    Ok(rounded as usize)
}

fn angular_mode(index: usize, count: usize, spacing_m: f64) -> f64 {
    let signed_index = if index <= count / 2 {
        index as f64
    } else {
        index as f64 - count as f64
    };
    TAU * signed_index / (count as f64 * spacing_m)
}
