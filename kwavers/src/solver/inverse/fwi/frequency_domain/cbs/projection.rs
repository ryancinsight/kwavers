//! BLI projection helpers for CBS source injection and receiver sampling.

use super::grid::{bli_weights, BliConfig, GridSpec, GridWeight};
use crate::core::error::{KwaversError, KwaversResult};
use crate::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    MultiRowRingArray, ElementPosition,
};
use num_complex::Complex64;

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
