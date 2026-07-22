//! Single-allocation point-matrix construction.

use leto::Array2;

use super::super::GeometryError;

pub(crate) fn collect_points<const DIMENSIONS: usize>(
    sample_count: usize,
    mut fill: impl FnMut(usize, &mut [f64; DIMENSIONS]) -> Result<(), GeometryError>,
) -> Result<Array2<f64>, GeometryError> {
    let element_count =
        sample_count
            .checked_mul(DIMENSIONS)
            .ok_or(GeometryError::ElementCountOverflow {
                sample_count,
                dimensions: DIMENSIONS,
            })?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(element_count)
        .map_err(|_| GeometryError::AllocationFailed { element_count })?;
    for index in 0..sample_count {
        let mut point = [0.0; DIMENSIONS];
        fill(index, &mut point)?;
        values.extend_from_slice(&point);
    }
    Ok(Array2::from_shape_vec([sample_count, DIMENSIONS], values)
        .expect("invariant: the exact pre-sized point buffer matches its matrix shape"))
}
