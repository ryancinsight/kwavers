use super::error::GasTransferError;
use eunomia::FloatElement;
use kwavers_physics::acoustics::bubble_dynamics::gas_transfer::spherical_material_coordinate;

pub(super) struct RadialGrid<T> {
    pub(super) faces: Vec<T>,
    pub(super) widths: Vec<T>,
    pub(super) inverse_distances: Vec<T>,
    pub(super) inner_weights: [T; 2],
}

impl<T: FloatElement> RadialGrid<T> {
    pub(super) fn new(extent: T, cells: usize) -> Result<Self, GasTransferError> {
        let two = T::from_count(2);
        let three = T::from_count(3);
        let count = T::from_count(cells);
        let faces = (0..=cells)
            .map(|j| {
                let fraction = T::from_count(j) / count;
                spherical_material_coordinate(extent * fraction * fraction)
            })
            .collect::<Vec<_>>();
        let widths: Vec<_> = faces.windows(2).map(|pair| pair[1] - pair[0]).collect();
        if !widths.iter().all(|&w| w > T::ZERO && w.is_finite()) {
            return Err(GasTransferError::Unrepresentable);
        }
        let inverse_distances = widths
            .windows(2)
            .map(|pair| two / (pair[0] + pair[1]))
            .chain(std::iter::once(two / widths[cells - 1]))
            .collect();
        // Fit a quadratic through the wall point and the mean over each
        // of the first two cells. Use the first width to scale the moments.
        let extent_ratio = faces[2] / widths[0];
        let first_moment = (T::ONE + extent_ratio) / two;
        let second_moment = (T::ONE + extent_ratio + extent_ratio * extent_ratio) / three;
        let determinant = second_moment / two - first_moment / three;
        let slope_scale = T::ONE / (widths[0] * determinant);
        let inner_weights = [second_moment * slope_scale, (T::ZERO - slope_scale) / three];
        if inner_weights.iter().any(|value| !value.is_finite()) {
            return Err(GasTransferError::Unrepresentable);
        }
        Ok(Self {
            faces,
            widths,
            inverse_distances,
            inner_weights,
        })
    }
}
