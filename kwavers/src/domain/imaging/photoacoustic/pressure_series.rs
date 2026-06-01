//! Validated time series of 3-D acoustic pressure snapshots.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::ops::Deref;

/// An ordered series of 3-D pressure-field snapshots \[Pa\] sampled during a
/// photoacoustic forward simulation.
///
/// # Invariants (enforced at construction)
/// - **non-empty**: at least one snapshot is present, so `[0]` / `spatial_dim`
///   are always valid;
/// - **dimensionally uniform**: every snapshot shares the spatial dimensions of
///   the first — they are samples of one fixed grid over time, so a ragged
///   series is a construction error, not a representable state.
///
/// The series is immutable after construction (no `DerefMut`, no public mutators),
/// so the invariants cannot be invalidated by later mutation. Read access is via
/// `Deref<Target = [Array3<f64>]>`, so the type transparently behaves like a
/// `&[Array3<f64>]` slice (`iter`, `len`, indexing, `last`, slicing, and
/// `&series` → `&[Array3<f64>]` coercion at call sites).
#[derive(Debug, Clone)]
pub struct PressureFieldSeries(Vec<Array3<f64>>);

impl PressureFieldSeries {
    /// Construct a validated pressure-field series.
    ///
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] if `snapshots` is empty or if any
    /// snapshot's spatial dimensions differ from the first.
    pub fn new(snapshots: Vec<Array3<f64>>) -> KwaversResult<Self> {
        let dim = snapshots
            .first()
            .ok_or_else(|| {
                KwaversError::InvalidInput(
                    "PressureFieldSeries requires at least one pressure snapshot".to_owned(),
                )
            })?
            .dim();
        if let Some(bad) = snapshots.iter().position(|field| field.dim() != dim) {
            return Err(KwaversError::InvalidInput(format!(
                "PressureFieldSeries snapshot {bad} has dimensions {:?}, expected {dim:?}",
                snapshots[bad].dim()
            )));
        }
        Ok(Self(snapshots))
    }

    /// Spatial dimensions `(nx, ny, nz)` shared by every snapshot.
    #[must_use]
    pub fn spatial_dim(&self) -> (usize, usize, usize) {
        // Non-empty by construction invariant.
        self.0[0].dim()
    }

    /// Consume the series, returning the underlying snapshot vector.
    #[must_use]
    pub fn into_inner(self) -> Vec<Array3<f64>> {
        self.0
    }
}

impl Deref for PressureFieldSeries {
    type Target = [Array3<f64>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_uniform_nonempty_series_and_preserves_order() {
        let a = Array3::<f64>::from_elem((4, 5, 6), 1.0);
        let b = Array3::<f64>::from_elem((4, 5, 6), 2.0);
        let series = PressureFieldSeries::new(vec![a, b]).expect("uniform dims accepted");

        assert_eq!(series.len(), 2);
        assert_eq!(series.spatial_dim(), (4, 5, 6));
        // Deref-to-slice access preserves snapshot order and values.
        assert_eq!(series[0][[0, 0, 0]], 1.0);
        assert_eq!(series.last().unwrap()[[3, 4, 5]], 2.0);
    }

    #[test]
    fn new_rejects_empty_series() {
        let result = PressureFieldSeries::new(Vec::new());
        match result {
            Err(KwaversError::InvalidInput(message)) => {
                assert!(message.contains("at least one"), "got: {message}");
            }
            other => panic!("expected InvalidInput for empty series, got {other:?}"),
        }
    }

    #[test]
    fn new_rejects_dimensionally_ragged_series() {
        let a = Array3::<f64>::zeros((4, 5, 6));
        let ragged = Array3::<f64>::zeros((4, 5, 7));
        let result = PressureFieldSeries::new(vec![a, ragged]);
        match result {
            Err(KwaversError::InvalidInput(message)) => {
                // The offending index (1) and both shapes must be reported.
                assert!(message.contains("snapshot 1"), "got: {message}");
                assert!(message.contains("(4, 5, 7)"), "got: {message}");
                assert!(message.contains("(4, 5, 6)"), "got: {message}");
            }
            other => panic!("expected InvalidInput for ragged series, got {other:?}"),
        }
    }

    #[test]
    fn into_inner_round_trips_the_snapshots() {
        let a = Array3::<f64>::from_elem((2, 2, 2), 3.0);
        let series = PressureFieldSeries::new(vec![a.clone()]).unwrap();
        let recovered = series.into_inner();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0], a);
    }
}
