//! Pitch-catch row generation for curved-array speed-shift scans.

use crate::core::error::{KwaversError, KwaversResult};

use super::super::types::SoundSpeedShiftSample;
use super::geometry::CurvedArray2d;
use super::validation::validate_scan;

/// Deterministic same-aperture pitch-catch scan over a 2-D curved array.
///
/// Rows are emitted with transmitter index as the outer loop and
/// `receiver_offsets` as the inner loop. Receiver index is
/// `(transmitter + offset) mod element_count`, so one scan definition is a
/// single source of truth for measured shifts, prediction, and reconstruction.
#[derive(Clone, Debug, PartialEq)]
pub struct CurvedArrayShiftScan {
    /// Physical curved-array geometry.
    pub array: CurvedArray2d,
    /// Positive receiver offsets in element-index units.
    pub receiver_offsets: Vec<usize>,
}

impl CurvedArrayShiftScan {
    /// Construct a deterministic curved-array scan.
    #[must_use]
    pub fn new(array: CurvedArray2d, receiver_offsets: Vec<usize>) -> Self {
        Self {
            array,
            receiver_offsets,
        }
    }

    /// Number of rows implied by the scan definition.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.array.element_count * self.receiver_offsets.len()
    }

    /// Return the physical array element coordinates.
    ///
    /// # Errors
    /// Returns [`crate::core::error::KwaversError`] when the array geometry is
    /// invalid.
    pub fn elements(
        &self,
    ) -> KwaversResult<Vec<crate::solver::inverse::same_aperture::PlanarPoint>> {
        self.array.elements()
    }

    /// Build zero-shift samples in deterministic row order.
    ///
    /// # Errors
    /// Returns [`crate::core::error::KwaversError`] when the geometry or
    /// receiver-offset set is invalid.
    pub fn samples(&self) -> KwaversResult<Vec<SoundSpeedShiftSample>> {
        let shifts = vec![0.0; self.row_count()];
        self.samples_with_time_shifts(&shifts)
    }

    /// Build measured shift samples in deterministic row order.
    ///
    /// `time_shifts_s[row]` must use the same row order as this scan:
    /// transmitter-major, then receiver-offset-minor.
    ///
    /// # Errors
    /// Returns [`crate::core::error::KwaversError`] when the geometry,
    /// receiver-offset set, or measured-shift vector violates the scan
    /// contract.
    pub fn samples_with_time_shifts(
        &self,
        time_shifts_s: &[f64],
    ) -> KwaversResult<Vec<SoundSpeedShiftSample>> {
        validate_scan(self)?;
        if time_shifts_s.len() != self.row_count() {
            return Err(KwaversError::DimensionMismatch(format!(
                "curved-array shift scan expected {} time shifts, got {}",
                self.row_count(),
                time_shifts_s.len()
            )));
        }
        if let Some((idx, value)) = time_shifts_s
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(KwaversError::InvalidInput(format!(
                "curved-array shift row {idx} has nonfinite time shift {value}"
            )));
        }

        let elements = self.array.elements()?;
        let mut samples = Vec::with_capacity(self.row_count());
        for tx_idx in 0..self.array.element_count {
            for offset in &self.receiver_offsets {
                let row = samples.len();
                let rx_idx = (tx_idx + *offset) % self.array.element_count;
                samples.push(SoundSpeedShiftSample::new(
                    elements[tx_idx],
                    elements[rx_idx],
                    time_shifts_s[row],
                ));
            }
        }
        Ok(samples)
    }
}
