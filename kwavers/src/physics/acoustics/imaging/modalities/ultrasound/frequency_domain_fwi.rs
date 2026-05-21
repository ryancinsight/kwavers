//! Frequency-domain ultrasound FWI identities for multi-row ring arrays.
//!
//! This module owns the physics-only contracts from Ali et al. (2025),
//! "3D Frequency-Domain Full Waveform Inversion for Whole-Breast Imaging
//! With a Multi-Row Ring Array", DOI: 10.1109/ojuffc.2025.3570253.
//!
//! # Formal contracts
//!
//! 1. Helmholtz state equation:
//!    `(laplacian + omega^2 s(x)^2) u = delta`.
//! 2. Receiver sampling:
//!    `p = K u`, where `K` extracts pressure at ring-array elements.
//! 3. Slowness sensitivity:
//!    differentiating `A(s)u = delta` gives
//!    `du/ds = -A^{-1}(dA/ds)u` with `dA/ds = 2 omega^2 s`.
//! 4. Complex source estimation:
//!    the least-squares source scale is
//!    `gamma = <p, d> / <p, p>` for predicted pressure `p` and data `d`.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Paper model identifier for audit trails.
pub const FREQUENCY_DOMAIN_FWI_MODEL: &str = "ali_2025_multi_row_ring_frequency_domain_ust_fwi";

/// Point in physical space [m].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RingPoint {
    pub x_m: f64,
    pub y_m: f64,
    pub z_m: f64,
}

/// Multi-row ring-array geometry.
#[derive(Clone, Debug)]
pub struct MultiRowRingArray {
    circumferential_elements: usize,
    rows: usize,
    diameter_m: f64,
    row_spacing_m: f64,
    elements: Vec<RingPoint>,
}

impl MultiRowRingArray {
    /// Construct a centered multi-row ring array.
    ///
    /// Rows are centered about `z = 0`; circumferential elements lie on a
    /// circle of radius `diameter_m / 2`.
    ///
    /// # Errors
    /// Returns an error if counts or metric parameters are invalid.
    pub fn new(
        circumferential_elements: usize,
        rows: usize,
        diameter_m: f64,
        row_spacing_m: f64,
    ) -> KwaversResult<Self> {
        if circumferential_elements < 2 {
            return Err(KwaversError::InvalidInput(format!(
                "circumferential_elements must be at least 2, got {circumferential_elements}"
            )));
        }
        if rows == 0 {
            return Err(KwaversError::InvalidInput(
                "multi-row ring array requires at least one row".to_owned(),
            ));
        }
        if !diameter_m.is_finite() || diameter_m <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "ring diameter must be positive and finite, got {diameter_m}"
            )));
        }
        if !row_spacing_m.is_finite() || row_spacing_m < 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "row spacing must be finite and nonnegative, got {row_spacing_m}"
            )));
        }

        let radius = 0.5 * diameter_m;
        let row_center = 0.5 * (rows.saturating_sub(1)) as f64;
        let mut elements = Vec::with_capacity(circumferential_elements * rows);
        for row in 0..rows {
            let z_m = (row as f64 - row_center) * row_spacing_m;
            for element in 0..circumferential_elements {
                let theta = 2.0 * PI * element as f64 / circumferential_elements as f64;
                elements.push(RingPoint {
                    x_m: radius * theta.cos(),
                    y_m: radius * theta.sin(),
                    z_m,
                });
            }
        }

        Ok(Self {
            circumferential_elements,
            rows,
            diameter_m,
            row_spacing_m,
            elements,
        })
    }

    /// Construct a topology-preserving array from explicit ordered element
    /// coordinates.
    ///
    /// This preserves the cylindrical transmit topology: element
    /// `row * circumferential_elements + angular` belongs to the given row and
    /// angular transmit index.
    ///
    /// # Errors
    /// Returns an error when topology, metadata, or coordinates are invalid.
    pub fn from_ordered_elements(
        circumferential_elements: usize,
        rows: usize,
        diameter_m: f64,
        row_spacing_m: f64,
        elements: Vec<RingPoint>,
    ) -> KwaversResult<Self> {
        Self::new(circumferential_elements, rows, diameter_m, row_spacing_m)?;
        let expected = circumferential_elements * rows;
        if elements.len() != expected {
            return Err(KwaversError::DimensionMismatch(format!(
                "ordered ring elements length mismatch: got {}, expected {}",
                elements.len(),
                expected
            )));
        }
        for point in &elements {
            if !point.x_m.is_finite() || !point.y_m.is_finite() || !point.z_m.is_finite() {
                return Err(KwaversError::InvalidInput(format!(
                    "ring element coordinate must be finite, got {point:?}"
                )));
            }
        }
        Ok(Self {
            circumferential_elements,
            rows,
            diameter_m,
            row_spacing_m,
            elements,
        })
    }

    /// Ali et al. (2025) proof-of-concept geometry: 256 x 32, 22 cm diameter,
    /// 2.4 mm row spacing.
    pub fn ali_2025() -> KwaversResult<Self> {
        Self::new(256, 32, 0.22, 0.0024)
    }

    #[must_use]
    pub fn circumferential_elements(&self) -> usize {
        self.circumferential_elements
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub fn diameter_m(&self) -> f64 {
        self.diameter_m
    }

    #[must_use]
    pub fn row_spacing_m(&self) -> f64 {
        self.row_spacing_m
    }

    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    #[must_use]
    pub fn elements(&self) -> &[RingPoint] {
        &self.elements
    }

    /// Return the row-spanning cylindrical-wave source for one angular index.
    ///
    /// In the cited acquisition, transmit `q` fires circumferential element
    /// `q` in every row simultaneously.
    #[must_use]
    pub fn cylindrical_source(&self, transmit_index: usize) -> Vec<RingPoint> {
        let angular = transmit_index % self.circumferential_elements;
        (0..self.rows)
            .map(|row| self.elements[row * self.circumferential_elements + angular])
            .collect()
    }
}

/// Frequencies used by Ali et al. (2025): 200 to 800 kHz, inclusive, 50 kHz step.
#[must_use]
pub fn ali_2025_frequency_sweep_hz() -> Vec<f64> {
    (200_000..=800_000).step_by(50_000).map(f64::from).collect()
}

/// Convert sound speed [m/s] to slowness [s/m].
///
/// # Errors
/// Returns an error if any voxel is nonpositive or nonfinite.
pub fn sound_speed_to_slowness(sound_speed_m_s: &Array3<f64>) -> KwaversResult<Array3<f64>> {
    let mut slowness = Array3::zeros(sound_speed_m_s.dim());
    for (dst, &speed) in slowness.iter_mut().zip(sound_speed_m_s.iter()) {
        if !speed.is_finite() || speed <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "sound speed must be positive and finite, got {speed}"
            )));
        }
        *dst = 1.0 / speed;
    }
    Ok(slowness)
}

/// Convert slowness [s/m] to sound speed [m/s].
///
/// # Errors
/// Returns an error if any voxel is nonpositive or nonfinite.
pub fn slowness_to_sound_speed(slowness_s_per_m: &Array3<f64>) -> KwaversResult<Array3<f64>> {
    let mut sound_speed = Array3::zeros(slowness_s_per_m.dim());
    for (dst, &slowness) in sound_speed.iter_mut().zip(slowness_s_per_m.iter()) {
        if !slowness.is_finite() || slowness <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "slowness must be positive and finite, got {slowness}"
            )));
        }
        *dst = 1.0 / slowness;
    }
    Ok(sound_speed)
}

/// Local derivative of the Helmholtz mass term with respect to slowness.
#[must_use]
pub fn helmholtz_slowness_derivative(omega_rad_s: f64, slowness_s_per_m: f64) -> f64 {
    2.0 * omega_rad_s * omega_rad_s * slowness_s_per_m
}

/// Least-squares complex source scale `gamma = <predicted, observed>/<predicted,predicted>`.
///
/// # Errors
/// Returns an error when vector lengths differ or the predicted vector has
/// zero energy.
pub fn complex_source_scale(
    predicted: &[Complex64],
    observed: &[Complex64],
) -> KwaversResult<Complex64> {
    if predicted.len() != observed.len() {
        return Err(KwaversError::DimensionMismatch(format!(
            "source scale vectors differ: predicted={}, observed={}",
            predicted.len(),
            observed.len()
        )));
    }

    let mut numerator = Complex64::new(0.0, 0.0);
    let mut denominator = 0.0;
    for (&p, &d) in predicted.iter().zip(observed.iter()) {
        numerator += p.conj() * d;
        denominator += p.norm_sqr();
    }

    if denominator <= f64::EPSILON {
        return Err(KwaversError::InvalidInput(
            "cannot estimate source scale from zero predicted pressure".to_owned(),
        ));
    }

    Ok(numerator / denominator)
}

/// Complex L2 objective `0.5 ||predicted - observed||_2^2`.
///
/// # Errors
/// Returns an error when vector lengths differ.
pub fn complex_l2_objective(predicted: &[Complex64], observed: &[Complex64]) -> KwaversResult<f64> {
    if predicted.len() != observed.len() {
        return Err(KwaversError::DimensionMismatch(format!(
            "objective vectors differ: predicted={}, observed={}",
            predicted.len(),
            observed.len()
        )));
    }
    Ok(0.5
        * predicted
            .iter()
            .zip(observed.iter())
            .map(|(&p, &d)| (p - d).norm_sqr())
            .sum::<f64>())
}

/// Root-mean-square error between reconstructed and reference sound speed volumes.
///
/// # Errors
/// Returns an error when volume shapes differ.
pub fn sound_speed_rmse(
    reconstructed_m_s: &Array3<f64>,
    reference_m_s: &Array3<f64>,
) -> KwaversResult<f64> {
    if reconstructed_m_s.dim() != reference_m_s.dim() {
        return Err(KwaversError::DimensionMismatch(format!(
            "RMSE volume shape mismatch: reconstructed {:?}, reference {:?}",
            reconstructed_m_s.dim(),
            reference_m_s.dim()
        )));
    }

    let mean = reconstructed_m_s
        .iter()
        .zip(reference_m_s.iter())
        .map(|(&a, &b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f64>()
        / reconstructed_m_s.len() as f64;
    Ok(mean.sqrt())
}

/// Pearson correlation coefficient between two sound speed volumes.
///
/// # Errors
/// Returns an error when shapes differ or either volume has zero variance.
pub fn sound_speed_pcc(
    reconstructed_m_s: &Array3<f64>,
    reference_m_s: &Array3<f64>,
) -> KwaversResult<f64> {
    if reconstructed_m_s.dim() != reference_m_s.dim() {
        return Err(KwaversError::DimensionMismatch(format!(
            "PCC volume shape mismatch: reconstructed {:?}, reference {:?}",
            reconstructed_m_s.dim(),
            reference_m_s.dim()
        )));
    }

    let n = reconstructed_m_s.len() as f64;
    let mean_a = reconstructed_m_s.iter().sum::<f64>() / n;
    let mean_b = reference_m_s.iter().sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (&a, &b) in reconstructed_m_s.iter().zip(reference_m_s.iter()) {
        let da = a - mean_a;
        let db = b - mean_b;
        covariance += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom <= f64::EPSILON {
        return Err(KwaversError::InvalidInput(
            "PCC requires nonzero variance in both volumes".to_owned(),
        ));
    }

    Ok(covariance / denom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn ali_geometry_matches_paper_counts_and_spacing() {
        let array = MultiRowRingArray::ali_2025().expect("geometry");

        assert_eq!(array.circumferential_elements(), 256);
        assert_eq!(array.rows(), 32);
        assert_eq!(array.element_count(), 8192);
        assert!((array.diameter_m() - 0.22).abs() <= f64::EPSILON);
        assert!((array.row_spacing_m() - 0.0024).abs() <= f64::EPSILON);

        let source = array.cylindrical_source(7);
        assert_eq!(source.len(), 32);
        for window in source.windows(2) {
            assert!((window[1].z_m - window[0].z_m - 0.0024).abs() <= 1.0e-12);
            assert!((window[1].x_m - window[0].x_m).abs() <= 1.0e-12);
            assert!((window[1].y_m - window[0].y_m).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn complex_source_scale_recovers_amplitude_and_phase() {
        let gamma = Complex64::new(0.75, -0.25);
        let predicted = [Complex64::new(1.0, 2.0), Complex64::new(-0.5, 0.25)];
        let observed = predicted.map(|value| gamma * value);

        let recovered = complex_source_scale(&predicted, &observed).expect("source scale");

        assert!((recovered - gamma).norm() <= 1.0e-14);
    }

    #[test]
    fn helmholtz_slowness_derivative_matches_finite_difference() {
        let omega = 2.0 * PI * 250_000.0_f64;
        let slowness: f64 = 1.0 / 1500.0;
        let epsilon: f64 = 1.0e-8;
        let slowness_plus = slowness + epsilon;
        let slowness_minus = slowness - epsilon;
        let f_plus = omega * omega * slowness_plus * slowness_plus;
        let f_minus = omega * omega * slowness_minus * slowness_minus;
        let finite_difference = (f_plus - f_minus) / (2.0 * epsilon);
        let analytic = helmholtz_slowness_derivative(omega, slowness);

        assert!((finite_difference - analytic).abs() / analytic.abs() <= 1.0e-10);
    }

    #[test]
    fn sound_speed_metrics_match_definitions() {
        let reference =
            Array3::from_shape_vec((1, 1, 3), vec![1500.0, 1510.0, 1520.0]).expect("shape");
        let reconstructed =
            Array3::from_shape_vec((1, 1, 3), vec![1501.0, 1511.0, 1521.0]).expect("shape");

        let rmse = sound_speed_rmse(&reconstructed, &reference).expect("rmse");
        let pcc = sound_speed_pcc(&reconstructed, &reference).expect("pcc");

        assert!((rmse - 1.0).abs() <= f64::EPSILON);
        assert!((pcc - 1.0).abs() <= 1.0e-14);
    }
}
