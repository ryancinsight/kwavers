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

use aequitas::systems::si::quantities::Length;
use aequitas::systems::si::units::Meter;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::Complex64;
use kwavers_transducer::transducers::{ElementPosition, TransducerGeometry};
use leto::Array3;

/// Paper model identifier for audit trails.
pub const FREQUENCY_DOMAIN_FWI_MODEL: &str = "ali_2025_multi_row_ring_frequency_domain_ust_fwi";

/// Multi-row ring-array geometry.
#[derive(Clone, Debug)]
pub struct MultiRowRingArray {
    circumferential_elements: usize,
    rows: usize,
    diameter: Length<f64>,
    row_spacing: Length<f64>,
    elements: Vec<ElementPosition>,
}

impl TransducerGeometry for MultiRowRingArray {
    fn elements(&self) -> &[ElementPosition] {
        &self.elements
    }
}

impl MultiRowRingArray {
    /// Construct a centered multi-row ring array.
    ///
    /// Rows are centered about `z = 0`; circumferential elements lie on a
    /// circle of radius `diameter / 2`.
    ///
    /// # Errors
    /// Returns an error if counts or metric parameters are invalid.
    pub fn new(
        circumferential_elements: usize,
        rows: usize,
        diameter: Length<f64>,
        row_spacing: Length<f64>,
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
        let diameter_m = diameter.in_unit::<Meter>();
        let row_spacing_m = row_spacing.in_unit::<Meter>();
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
                let theta = TWO_PI * element as f64 / circumferential_elements as f64;
                elements.push(ElementPosition {
                    x: Length::from_unit::<Meter>(radius * theta.cos()),
                    y: Length::from_unit::<Meter>(radius * theta.sin()),
                    z: Length::from_unit::<Meter>(z_m),
                });
            }
        }

        Ok(Self {
            circumferential_elements,
            rows,
            diameter,
            row_spacing,
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
        diameter: Length<f64>,
        row_spacing: Length<f64>,
        elements: Vec<ElementPosition>,
    ) -> KwaversResult<Self> {
        Self::new(circumferential_elements, rows, diameter, row_spacing)?;
        let expected = circumferential_elements * rows;
        if elements.len() != expected {
            return Err(KwaversError::DimensionMismatch(format!(
                "ordered ring elements length mismatch: got {}, expected {}",
                elements.len(),
                expected
            )));
        }
        for point in &elements {
            if !point.x.in_unit::<Meter>().is_finite()
                || !point.y.in_unit::<Meter>().is_finite()
                || !point.z.in_unit::<Meter>().is_finite()
            {
                return Err(KwaversError::InvalidInput(format!(
                    "ring element coordinate must be finite, got {point:?}"
                )));
            }
        }
        Ok(Self {
            circumferential_elements,
            rows,
            diameter,
            row_spacing,
            elements,
        })
    }

    /// Ali et al. (2025) proof-of-concept geometry: 256 x 32, 22 cm diameter,
    /// 2.4 mm row spacing.
    ///
    /// # Errors
    ///
    /// Returns the geometry-construction error when the fixed published dimensions violate the
    /// ring-array invariants.
    pub fn ali_2025() -> KwaversResult<Self> {
        Self::new(
            256,
            32,
            Length::from_unit::<Meter>(0.22),
            Length::from_unit::<Meter>(0.0024),
        )
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
    pub fn diameter(&self) -> Length<f64> {
        self.diameter
    }

    #[must_use]
    pub fn row_spacing(&self) -> Length<f64> {
        self.row_spacing
    }

    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    #[must_use]
    pub fn elements(&self) -> &[ElementPosition] {
        &self.elements
    }

    /// Return the row-spanning cylindrical-wave source for one angular index.
    ///
    /// In the cited acquisition, transmit `q` fires circumferential element
    /// `q` in every row simultaneously.
    #[must_use]
    pub fn cylindrical_source(&self, transmit_index: usize) -> Vec<ElementPosition> {
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
    let shape = sound_speed_m_s.shape();
    let mut slowness = Array3::<f64>::zeros([shape[0], shape[1], shape[2]]);
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
    let shape = slowness_s_per_m.shape();
    let mut sound_speed = Array3::<f64>::zeros([shape[0], shape[1], shape[2]]);
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
/// Returns an error when volume shapes differ or when both volumes are empty.
pub fn sound_speed_rmse(
    reconstructed_m_s: &Array3<f64>,
    reference_m_s: &Array3<f64>,
) -> KwaversResult<f64> {
    if reconstructed_m_s.shape() != reference_m_s.shape() {
        return Err(KwaversError::DimensionMismatch(format!(
            "RMSE volume shape mismatch: reconstructed {:?}, reference {:?}",
            reconstructed_m_s.shape(),
            reference_m_s.shape()
        )));
    }
    let n = reconstructed_m_s.iter().count();
    if n == 0 {
        return Err(KwaversError::InvalidInput(
            "RMSE requires non-empty volumes".to_owned(),
        ));
    }
    let mean = reconstructed_m_s
        .iter()
        .zip(reference_m_s.iter())
        .map(|(&a, &b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f64>()
        / n as f64;
    Ok(mean.sqrt())
}

/// Pearson correlation coefficient between two sound speed volumes.
///
/// # Errors
/// Returns an error when shapes differ, volumes are empty, or either volume has zero variance.
pub fn sound_speed_pcc(
    reconstructed_m_s: &Array3<f64>,
    reference_m_s: &Array3<f64>,
) -> KwaversResult<f64> {
    if reconstructed_m_s.shape() != reference_m_s.shape() {
        return Err(KwaversError::DimensionMismatch(format!(
            "PCC volume shape mismatch: reconstructed {:?}, reference {:?}",
            reconstructed_m_s.shape(),
            reference_m_s.shape()
        )));
    }
    let n_usize = reconstructed_m_s.iter().count();
    if n_usize == 0 {
        return Err(KwaversError::InvalidInput(
            "PCC requires non-empty volumes".to_owned(),
        ));
    }
    let n = n_usize as f64;
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

/// Two opposed linear arrays on a rotation stage for transmission-USCT FWI.
///
/// The acquisition sweeps `view_count` uniformly-spaced angles from `0` to
/// `2π` (exclusive). At each view, array 1 (the transmitter side) fires one
/// element at a time while both arrays record: `receiver_count()` = `2 *
/// elements_per_array`. `transmission_count()` = `elements_per_array *
/// view_count`.
///
/// Element positions are pre-computed at construction so that the inner loop
/// of the inversion does not allocate per transmit.
///
/// # Geometry
///
/// At view angle `θ = view * 2π / view_count`, array 1 sits at
/// `(+standoff * cos θ,  +standoff * sin θ)` and array 2 at the opposite
/// side.  Elements are uniformly spaced along the direction perpendicular to
/// the standoff axis in the `(x, y)` plane; `z = 0` for all elements (2-D
/// acquisition).
///
/// # References
///
/// Atlas ADR 116 — per-view element-position rotation for the rotating
/// opposed-linear-array acquisition.
#[derive(Clone, Debug)]
pub struct RotatingOpposedLinearArray {
    elements_per_array: usize,
    view_count: usize,
    /// Flat `[transmission_count]` single-element source positions.
    /// Transmit `t` fires `sources[t]`.
    sources: Vec<ElementPosition>,
    /// Per-view receiver lists: `receivers_per_view[v]` is the full
    /// `[2 * elements_per_array]` receiver set at view `v`.
    receivers_per_view: Vec<Vec<ElementPosition>>,
}

impl RotatingOpposedLinearArray {
    /// Construct the array, pre-computing all element positions.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` for non-positive counts or
    /// non-positive, non-finite metric parameters.
    pub fn new(
        elements_per_array: usize,
        element_pitch_m: f64,
        standoff_m: f64,
        view_count: usize,
    ) -> KwaversResult<Self> {
        if elements_per_array < 2 {
            return Err(KwaversError::InvalidInput(format!(
                "elements_per_array must be at least 2, got {elements_per_array}"
            )));
        }
        if view_count == 0 {
            return Err(KwaversError::InvalidInput(
                "view_count must be at least 1".to_owned(),
            ));
        }
        if !element_pitch_m.is_finite() || element_pitch_m <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "element_pitch_m must be positive and finite, got {element_pitch_m}"
            )));
        }
        if !standoff_m.is_finite() || standoff_m <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "standoff_m must be positive and finite, got {standoff_m}"
            )));
        }

        let angular_step = TWO_PI / view_count as f64;

        let mut receivers_per_view = Vec::with_capacity(view_count);
        let mut sources = Vec::with_capacity(elements_per_array * view_count);

        for view in 0..view_count {
            let theta = view as f64 * angular_step;
            let (sin_t, cos_t) = theta.sin_cos();

            // Unit vector along the array (perpendicular to standoff axis in xy).
            let (arr_x, arr_y) = (-sin_t, cos_t);

            // Array 1 centre: (+standoff * cos θ, +standoff * sin θ).
            let c1x = standoff_m * cos_t;
            let c1y = standoff_m * sin_t;

            // Array 2 centre: opposite side.
            let c2x = -standoff_m * cos_t;
            let c2y = -standoff_m * sin_t;

            let mut view_receivers = Vec::with_capacity(2 * elements_per_array);

            for k in 0..elements_per_array {
                let offset = (k as f64 - 0.5 * (elements_per_array as f64 - 1.0)) * element_pitch_m;
                // Array 1 element k — also the source for transmit `view * n + k`.
                let p1 = ElementPosition {
                    x: Length::from_unit::<Meter>(c1x + arr_x * offset),
                    y: Length::from_unit::<Meter>(c1y + arr_y * offset),
                    z: Length::from_unit::<Meter>(0.0),
                };
                sources.push(p1);
                view_receivers.push(p1);
            }
            for k in 0..elements_per_array {
                let offset = (k as f64 - 0.5 * (elements_per_array as f64 - 1.0)) * element_pitch_m;
                let p2 = ElementPosition {
                    x: Length::from_unit::<Meter>(c2x + arr_x * offset),
                    y: Length::from_unit::<Meter>(c2y + arr_y * offset),
                    z: Length::from_unit::<Meter>(0.0),
                };
                view_receivers.push(p2);
            }
            receivers_per_view.push(view_receivers);
        }

        Ok(Self {
            elements_per_array,
            view_count,
            sources,
            receivers_per_view,
        })
    }

    /// Number of elements per linear array.
    #[must_use]
    pub fn elements_per_array(&self) -> usize {
        self.elements_per_array
    }

    /// Number of rotation-stage views.
    #[must_use]
    pub fn view_count(&self) -> usize {
        self.view_count
    }

    /// Sources slice for transmit `t` (single-element).
    ///
    /// # Panics
    /// Panics when `t >= transmission_count()`.
    #[must_use]
    pub fn transmit_sources(&self, t: usize) -> &[ElementPosition] {
        std::slice::from_ref(&self.sources[t])
    }

    /// Receiver slice for transmit `t` (all `2 * elements_per_array` elements).
    ///
    /// # Panics
    /// Panics when `t >= transmission_count()`.
    #[must_use]
    pub fn transmit_receivers(&self, t: usize) -> &[ElementPosition] {
        let view = t / self.elements_per_array;
        &self.receivers_per_view[view]
    }

    /// Total transmit events.
    #[must_use]
    pub fn transmission_count(&self) -> usize {
        self.elements_per_array * self.view_count
    }

    /// Receiver count (constant across transmits).
    #[must_use]
    pub fn receiver_count(&self) -> usize {
        2 * self.elements_per_array
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use std::f64::consts::PI;

    #[test]
    fn ali_geometry_matches_paper_counts_and_spacing() {
        let array = MultiRowRingArray::ali_2025().expect("geometry");

        assert_eq!(array.circumferential_elements(), 256);
        assert_eq!(array.rows(), 32);
        assert_eq!(array.element_count(), 8192);
        assert!((array.diameter().in_unit::<Meter>() - 0.22).abs() <= f64::EPSILON);
        assert!((array.row_spacing().in_unit::<Meter>() - 0.0024).abs() <= f64::EPSILON);

        let source = array.cylindrical_source(7);
        assert_eq!(source.len(), 32);
        for window in source.windows(2) {
            assert!(
                (window[1].z.in_unit::<Meter>() - window[0].z.in_unit::<Meter>() - 0.0024).abs()
                    <= 1.0e-12
            );
            assert!(
                (window[1].x.in_unit::<Meter>() - window[0].x.in_unit::<Meter>()).abs() <= 1.0e-12
            );
            assert!(
                (window[1].y.in_unit::<Meter>() - window[0].y.in_unit::<Meter>()).abs() <= 1.0e-12
            );
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
        let slowness: f64 = 1.0 / SOUND_SPEED_WATER_SIM;
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
            Array3::from_shape_vec([1, 1, 3], vec![SOUND_SPEED_WATER_SIM, 1510.0, 1520.0])
                .expect("shape");
        let reconstructed =
            Array3::from_shape_vec([1, 1, 3], vec![1501.0, 1511.0, 1521.0]).expect("shape");

        let rmse = sound_speed_rmse(&reconstructed, &reference).expect("rmse");
        let pcc = sound_speed_pcc(&reconstructed, &reference).expect("pcc");

        assert!((rmse - 1.0).abs() <= f64::EPSILON);
        assert!((pcc - 1.0).abs() <= 1.0e-14);
    }

    #[test]
    fn rotating_array_counts_are_correct() {
        let arr = RotatingOpposedLinearArray::new(8, 1.5e-3, 0.1, 180).expect("construction");
        assert_eq!(arr.transmission_count(), 8 * 180);
        assert_eq!(arr.receiver_count(), 16);
        assert_eq!(arr.transmit_sources(0).len(), 1);
        assert_eq!(arr.transmit_receivers(0).len(), 16);
    }

    #[test]
    fn rotating_array_at_view_zero_is_opposed() {
        // At θ=0: array 1 at (+standoff, *, 0), array 2 at (−standoff, *, 0).
        let standoff = 0.1_f64;
        let pitch = 1.5e-3_f64;
        let n = 4_usize;
        let arr = RotatingOpposedLinearArray::new(n, pitch, standoff, 4).expect("construction");

        // Transmit element 0 of view 0 → first source.
        let src = arr.transmit_sources(0)[0];
        let sx = src.x.in_unit::<Meter>();
        let sy = src.y.in_unit::<Meter>();
        // At θ=0: centre of array 1 is (+standoff, 0).
        // Array is along y (arr_x=-sin0=0, arr_y=cos0=1).
        // Element 0 offset = (0 - 1.5) * pitch = -1.5 * 1.5e-3.
        let expected_sx = standoff;
        let expected_sy = (0.0 - 0.5 * (n as f64 - 1.0)) * pitch;
        assert!(
            (sx - expected_sx).abs() <= 1.0e-12,
            "source x: {sx} vs {expected_sx}"
        );
        assert!(
            (sy - expected_sy).abs() <= 1.0e-12,
            "source y: {sy} vs {expected_sy}"
        );

        // Receivers: array 1 (n positions) then array 2 (n positions).
        let rx = arr.transmit_receivers(0);
        assert_eq!(rx.len(), 2 * n);
        // Array 1 elements should be at x ≈ +standoff.
        for r in &rx[..n] {
            assert!((r.x.in_unit::<Meter>() - standoff).abs() <= 1.0e-12);
        }
        // Array 2 elements should be at x ≈ −standoff.
        for r in &rx[n..] {
            assert!((r.x.in_unit::<Meter>() + standoff).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn rotating_array_rotation_round_trip() {
        // Rotating by θ then −θ must reproduce the original positions.
        let arr = RotatingOpposedLinearArray::new(4, 1.5e-3, 0.1, 8).expect("construction");
        let n_per = arr.elements_per_array();

        // View 0 source positions.
        for elem in 0..n_per {
            let t0 = elem; // view 0, element `elem`
            let src0 = arr.transmit_sources(t0)[0];
            let rx0 = &arr.transmit_receivers(t0).to_vec();

            // After full 360° rotation (view_count views) we are back at view 0.
            let t_full = 8 * n_per + elem; // would wrap back to same view if > transmission_count
                                           // Instead, verify: view 0 and view 8 (= view_count) are the same by construction.
                                           // The geometry repeats modulo view_count so view % view_count == 0 equals view 0.
                                           // Since transmission_count = n * view_count, transmit n*view_count+elem is out of
                                           // range. Instead verify that transmit elem (view 0) matches view 8%8 = 0 by checking
                                           // the angular step produces 360° total: step = TWO_PI / view_count.
            let _ = t_full;

            // Round-trip: rotate by +2*PI/8 then −2*PI/8. Element positions at view 1 then
            // back-rotated by the same angle must equal view 0.
            let step = TWO_PI / 8.0;
            let t1 = n_per + elem; // view 1, same element
            let src1 = arr.transmit_sources(t1)[0];
            let cos_neg = (-step).cos();
            let sin_neg = (-step).sin();
            let bx = src1.x.in_unit::<Meter>() * cos_neg - src1.y.in_unit::<Meter>() * sin_neg;
            let by = src1.x.in_unit::<Meter>() * sin_neg + src1.y.in_unit::<Meter>() * cos_neg;
            assert!(
                (bx - src0.x.in_unit::<Meter>()).abs() <= 1.0e-12,
                "round-trip x: {bx} vs {}",
                src0.x.in_unit::<Meter>()
            );
            assert!(
                (by - src0.y.in_unit::<Meter>()).abs() <= 1.0e-12,
                "round-trip y: {by} vs {}",
                src0.y.in_unit::<Meter>()
            );

            // Same for receivers.
            let rx1 = &arr.transmit_receivers(t1).to_vec();
            for (r0, r1) in rx0.iter().zip(rx1.iter()) {
                let bxr = r1.x.in_unit::<Meter>() * cos_neg - r1.y.in_unit::<Meter>() * sin_neg;
                let byr = r1.x.in_unit::<Meter>() * sin_neg + r1.y.in_unit::<Meter>() * cos_neg;
                assert!((bxr - r0.x.in_unit::<Meter>()).abs() <= 1.0e-12);
                assert!((byr - r0.y.in_unit::<Meter>()).abs() <= 1.0e-12);
            }
        }
    }

    #[test]
    fn rotating_array_view_separation_is_equal() {
        // All adjacent views should be separated by the same angle.
        let n_views = 6_usize;
        let arr = RotatingOpposedLinearArray::new(4, 1.5e-3, 0.1, n_views).expect("construction");
        let n = arr.elements_per_array();
        let step = TWO_PI / n_views as f64;

        for v in 0..n_views - 1 {
            let s0 = arr.transmit_sources(v * n)[0];
            let s1 = arr.transmit_sources((v + 1) * n)[0];
            // Angle between adjacent view-0-element-0 source positions.
            let a0 = s0.y.in_unit::<Meter>().atan2(s0.x.in_unit::<Meter>());
            let a1 = s1.y.in_unit::<Meter>().atan2(s1.x.in_unit::<Meter>());
            let diff = (a1 - a0).abs();
            // Normalise to [0, π].
            let diff = if diff > PI { TWO_PI - diff } else { diff };
            assert!(
                (diff - step).abs() <= 1.0e-12,
                "view {v}: step {diff} vs expected {step}"
            );
        }
    }
}
