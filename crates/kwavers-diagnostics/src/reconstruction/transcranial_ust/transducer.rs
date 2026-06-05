//! Deterministic transcranial focused-bowl acquisition geometry.

use std::f64::consts::TAU;

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_transducer::transducers::focused::{BowlAngularBounds, BowlConfig, BowlTransducer};
use kwavers_transducer::transducers::{ElementPosition, TransducerGeometry};

const GEOMETRY_UNIT_FREQUENCY_HZ: f64 = 1.0;
const GEOMETRY_UNIT_AMPLITUDE_PA: f64 = 1.0;

/// Transcranial focused bowl used by the encoded slice model.
#[derive(Clone, Debug)]
pub struct TranscranialBowlGeometry {
    pub elements: Vec<ElementPosition>,
}

impl TransducerGeometry for TranscranialBowlGeometry {
    fn elements(&self) -> &[ElementPosition] {
        &self.elements
    }

    fn receiver_indices(&self, offsets: &[usize]) -> Vec<usize> {
        // Override the default cyclic offset mapping: a transcranial bowl
        // has continuous rotational symmetry about the cap axis, so the
        // natural receiver for "source `s` at offset `q`" is the element
        // closest to source `s` rotated by `2π · q / N` around the z-axis.
        let mut indices = Vec::with_capacity(self.len() * offsets.len());
        for source_idx in 0..self.len() {
            for offset in offsets {
                let azimuth = TAU * *offset as f64 / self.len() as f64;
                indices.push(self.nearest_rotated_azimuth(source_idx, azimuth));
            }
        }
        indices
    }
}

impl TranscranialBowlGeometry {
    /// Place `element_count` elements on a deterministic equal-area bowl aperture.
    pub fn from_aperture(
        element_count: usize,
        radius_m: f64,
        aperture: BowlAngularBounds,
    ) -> KwaversResult<Self> {
        if element_count < 8 {
            return Err(KwaversError::InvalidInput(
                "TranscranialBowlGeometry requires at least 8 elements".to_owned(),
            ));
        }
        if !radius_m.is_finite() || radius_m <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "TranscranialBowlGeometry radius must be finite and positive".to_owned(),
            ));
        }

        let config = BowlConfig::from_focus_axis(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            radius_m,
            2.0 * radius_m,
            GEOMETRY_UNIT_FREQUENCY_HZ,
            GEOMETRY_UNIT_AMPLITUDE_PA,
        )?;
        let bowl = BowlTransducer::with_angular_bounds(config, aperture, element_count)?;
        let elements = bowl
            .element_positions()
            .iter()
            .map(|position| ElementPosition {
                x_m: position[0],
                y_m: position[1],
                z_m: position[2],
            })
            .collect();
        Ok(Self { elements })
    }

    // `len()`, `is_empty()`, and `receiver_indices()` are now provided by the
    // `TransducerGeometry` trait impl below (the bowl-specific azimuthal-
    // rotation override of `receiver_indices` lives there).

    fn nearest_rotated_azimuth(&self, source_idx: usize, azimuth: f64) -> usize {
        let source = self.elements[source_idx];
        let (sin_phi, cos_phi) = azimuth.sin_cos();
        let target = ElementPosition {
            x_m: cos_phi * source.x_m - sin_phi * source.y_m,
            y_m: sin_phi * source.x_m + cos_phi * source.y_m,
            z_m: source.z_m,
        };

        let mut best_idx = if source_idx == 0 { 1 } else { 0 };
        let mut best_dist = squared_distance(self.elements[best_idx], target);
        for (idx, element) in self.elements.iter().copied().enumerate() {
            if idx == source_idx {
                continue;
            }
            let dist = squared_distance(element, target);
            if dist < best_dist {
                best_idx = idx;
                best_dist = dist;
            }
        }
        best_idx
    }
}

fn squared_distance(a: ElementPosition, b: ElementPosition) -> f64 {
    (a.x_m - b.x_m).powi(2) + (a.y_m - b.y_m).powi(2) + (a.z_m - b.z_m).powi(2)
}
