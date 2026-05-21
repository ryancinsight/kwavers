//! Deterministic 1024-element transcranial focused-bowl acquisition geometry.

use std::f64::consts::{FRAC_PI_2, TAU};

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::source::transducers::focused::{BowlConfig, BowlTransducer};

const GEOMETRY_UNIT_FREQUENCY_HZ: f64 = 1.0;
const GEOMETRY_UNIT_AMPLITUDE_PA: f64 = 1.0;

/// Cartesian position of one array element on the water-coupled bowl surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementPosition {
    pub x_m: f64,
    pub y_m: f64,
    pub z_m: f64,
}

/// Transcranial focused bowl used by the encoded slice model.
#[derive(Clone, Debug)]
pub struct TranscranialBowlGeometry {
    pub elements: Vec<ElementPosition>,
}

impl TranscranialBowlGeometry {
    /// Place `element_count` elements on a deterministic equal-area hemisphere.
    pub fn uniform(element_count: usize, radius_m: f64) -> KwaversResult<Self> {
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

        let config = BowlConfig::hemispherical(
            [0.0, 0.0, radius_m],
            [0.0, 0.0, 0.0],
            GEOMETRY_UNIT_FREQUENCY_HZ,
            GEOMETRY_UNIT_AMPLITUDE_PA,
        );
        let bowl = BowlTransducer::with_polar_span(config, FRAC_PI_2, element_count)?;
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

    /// Map azimuthal receiver offsets to nearest physical elements.
    pub(super) fn receiver_indices(&self, offsets: &[usize]) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.len() * offsets.len());
        for source_idx in 0..self.len() {
            for offset in offsets {
                let azimuth = TAU * *offset as f64 / self.len() as f64;
                indices.push(self.nearest_rotated_azimuth(source_idx, azimuth));
            }
        }
        indices
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

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
