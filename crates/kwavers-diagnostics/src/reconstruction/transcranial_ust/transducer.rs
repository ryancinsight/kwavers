//! Deterministic transcranial focused-bowl acquisition geometry.

use std::f64::consts::TAU;

use aequitas::systems::si::quantities::{Frequency, Length, Pressure};
use aequitas::systems::si::units::{Hertz, Meter, Pascal};
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
    ///
    /// # Errors
    ///
    /// Returns an error when the element count, radius, or angular aperture is
    /// invalid.
    pub fn from_aperture(
        element_count: usize,
        radius: Length<f64>,
        aperture: BowlAngularBounds,
    ) -> KwaversResult<Self> {
        if element_count < 8 {
            return Err(KwaversError::InvalidInput(
                "TranscranialBowlGeometry requires at least 8 elements".to_owned(),
            ));
        }
        let radius_m = radius.in_unit::<Meter>();
        if !radius_m.is_finite() || radius_m <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "TranscranialBowlGeometry radius must be finite and positive".to_owned(),
            ));
        }

        let config = BowlConfig::from_focus_axis(
            [Length::from_unit::<Meter>(0.0); 3],
            [0.0, 0.0, -1.0],
            Length::from_unit::<Meter>(radius_m),
            Length::from_unit::<Meter>(2.0 * radius_m),
            Frequency::from_unit::<Hertz>(GEOMETRY_UNIT_FREQUENCY_HZ),
            Pressure::from_unit::<Pascal>(GEOMETRY_UNIT_AMPLITUDE_PA),
        )?;
        let bowl = BowlTransducer::with_angular_bounds(config, aperture, element_count)?;
        let elements = bowl
            .element_positions()
            .iter()
            .map(|position| ElementPosition {
                x: position[0],
                y: position[1],
                z: position[2],
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
        let source_x = source.x.in_unit::<Meter>();
        let source_y = source.y.in_unit::<Meter>();
        let source_z = source.z.in_unit::<Meter>();
        let target = ElementPosition {
            x: Length::from_unit::<Meter>(cos_phi * source_x - sin_phi * source_y),
            y: Length::from_unit::<Meter>(sin_phi * source_x + cos_phi * source_y),
            z: Length::from_unit::<Meter>(source_z),
        };

        let mut best_idx = usize::from(source_idx == 0);
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
    let dx = a.x.in_unit::<Meter>() - b.x.in_unit::<Meter>();
    let dy = a.y.in_unit::<Meter>() - b.y.in_unit::<Meter>();
    let dz = a.z.in_unit::<Meter>() - b.z.in_unit::<Meter>();
    dx.powi(2) + dy.powi(2) + dz.powi(2)
}
