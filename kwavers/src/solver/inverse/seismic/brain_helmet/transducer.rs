//! Deterministic 1024-element hemispherical acquisition geometry.

use std::f64::consts::{PI, TAU};

use crate::core::error::{KwaversError, KwaversResult};

/// Cartesian position of one array element on the water-coupled helmet surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementPosition {
    pub x_m: f64,
    pub y_m: f64,
    pub z_m: f64,
}

/// INSIGHTEC-style hemispherical cap used by the encoded slice model.
#[derive(Clone, Debug)]
pub struct HelmetHemisphereGeometry {
    pub elements: Vec<ElementPosition>,
}

impl HelmetHemisphereGeometry {
    /// Place `element_count` elements on a deterministic equal-area hemisphere.
    pub fn uniform(element_count: usize, radius_m: f64) -> KwaversResult<Self> {
        if element_count < 8 {
            return Err(KwaversError::InvalidInput(
                "HelmetHemisphereGeometry requires at least 8 elements".to_owned(),
            ));
        }
        if !radius_m.is_finite() || radius_m <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "HelmetHemisphereGeometry radius must be finite and positive".to_owned(),
            ));
        }

        let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
        let elements = (0..element_count)
            .map(|idx| {
                let z_m = radius_m * (idx as f64 + 0.5) / element_count as f64;
                let radial_m = (radius_m.mul_add(radius_m, -z_m * z_m)).max(0.0).sqrt();
                let phi = golden_angle * idx as f64;
                ElementPosition {
                    x_m: radial_m * phi.cos(),
                    y_m: radial_m * phi.sin(),
                    z_m,
                }
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
