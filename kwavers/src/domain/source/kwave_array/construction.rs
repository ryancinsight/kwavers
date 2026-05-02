//! Constructors and element-addition methods for [`KWaveArray`].

use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;

use super::{ArrayTransform, ElementShape, KWaveArray};

impl KWaveArray {
    /// Create a new empty `KWaveArray`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            frequency: 1e6,
            sound_speed: SOUND_SPEED_TISSUE,
            _element_width: 0.5e-3,
            array_transform: None,
        }
    }

    /// Create a new array with specified frequency and sound speed.
    #[must_use]
    pub fn with_params(frequency: f64, sound_speed: f64) -> Self {
        Self {
            elements: Vec::new(),
            frequency,
            sound_speed,
            _element_width: 0.5e-3,
            array_transform: None,
        }
    }

    /// Set the element width (for arc discretization).
    #[must_use]
    pub fn with_element_width(mut self, width: f64) -> Self {
        self._element_width = width;
        self
    }

    /// Get the operating frequency [Hz].
    #[must_use]
    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    /// Update the operating frequency while preserving existing elements.
    pub fn set_frequency(&mut self, frequency: f64) {
        self.frequency = frequency;
    }

    /// Update the sound speed while preserving existing elements.
    pub fn set_sound_speed(&mut self, sound_speed: f64) {
        self.sound_speed = sound_speed;
    }

    /// Install a global translation + intrinsic X-Y-Z Euler rotation (degrees)
    /// applied to every element at rasterization time. Mirrors
    /// k-wave-python's `kWaveArray.set_array_position(translation, rotation)`.
    /// Passing zero for both is equivalent to the identity transform.
    pub fn set_array_position(
        &mut self,
        translation: (f64, f64, f64),
        euler_xyz_deg: (f64, f64, f64),
    ) {
        self.array_transform = Some(ArrayTransform {
            translation,
            euler_xyz_deg,
        });
    }

    /// Remove the global array transform if one was previously installed.
    pub fn clear_array_position(&mut self) {
        self.array_transform = None;
    }

    /// Add an arc-shaped element with default ±45° angular span.
    ///
    /// # Arguments
    /// * `position` - Arc center `[x, y, z]` in metres
    /// * `radius`   - Arc radius in metres
    /// * `diameter` - Element diameter in metres
    pub fn add_arc_element(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameter: f64,
    ) -> &mut Self {
        self.elements.push(ElementShape::Arc {
            position,
            radius,
            diameter,
            start_angle: -45.0,
            end_angle: 45.0,
        });
        self
    }

    /// Add an arc-shaped element with custom angular span.
    ///
    /// # Arguments
    /// * `position`    - Arc center `[x, y, z]` in metres
    /// * `radius`      - Arc radius in metres
    /// * `diameter`    - Element diameter in metres
    /// * `start_angle` - Start angle in degrees
    /// * `end_angle`   - End angle in degrees
    pub fn add_arc_element_with_angles(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameter: f64,
        start_angle: f64,
        end_angle: f64,
    ) -> &mut Self {
        self.elements.push(ElementShape::Arc {
            position,
            radius,
            diameter,
            start_angle,
            end_angle,
        });
        self
    }

    /// Add an axis-aligned rectangular element.
    ///
    /// # Arguments
    /// * `position` - Center position `[x, y, z]` in metres
    /// * `width`    - Width in x-direction [m]
    /// * `height`   - Height in y-direction [m]
    /// * `length`   - Length in z-direction [m]
    pub fn add_rect_element(
        &mut self,
        position: (f64, f64, f64),
        width: f64,
        height: f64,
        length: f64,
    ) -> &mut Self {
        self.elements.push(ElementShape::Rect {
            position,
            width,
            height,
            length,
            euler_xyz_deg: (0.0, 0.0, 0.0),
        });
        self
    }

    /// Add a rectangular element rotated about its center by intrinsic X-Y-Z
    /// Euler angles (degrees). Matches the upstream k-wave-python
    /// `KWaveArray.add_rect_element` rotation contract used by the linear
    /// array transducer example.
    pub fn add_rect_rot_element(
        &mut self,
        position: (f64, f64, f64),
        width: f64,
        height: f64,
        length: f64,
        euler_xyz_deg: (f64, f64, f64),
    ) -> &mut Self {
        self.elements.push(ElementShape::Rect {
            position,
            width,
            height,
            length,
            euler_xyz_deg,
        });
        self
    }

    /// Add a disc-shaped element.
    ///
    /// # Arguments
    /// * `position`       - Center position `[x, y, z]` in metres
    /// * `diameter`       - Disc diameter in metres
    /// * `focus_position` - Optional focus point defining the disc normal
    pub fn add_disc_element(
        &mut self,
        position: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
    ) -> &mut Self {
        self.elements.push(ElementShape::Disc {
            position,
            diameter,
            focus_position,
        });
        self
    }

    /// Add a bowl-shaped (focused) element.
    ///
    /// # Arguments
    /// * `position` - Bowl center position `[x, y, z]` in metres
    /// * `radius`   - Radius of curvature [m]
    /// * `diameter` - Bowl aperture diameter [m]
    pub fn add_bowl_element(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameter: f64,
    ) -> &mut Self {
        self.elements.push(ElementShape::Bowl {
            position,
            radius,
            diameter,
        });
        self
    }

    /// Add an annular spherical-cap element — a bowl section bounded by inner
    /// and outer aperture diameters. Mirrors k-wave-python's
    /// `add_annular_element`. Uses the same orientation convention as
    /// `add_bowl_element`.
    pub fn add_annular_element(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        inner_diameter: f64,
        outer_diameter: f64,
    ) -> &mut Self {
        assert!(
            outer_diameter > inner_diameter && inner_diameter >= 0.0,
            "annulus requires 0 ≤ inner_diameter < outer_diameter \
             (got inner={inner_diameter}, outer={outer_diameter})",
        );
        self.elements.push(ElementShape::Annulus {
            position,
            radius,
            inner_diameter,
            outer_diameter,
        });
        self
    }

    /// Add a concentric annular array (sequence of annuli sharing a common
    /// center of curvature). Each `(inner_diameter, outer_diameter)` pair
    /// becomes one `ElementShape::Annulus`, matching k-wave-python's
    /// `add_annular_array`.
    pub fn add_annular_array(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameters: &[(f64, f64)],
    ) -> &mut Self {
        for &(inner_d, outer_d) in diameters {
            self.add_annular_element(position, radius, inner_d, outer_d);
        }
        self
    }
}
