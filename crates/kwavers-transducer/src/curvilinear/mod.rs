//! Curvilinear (convex) transducer array geometry.
//!
//! A curvilinear array — the clinical abdominal/curved probe — places its
//! elements on a **convex circular arc** of radius `R_c` (the radius of
//! curvature) rather than on a flat line. Each element faces radially outward
//! from the centre of curvature, so the array insonifies a diverging sector and
//! gives a wide field of view from a small footprint.
//!
//! This module is the geometry single-source-of-truth: it produces, for an
//! `N`-element convex array, each element's centre position, outward normal, and
//! along-array tangent, plus transmit-focusing delays. The positions/normals
//! feed the grid-rasterizing [`crate::kwave_array`] element model (`Rect`/`Arc`
//! elements oriented by the normals) or a [`kwavers_source::Source`].
//!
//! # Geometry
//!
//! The array apex sits at the origin facing `+z`; the centre of curvature is at
//! `C = (0, 0, −R_c)`. Element `i` subtends angle
//! `θ_i = (i − (N−1)/2)·Δθ` about `C`, giving
//!
//! ```text
//! position(i) = ( R_c·sin θ_i , 0 , R_c·(cos θ_i − 1) )
//! normal(i)   = ( sin θ_i , 0 , cos θ_i )          (outward radial, unit)
//! tangent(i)  = ( cos θ_i , 0 , −sin θ_i )         (along the array, unit)
//! ```
//!
//! so `‖position(i) − C‖ = R_c` exactly and the apex element (`θ = 0`) is at the
//! origin facing `+z`. The lateral axis is `x`; the array is uniform in `y`
//! (elevation), matching a 1-D curved probe.
//!
//! # References
//! - Szabo, T. L. (2014). *Diagnostic Ultrasound Imaging*, 2nd ed., §7
//!   (array geometries). Curved/convex arrays.

use kwavers_core::error::{KwaversError, KwaversResult};

/// A convex (curvilinear) transducer-array layout on a circular arc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConvexArrayGeometry {
    radius_of_curvature: f64,
    num_elements: usize,
    angular_pitch: f64,
}

impl ConvexArrayGeometry {
    /// Build from an explicit angular pitch `Δθ` [rad] between adjacent elements.
    ///
    /// # Errors
    /// - [`KwaversError::InvalidInput`] if `radius_of_curvature ≤ 0`,
    ///   `num_elements == 0`, or `angular_pitch` is non-finite/`≤ 0`.
    pub fn from_angular_pitch(
        radius_of_curvature: f64,
        num_elements: usize,
        angular_pitch: f64,
    ) -> KwaversResult<Self> {
        if !radius_of_curvature.is_finite() || radius_of_curvature <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "ConvexArrayGeometry requires radius_of_curvature > 0, got {radius_of_curvature}"
            )));
        }
        if num_elements == 0 {
            return Err(KwaversError::InvalidInput(
                "ConvexArrayGeometry requires num_elements > 0".to_owned(),
            ));
        }
        if !angular_pitch.is_finite() || angular_pitch <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "ConvexArrayGeometry requires angular_pitch > 0, got {angular_pitch}"
            )));
        }
        Ok(Self {
            radius_of_curvature,
            num_elements,
            angular_pitch,
        })
    }

    /// Build from an arc-length pitch (centre-to-centre along the arc) [m].
    ///
    /// The angular pitch is `Δθ = arc_pitch / R_c`.
    ///
    /// # Errors
    /// - [`KwaversError::InvalidInput`] on the same conditions as
    ///   [`Self::from_angular_pitch`], with `arc_pitch_m > 0`.
    pub fn from_arc_pitch(
        radius_of_curvature: f64,
        num_elements: usize,
        arc_pitch_m: f64,
    ) -> KwaversResult<Self> {
        if !arc_pitch_m.is_finite() || arc_pitch_m <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "ConvexArrayGeometry requires arc_pitch_m > 0, got {arc_pitch_m}"
            )));
        }
        Self::from_angular_pitch(
            radius_of_curvature,
            num_elements,
            arc_pitch_m / radius_of_curvature,
        )
    }

    /// Build from the total angular span `Θ` [rad] of the aperture.
    ///
    /// Distributes `num_elements` over `[−Θ/2, +Θ/2]`, so `Δθ = Θ / (N−1)`. For a
    /// single element the angular pitch is unused; a sentinel `Θ` is required `> 0`.
    ///
    /// # Errors
    /// - [`KwaversError::InvalidInput`] if `num_elements < 2` (use
    ///   [`Self::from_angular_pitch`] for a single element) or `total_angle ≤ 0`.
    pub fn from_total_angle(
        radius_of_curvature: f64,
        num_elements: usize,
        total_angle_rad: f64,
    ) -> KwaversResult<Self> {
        if num_elements < 2 {
            return Err(KwaversError::InvalidInput(
                "ConvexArrayGeometry::from_total_angle requires num_elements >= 2".to_owned(),
            ));
        }
        if !total_angle_rad.is_finite() || total_angle_rad <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "ConvexArrayGeometry requires total_angle_rad > 0, got {total_angle_rad}"
            )));
        }
        Self::from_angular_pitch(
            radius_of_curvature,
            num_elements,
            total_angle_rad / (num_elements - 1) as f64,
        )
    }

    /// Radius of curvature `R_c` [m].
    #[must_use]
    pub fn radius_of_curvature(&self) -> f64 {
        self.radius_of_curvature
    }

    /// Number of elements.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    /// Angular pitch `Δθ` [rad].
    #[must_use]
    pub fn angular_pitch(&self) -> f64 {
        self.angular_pitch
    }

    /// Centre of curvature `C = (0, 0, −R_c)`.
    #[must_use]
    pub fn curvature_center(&self) -> [f64; 3] {
        [0.0, 0.0, -self.radius_of_curvature]
    }

    /// Subtended angle `θ_i` [rad] of element `i` about the centre of curvature
    /// (symmetric about the apex: `θ = 0` at the centre element).
    #[must_use]
    pub fn element_angle(&self, i: usize) -> f64 {
        (i as f64 - (self.num_elements as f64 - 1.0) / 2.0) * self.angular_pitch
    }

    /// Centre position of element `i` [m].
    #[must_use]
    pub fn element_position(&self, i: usize) -> [f64; 3] {
        let theta = self.element_angle(i);
        let r = self.radius_of_curvature;
        [r * theta.sin(), 0.0, r * (theta.cos() - 1.0)]
    }

    /// Outward (radial) unit normal of element `i`.
    #[must_use]
    pub fn element_normal(&self, i: usize) -> [f64; 3] {
        let theta = self.element_angle(i);
        [theta.sin(), 0.0, theta.cos()]
    }

    /// Along-array unit tangent of element `i` (direction of increasing index).
    #[must_use]
    pub fn element_tangent(&self, i: usize) -> [f64; 3] {
        let theta = self.element_angle(i);
        [theta.cos(), 0.0, -theta.sin()]
    }

    /// All element centre positions.
    #[must_use]
    pub fn positions(&self) -> Vec<[f64; 3]> {
        (0..self.num_elements)
            .map(|i| self.element_position(i))
            .collect()
    }

    /// All element outward normals.
    #[must_use]
    pub fn normals(&self) -> Vec<[f64; 3]> {
        (0..self.num_elements)
            .map(|i| self.element_normal(i))
            .collect()
    }

    /// Arc-length pitch (centre-to-centre along the arc) `R_c·Δθ` [m].
    #[must_use]
    pub fn arc_pitch(&self) -> f64 {
        self.radius_of_curvature * self.angular_pitch
    }

    /// Total angular span `(N−1)·Δθ` [rad].
    #[must_use]
    pub fn total_angular_span(&self) -> f64 {
        (self.num_elements.saturating_sub(1)) as f64 * self.angular_pitch
    }

    /// Lateral aperture width (chord across the array) `2 R_c·sin(span/2)` [m].
    #[must_use]
    pub fn aperture_width(&self) -> f64 {
        2.0 * self.radius_of_curvature * (self.total_angular_span() / 2.0).sin()
    }

    /// Transmit-focusing time delays [s] for a focal point.
    ///
    /// Returns `delay_i = (d_max − d_i) / c`, where `d_i = ‖position(i) − focal‖`
    /// and `d_max = maxᵢ d_i`. The farthest element fires first (delay 0) and the
    /// nearest fires last, so the emitted wavefronts arrive at the focus
    /// simultaneously. All delays are `≥ 0`.
    ///
    /// # Errors
    /// - [`KwaversError::InvalidInput`] if `sound_speed` is non-finite/`≤ 0`.
    pub fn focusing_delays(
        &self,
        focal_point: [f64; 3],
        sound_speed: f64,
    ) -> KwaversResult<Vec<f64>> {
        if !sound_speed.is_finite() || sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "focusing_delays requires sound_speed > 0, got {sound_speed}"
            )));
        }
        let distances: Vec<f64> = (0..self.num_elements)
            .map(|i| {
                let p = self.element_position(i);
                let dx = p[0] - focal_point[0];
                let dy = p[1] - focal_point[1];
                let dz = p[2] - focal_point[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();
        let d_max = distances.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Ok(distances
            .iter()
            .map(|&d| (d_max - d) / sound_speed)
            .collect())
    }
}

#[cfg(test)]
mod tests;
