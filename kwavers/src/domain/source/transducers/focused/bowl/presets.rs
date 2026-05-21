//! Focused bowl configuration constructors.

use super::BowlConfig;

impl BowlConfig {
    /// Build a spherical-cap bowl from its vertex and acoustic focus.
    ///
    /// The radius of curvature is the Euclidean distance from `vertex_m` to
    /// `focus_m`. `diameter_m` controls the cap aperture; use
    /// [`Self::hemispherical`] when the intended aperture is a full hemisphere.
    #[must_use]
    pub fn from_vertex_focus(
        vertex_m: [f64; 3],
        focus_m: [f64; 3],
        diameter_m: f64,
        frequency_hz: f64,
        amplitude_pa: f64,
    ) -> Self {
        Self {
            radius_of_curvature: distance3(vertex_m, focus_m),
            diameter: diameter_m,
            center: vertex_m,
            focus: focus_m,
            frequency: frequency_hz,
            amplitude: amplitude_pa,
            ..Self::default()
        }
    }

    /// Build a hemispherical focused bowl from its vertex and acoustic focus.
    ///
    /// This source-domain constructor supports fixed-count hemispherical arrays
    /// without naming the source after a clinical anatomy or device topology.
    #[must_use]
    pub fn hemispherical(
        vertex_m: [f64; 3],
        focus_m: [f64; 3],
        frequency_hz: f64,
        amplitude_pa: f64,
    ) -> Self {
        let radius_m = distance3(vertex_m, focus_m);
        Self::from_vertex_focus(
            vertex_m,
            focus_m,
            2.0 * radius_m,
            frequency_hz,
            amplitude_pa,
        )
    }

    /// Return a config using the requested discretization element size [m].
    #[must_use]
    pub fn with_element_size(mut self, element_size_m: f64) -> Self {
        self.element_size = Some(element_size_m);
        self
    }
}

fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0])
        .mul_add(
            a[0] - b[0],
            (a[1] - b[1]).mul_add(a[1] - b[1], (a[2] - b[2]) * (a[2] - b[2])),
        )
        .sqrt()
}
