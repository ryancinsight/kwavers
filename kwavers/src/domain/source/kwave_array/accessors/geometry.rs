use super::super::{ElementShape, KWaveArray};

impl KWaveArray {
    /// Number of elements in the array.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Centroids of all elements with the global array transform applied.
    #[must_use]
    pub fn get_element_positions(&self) -> Vec<(f64, f64, f64)> {
        self.elements
            .iter()
            .map(|e| {
                let local = match e {
                    ElementShape::Arc { position, .. } => *position,
                    ElementShape::Rect { position, .. } => *position,
                    ElementShape::Disc { position, .. } => *position,
                    ElementShape::Bowl { position, .. } => *position,
                    ElementShape::Annulus { position, .. } => *position,
                };
                self.apply_transform_point(local)
            })
            .collect()
    }

    /// Compute the total geometric measure of all elements.
    ///
    /// Bowl and disc elements contribute area; arc elements contribute arc
    /// length; rect elements contribute `width × height`; annuli contribute
    /// the spherical-ring area.
    #[must_use]
    pub fn compute_total_surface_area(&self) -> f64 {
        self.elements
            .iter()
            .map(|e| match e {
                ElementShape::Bowl {
                    radius: r,
                    diameter: d,
                    ..
                } => Self::bowl_surface_area(*r, *d),
                ElementShape::Disc { diameter: d, .. } => std::f64::consts::PI * (d / 2.0).powi(2),
                ElementShape::Rect {
                    width: w,
                    height: h,
                    ..
                } => w * h,
                ElementShape::Arc {
                    radius: r,
                    start_angle: s,
                    end_angle: e,
                    ..
                } => Self::arc_line_length(*r, *s, *e),
                ElementShape::Annulus {
                    radius: r,
                    inner_diameter: di,
                    outer_diameter: d_o,
                    ..
                } => Self::annulus_surface_area(*r, *di, *d_o),
            })
            .sum()
    }
}
