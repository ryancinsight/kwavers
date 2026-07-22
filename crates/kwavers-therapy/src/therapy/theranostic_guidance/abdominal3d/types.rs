use super::super::geometry::Point3;

/// 3-D geometry of a focused bowl array placed at the skin surface.
#[derive(Clone, Debug)]
pub struct AbdominalArrayPlacement3D {
    /// Sub-sampled exterior skin surface of the body `m`.
    pub body_surface_points_m: Vec<Point3>,
    /// Sub-sampled exterior surface of the target organ `m`.
    pub organ_surface_points_m: Vec<Point3>,
    /// Element positions on the concave bowl, outside the skin `m`.
    pub therapy_elements_m: Vec<Point3>,
    /// Start points of beam visualisation rays (= element positions) `m`.
    pub beam_start_points_m: Vec<Point3>,
    /// End points of beam visualisation rays (= focus) `m`.
    pub beam_end_points_m: Vec<Point3>,
    /// Organ centroid (geometric focus target) `m`.
    pub focus_m: Point3,
    /// Nearest exterior skin contact point to the focus `m`.
    pub skin_contact_m: Point3,
    /// Spherical bowl radius (focal length) `m`.
    pub transducer_radius_m: f64,
    /// Label of the anatomy: "liver" or "kidney".
    pub anatomy_label: String,
}
