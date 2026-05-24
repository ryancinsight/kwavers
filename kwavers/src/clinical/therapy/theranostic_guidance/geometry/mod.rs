//! Source and receiver placement for theranostic ultrasound arrays.

mod boundary;
mod focused_bowl;
mod layout;
mod types;

pub(crate) use types::IndexBounds3;
pub use types::{Point2, Point3};

pub(super) use boundary::{active_bounds_3d, centered_origin_2d, is_boundary_2d, is_boundary_3d};
pub(super) use focused_bowl::{
    focused_bowl_cap_points, FocusedBowlCapSpec, FocusedBowlVertexDirection,
};

pub use layout::{
    angle_span, build_device_layout, placement_metrics, DeviceLayout, DevicePlacementMetrics,
};
