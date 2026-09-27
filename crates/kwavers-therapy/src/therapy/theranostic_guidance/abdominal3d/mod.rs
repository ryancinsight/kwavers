pub(crate) mod bowl;
pub(super) mod placement;
pub(crate) mod placement_geometry;
pub(super) mod types;

pub use placement::plan_abdominal_array_placement;
pub use types::AbdominalArrayPlacement3D;

#[cfg(test)]
mod tests;
