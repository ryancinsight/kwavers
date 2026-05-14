//! Curved-array acquisition geometry for 2-D straight-ray shift imaging.

mod geometry;
mod sampling;
#[cfg(test)]
mod tests;
mod validation;

pub use geometry::CurvedArray2d;
pub use sampling::CurvedArrayShiftScan;
