//! Brain Atlas Integration
//!
//! Provides stereotactic coordinate systems and anatomical reference data
//! for functional ultrasound imaging.
//!
//! References:
//! - Allen Brain Atlas: <http://mouse.brain-map.org>
//! - Paxinos & Watson (2013). *The Rat Brain in Stereotaxic Coordinates*
//! - Franklin & Paxinos (2008). *The Mouse Brain in Stereotaxic Coordinates*
//!
//! Partitioned by responsibility:
//! - `construction` — constructors, default phantom, private geometry helpers.
//! - `query`        — coordinate conversions and accessor methods.

mod construction;
mod query;
#[cfg(test)]
mod tests;

use leto::Array3 as LetoArray3;
use leto::Array3;

/// Brain atlas reference data.
#[derive(Debug, Clone)]
pub struct BrainAtlas {
    /// Reference image (template).
    pub(super) reference_image: LetoArray3<f64>,
    /// Brain region annotations.
    pub(super) annotation: Array3<u32>,
    /// Voxel size (mm).
    pub(super) voxel_size: [f64; 3],
    /// Brain center coordinates (mm).
    pub(super) brain_center: [f64; 3],
    /// Atlas grid shape.
    pub(super) shape: (usize, usize, usize),
}
