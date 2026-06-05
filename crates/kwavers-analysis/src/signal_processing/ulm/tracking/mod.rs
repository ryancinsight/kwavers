//! Multi-frame Microbubble Tracking via Hungarian Algorithm
//!
//! Links bubble detections across consecutive frames into continuous tracks using
//! the linear assignment problem (Hungarian algorithm, O(n³)).
//!
//! # References
//!
//! - Kuhn, H. W. (1955). The Hungarian method for the assignment problem.
//! - Munkres, J. (1957). Algorithms for the assignment and transportation problems.

mod tracker;
mod types;

#[cfg(test)]
mod tests;

pub use tracker::HungarianTracker;
pub use types::{BubbleTrack, TrackingConfig};
