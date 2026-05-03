//! Integration tests for narrowband beamforming pipeline.
//!
//! Validates end-to-end pipeline: steering → snapshots → Capon spectrum.

mod helpers;
mod invariance;
mod pipeline;
mod snapshot_consistency;
mod steering_unit;
