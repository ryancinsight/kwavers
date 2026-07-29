//! Integration tests for narrowband beamforming pipeline.
//!
//! Validates end-to-end pipeline: steering → snapshots → Capon spectrum.

mod test_data;
mod invariance;
mod pipeline;
mod snapshot_consistency;
mod steering_unit;

