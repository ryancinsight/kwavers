//! Reverse-time migration for the planar seismic workflow.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_solver::inverse::fwi::time_domain::FwiGeometry;
use kwavers_solver::inverse::seismic::{
    parameters::{ImagingCondition, RtmSettings, SeismicBoundaryType, StorageStrategy},
    rtm::RtmProcessor,
};
use leto::{Array2, Array3};

use super::{NX, NY, NZ};

/// Migrate the first observed gather into a normalized zero-lag image.
pub(super) fn run_rtm(
    shots: &[(FwiGeometry, Array2<f64>)],
    grid: &Grid,
) -> KwaversResult<Array3<f64>> {
    let Some((geometry, observations)) = shots.first() else {
        return Err(KwaversError::InvalidInput(
            "RTM requires at least one observed shot".to_owned(),
        ));
    };

    let trace_count = observations.shape()[0];
    let active_receiver_count = geometry
        .sensor_mask
        .iter()
        .filter(|&&active| active)
        .count();
    if active_receiver_count != trace_count {
        return Err(KwaversError::InvalidInput(format!(
            "RTM receiver mismatch: sensor mask has {active_receiver_count} active receivers, observations have {trace_count} traces"
        )));
    }

    let mut receiver_snapshot = Array3::<f64>::zeros((NX, NY, NZ));
    let mut receiver_index = 0;
    for ([i, _j, k], &active) in geometry.sensor_mask.indexed_iter() {
        if !active {
            continue;
        }
        let trace = observations
            .index_axis::<1>(0, receiver_index)
            .map_err(|error| {
                KwaversError::InvalidInput(format!("RTM receiver trace access failed: {error}"))
            })?;
        let sample_count = trace.shape()[0].max(1);
        let rms =
            (trace.iter().map(|&sample| sample * sample).sum::<f64>() / sample_count as f64).sqrt();
        receiver_snapshot[[i, 0, k]] = rms;
        receiver_index += 1;
    }

    let settings = RtmSettings {
        imaging_condition: ImagingCondition::Normalized,
        storage_strategy: StorageStrategy::Full,
        boundary_type: SeismicBoundaryType::Absorbing,
        apply_laplacian: true,
    };
    let image = RtmProcessor::new(settings)
        .migrate(&receiver_snapshot, &receiver_snapshot, grid)
        .map_err(|error| KwaversError::InvalidInput(format!("RTM migration failed: {error:#}")))?;
    let peak = image.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    println!("  RTM image completed — peak amplitude: {peak:.4}");
    Ok(image)
}
