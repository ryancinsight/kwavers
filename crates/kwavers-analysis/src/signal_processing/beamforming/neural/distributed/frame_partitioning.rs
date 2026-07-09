//! Frame partitioning helpers for distributed neural beamforming.
//!
//! This module owns the scheduling primitives for frame-major RF volumes:
//! active-processor selection, bounded partition sizing, and chunk metadata.

use kwavers_solver::interface::pinn_beamforming::{
    DistributedConfig, LoadBalancingStrategy, PinnBeamformingDecompositionStrategy,
};
use leto::Array3;
use std::ops::Range;

/// Chunk result produced by a distributed worker.
#[derive(Debug)]
pub(crate) struct DistributedChunkResult {
    pub start: usize,
    pub processor_index: usize,
    pub processing_time_ms: f64,
    pub volume: Array3<f32>,
    pub uncertainty: Array3<f32>,
    pub confidence: Array3<f32>,
}

pub(crate) fn active_processor_indices(processor_count: usize, gpu_health: &[bool]) -> Vec<usize> {
    if gpu_health.len() == processor_count {
        let active: Vec<usize> = gpu_health
            .iter()
            .enumerate()
            .filter_map(|(idx, healthy)| healthy.then_some(idx))
            .collect();

        if !active.is_empty() {
            return active;
        }
    }

    (0..processor_count).collect()
}

pub(crate) fn equal_partition_sizes(frame_count: usize, worker_count: usize) -> Vec<usize> {
    if worker_count == 0 {
        return Vec::new();
    }

    let base = frame_count / worker_count;
    let remainder = frame_count % worker_count;

    (0..worker_count)
        .map(|worker_idx| base + usize::from(worker_idx < remainder))
        .collect()
}

/// Weighted partition sizes.
/// # Panics
/// - Panics if an internal precondition is violated.
///
pub(crate) fn weighted_partition_sizes(
    frame_count: usize,
    active_indices: &[usize],
    batch_size: usize,
    gpu_load: &[f32],
) -> Vec<usize> {
    let mut weights: Vec<f64> = active_indices
        .iter()
        .map(|&idx| {
            let load = gpu_load.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0) as f64;
            1.0 - load
        })
        .collect();

    if weights.iter().all(|&weight| weight <= 0.0) {
        weights.fill(1.0);
    }

    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return equal_partition_sizes(frame_count, active_indices.len());
    }

    let mut sizes = Vec::with_capacity(weights.len());
    let mut remainders = Vec::with_capacity(weights.len());

    for &weight in &weights {
        let exact = frame_count as f64 * weight / total_weight;
        let floored = exact.floor() as usize;
        sizes.push(floored.min(batch_size));
        remainders.push(exact - floored as f64);
    }

    let mut remaining = frame_count.saturating_sub(sizes.iter().sum::<usize>());

    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by(|left, right| {
        remainders[*right]
            .total_cmp(&remainders[*left])
            .then_with(|| weights[*right].total_cmp(&weights[*left]))
            .then_with(|| left.cmp(right))
    });

    for idx in order {
        if remaining == 0 {
            break;
        }

        if sizes[idx] >= batch_size {
            continue;
        }

        let available = batch_size - sizes[idx];
        let add = available.min(remaining);
        sizes[idx] += add;
        remaining -= add;
    }

    if remaining > 0 {
        for size in sizes.iter_mut() {
            if remaining == 0 {
                break;
            }

            if *size >= batch_size {
                continue;
            }

            let available = batch_size - *size;
            let add = available.min(remaining);
            *size += add;
            remaining -= add;
        }
    }

    debug_assert_eq!(remaining, 0);
    sizes
}

pub(crate) fn partition_round_sizes(
    config: &DistributedConfig,
    frame_count: usize,
    active_indices: &[usize],
    batch_size: usize,
    gpu_load: &[f32],
) -> Vec<usize> {
    match (config.decomposition, config.load_balancing) {
        (
            PinnBeamformingDecompositionStrategy::Hybrid,
            LoadBalancingStrategy::Dynamic | LoadBalancingStrategy::Adaptive,
        ) => weighted_partition_sizes(frame_count, active_indices, batch_size, gpu_load),
        _ => equal_partition_sizes(frame_count, active_indices.len()),
    }
}

pub(crate) fn build_ranges(start: usize, sizes: &[usize]) -> Vec<Range<usize>> {
    let mut cursor = start;
    let mut ranges = Vec::with_capacity(sizes.len());

    for &size in sizes {
        let end = cursor + size;
        ranges.push(cursor..end);
        cursor = end;
    }

    ranges
}
