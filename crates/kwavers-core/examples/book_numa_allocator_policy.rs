//! Book example: NUMA policy-aware arena allocation.
//!
//! This example keeps the compute-layer placement contract explicit at the
//! allocation seam while preserving the `kwavers-core` domain boundary.

use kwavers_core::arena::{ArenaLayoutNumaPolicy, NumaAwareAllocator, NumaNodeId, CACHE_LINE_SIZE};
use kwavers_core::error::KwaversResult;

fn main() -> KwaversResult<()> {
    let elements = 1024usize;
    let bytes = elements * std::mem::size_of::<f32>();

    // First-touch placement: pages are touched in parallel to establish locality.
    let mut first_touch = NumaAwareAllocator::with_policy(ArenaLayoutNumaPolicy::FirstTouch);
    let first_touch_ptr = first_touch.allocate(bytes, CACHE_LINE_SIZE)?;
    first_touch.first_touch_parallel(first_touch_ptr.cast::<f32>(), elements);

    // Explicit bind policy: on Linux this requests mbind(2) for node 0.
    // Non-Linux platforms fall back to first-touch behavior.
    let mut bound =
        NumaAwareAllocator::with_policy(ArenaLayoutNumaPolicy::BindToNode(NumaNodeId::new(0)));
    let _bound_ptr = bound.allocate(bytes, CACHE_LINE_SIZE)?;

    Ok(())
}
