//! Provider-owned traversal adapters for therapy kernels.

mod dual_output;
mod single_output;
mod triple_output;

pub(crate) use dual_output::{zip_two_mut_four_refs, zip_two_mut_ref, zip_two_mut_two_refs};
pub(crate) use single_output::{
    zip_mut_five_refs, zip_mut_four_refs, zip_mut_ref, zip_mut_three_refs,
};
pub(crate) use triple_output::{zip_three_mut_three_refs, zip_three_mut_two_refs};

const THERAPY_CHUNK_SIZE: usize = 4096;

/// Row-major odometer increment of a logical multi-index; returns `false` once
/// the index wraps past the final element. Drives the strided fallback walks.
#[inline]
fn next_index<const N: usize>(index: &mut [usize; N], shape: &[usize; N]) -> bool {
    for d in (0..N).rev() {
        index[d] += 1;
        if index[d] < shape[d] {
            return true;
        }
        index[d] = 0;
    }
    false
}
