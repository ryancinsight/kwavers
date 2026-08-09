//! Atlas parallel-provider adapters for physics field traversal.

mod indexed;
mod zip;

pub(crate) use indexed::{
    for_each_indexed_mut, for_each_indexed_mut_four_refs, for_each_indexed_mut_three_refs,
    for_each_indexed_pair_mut, for_each_indexed_three_mut,
};
pub(crate) use zip::{
    zip_mut_four_refs, zip_mut_ref, zip_mut_three_refs, zip_mut_two_refs, zip_two_mut_four_refs,
    zip_two_mut_two_refs,
};

/// Element count per chunk for the chunked parallel traversal kernels.
const FIELD_CHUNK_SIZE: usize = 1024;
