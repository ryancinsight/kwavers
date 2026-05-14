//! Stacked Newton state-vector layout.
//!
//! Monolithic coupling stores participating fields in one `Array3<f64>` by
//! stacking full field volumes along axis 0.  This module owns that
//! representation: deterministic field ordering, pack/unpack, and zero-copy
//! field-block views over the stacked state.

mod block;
mod layout;

pub(in crate::solver::multiphysics::monolithic) use block::field_block_view;
pub(in crate::solver::multiphysics::monolithic) use layout::{
    flatten_fields, sorted_field_keys, unflatten_fields,
};

#[cfg(test)]
mod tests;
