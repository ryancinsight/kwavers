//! Line-search update for source-encoded multiparameter nonlinear 3-D FWI.

mod candidate;
mod objective;
mod search;
#[cfg(test)]
mod tests;
mod types;

pub(super) use objective::objective_for_model;
pub(super) use search::apply_line_search;
pub(super) use types::{LineSearchInput, LineSearchWorkspace, ObjectiveInput};
