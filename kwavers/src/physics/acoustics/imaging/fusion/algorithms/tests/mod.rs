// Re-export parent scope so submodule test files use `super::*` uniformly.
pub(super) use super::*;
pub(super) use ndarray::Array3;

mod deep_fusion;
mod feature_based;
mod intensity_projection;
mod lifecycle;
mod maximum_likelihood;
mod pca;
mod registration;
mod weighted_average;
