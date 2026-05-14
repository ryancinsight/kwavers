//! Same-aperture finite-frequency inverse kernels.
//!
//! This module owns reusable algebra for ultrasound transmit/receive inverse
//! models where the therapeutic aperture also supplies monitoring channels.
//! Clinical modules provide anatomy, CT/NIfTI preparation, and device layout;
//! this module owns active-support indexing, finite-frequency row operators,
//! and graph-H1 regularized PCG.

mod active_grid;
mod encoded;
mod finite_frequency;
mod graph_pcg;
mod linear_operator;
mod operator;
mod row_matrix;

pub use active_grid::{active_grid, image_from_vector, vector_from_image, ActiveGrid, PlanarPoint};
pub use encoded::{encode_measurements, EncodedOperator, EncodingSpec};
pub use finite_frequency::{
    build_fundamental_matrix, build_harmonic_matrix, build_passive_matrix,
    build_ultraharmonic_matrix, fundamental_operator, harmonic_operator, passive_operator,
    ultraharmonic_operator, SameApertureMedium, SameApertureSettings, C_REF_M_S,
};
pub use graph_pcg::{solve_tikhonov_h1, PcgResult, PcgSettings, SAME_APERTURE_OPERATOR_MODEL};
pub use linear_operator::LinearOperator;
pub use operator::FiniteFrequencyOperator;
pub use row_matrix::RowMatrix;

#[cfg(test)]
mod tests;
