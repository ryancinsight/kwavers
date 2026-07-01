//! PyO3 bindings for `kwavers_math::statistics` validation metrics (book §19).

mod arrays;
mod correlation;
mod metrics;

pub use correlation::{
    pearson, phase_error_degrees_for_correlation, phase_shift_correlation_curve,
};
pub use metrics::{psnr, rmse, validation_psnr_from_relative_rmse};
