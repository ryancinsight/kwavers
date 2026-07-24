//! Statistical quality metrics — SSOT: `leto_ops::application::statistics`.
//!
//! Re-exported here as the kwavers vocabulary so higher layers depend on one
//! import path.
pub use leto_ops::application::statistics::{
    nrmse, normalized_rmse, pearson, percentile_range, phase_error_degrees_for_correlation,
    phase_shift_correlation_curve, psnr, rmse, validation_psnr_from_relative_rmse,
};
