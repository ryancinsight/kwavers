//! Full ULM detection pipeline: SVD filtering + Gaussian localization.

use super::clutter::UlmSvdClutterFilter;
use super::localize::GaussianLocalizer;
use super::types::{BubbleDetection, GaussianLocalizationConfig, SvdClutterConfig};
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::Array2;

/// Full ULM detection pipeline: SVD filtering + Gaussian localization.
#[derive(Debug)]
pub struct UlmDetector {
    clutter_filter: UlmSvdClutterFilter,
    localizer: GaussianLocalizer,
}

impl UlmDetector {
    #[must_use]
    pub fn new(clutter_cfg: SvdClutterConfig, loc_cfg: GaussianLocalizationConfig) -> Self {
        Self {
            clutter_filter: UlmSvdClutterFilter::new(clutter_cfg),
            localizer: GaussianLocalizer::new(loc_cfg),
        }
    }

    /// Process a block of IQ frames and return all bubble detections.
    ///
    /// # Arguments
    /// * `iq_block` — IQ matrix \[N_px × N_t\] (pixels × frames, linearized 2D→1D)
    /// * `n_z` — number of axial pixels (N_px = n_z × n_x)
    /// * `n_x` — number of lateral pixels
    /// # Errors
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn process_block(
        &self,
        iq_block: &Array2<f64>,
        n_z: usize,
        n_x: usize,
    ) -> KwaversResult<Vec<BubbleDetection>> {
        let (bubble_data, _k) = self.clutter_filter.filter(iq_block)?;
        let n_t = bubble_data.shape()[1];
        let mut all_detections = Vec::new();

        for t in 0..n_t {
            let frame_col = bubble_data
                .index_axis::<1>(1, t)
                .expect("invariant: column index within bounds");
            if frame_col.shape()[0] != n_z * n_x {
                return Err(KwaversError::Numerical(NumericalError::SolverFailed {
                    method: "ULM detect".to_owned(),
                    reason: format!(
                        "pixel count {} ≠ n_z×n_x = {}×{}={}",
                        frame_col.shape()[0],
                        n_z,
                        n_x,
                        n_z * n_x
                    ),
                }));
            }
            let envelope =
                Array2::from_shape_vec((n_z, n_x), frame_col.iter().map(|v| v.abs()).collect())
                    .map_err(|e| {
                        KwaversError::Numerical(NumericalError::SolverFailed {
                            method: "ULM reshape".to_owned(),
                            reason: e.to_string(),
                        })
                    })?;

            let frame_dets = self.localizer.localize_frame(&envelope, t)?;
            all_detections.extend(frame_dets);
        }

        Ok(all_detections)
    }
}