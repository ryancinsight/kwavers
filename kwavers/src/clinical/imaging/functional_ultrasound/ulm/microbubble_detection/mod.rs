//! Microbubble Detection and Sub-pixel Localization for ULM
//!
//! # Overview
//!
//! Implements the two-stage ULM detection pipeline:
//! 1. **SVD Clutter Filter** — separates microbubble signal from tissue clutter
//!    using spatiotemporal singular value decomposition.
//! 2. **Gaussian Sub-pixel Localization** — fits each detected peak to a 2D Gaussian
//!    to achieve sub-pixel centroid precision.
//!
//! # References
//!
//! - Demené, C., et al. (2015). *IEEE Trans. Med. Imaging* 34(11):2271–2285.
//! - Errico, C., et al. (2015). *Nature* 527:499–502.
//! - Gavish, M., & Donoho, D. L. (2014). *IEEE Trans. Inf. Theory* 60(8):5040–5053.
//! - Thompson, R. E., et al. (2002). *Biophys. J.* 82(5):2775–2783.

mod clutter;
mod detector;
mod localize;
#[cfg(test)]
mod tests;
mod types;

pub use clutter::SvdClutterFilter;
pub use detector::UlmDetector;
pub use localize::GaussianLocalizer;
pub use types::{BubbleDetection, LocalizationConfig, SvdClutterConfig};
