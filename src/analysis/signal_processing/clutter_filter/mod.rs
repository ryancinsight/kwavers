//! Clutter Filtering for Ultrasound Doppler Imaging
//!
//! This module implements clutter filtering techniques to separate blood flow signals
//! from tissue motion in ultrasound Doppler imaging, essential for functional ultrasound
//! and vascular imaging applications.
//!
//! # Overview
//!
//! Clutter filtering removes low-velocity, high-amplitude tissue signals to reveal
//! small, high-velocity blood flow signals. This is critical for Power Doppler and
//! functional ultrasound imaging where blood flow changes indicate neural activity.
//!
//! ## Implementation Status
//!
//! ✅ **Spatiotemporal SVD Clutter Filter** - IMPLEMENTED
//! - Singular Value Decomposition on slow-time data matrix
//! - Tissue/blood subspace separation via singular value thresholding
//! - Manual and automatic rank selection
//! - Temporal filtering across ensemble blocks
//! - Implementation: `svd_filter.rs` (444 lines)
//! - References: Demené et al. (2015), Baranger et al. (2018)
//!
//! ✅ **Polynomial Regression Filter** - IMPLEMENTED
//! - Polynomial fit to slow-time data (configurable order 2-6)
//! - Subtraction of fitted polynomial from signal
//! - Per-pixel polynomial regression
//! - Implementation: `polynomial_filter.rs` (390 lines)
//! - References: Bjaerum et al. (2002)
//!
//! ✅ **IIR High-Pass Filter** - IMPLEMENTED
//! - Butterworth high-pass filter design
//! - Configurable cutoff frequency (typically 50-100 Hz)
//! - Zero-phase filtering (forward-backward)
//! - Implementation: `iir_filter.rs` (411 lines)
//! - References: Jensen (1996)
//!
//! ✅ **Adaptive Clutter Rejection** - IMPLEMENTED
//! - Eigenfilter for adaptive clutter rejection
//! - Clutter-to-blood ratio (CBR) estimation
//! - Multiple subspace separation methods
//! - Implementation: `adaptive_filter.rs` (571 lines)
//! - References: Ledoux et al. (1997)
//!
//! # Clutter Filtering Methods
//!
//! ## 1. Spatiotemporal SVD Filter (Recommended for fUS)
//!
//! **Principle:**
//! - Stack slow-time signals into matrix S (space × time)
//! - Decompose: S = UΣV^T
//! - Tissue clutter: Low-rank, high singular values (σ₁, σ₂, ..., σₖ)
//! - Blood flow: Remaining subspace (σₖ₊₁, ..., σₙ)
//!
//! **Algorithm:**
//! ```text
//! 1. Form slow-time matrix: S[i,t] = RF signal at pixel i, time t
//! 2. Compute SVD: [U, Σ, V] = svd(S)
//! 3. Threshold singular values: Keep only σₖ₊₁ to σₙ
//! 4. Reconstruct: S_filtered = U[:,k+1:] * Σ[k+1:,k+1:] * V[:,k+1:]^T
//! 5. Compute Power Doppler: PD[i] = Σₜ |S_filtered[i,t]|²
//! ```
//!
//! **Advantages:**
//! - Superior clutter rejection (~40 dB)
//! - Preserves blood signal integrity
//! - Adaptive to local tissue motion
//!
//! **Disadvantages:**
//! - Computationally expensive (SVD per block)
//! - Requires sufficient ensemble length (>50 frames)
//!
//! ## 2. Polynomial Regression Filter
//!
//! **Principle:**
//! - Fit polynomial to slow-time signal
//! - Subtract fitted curve (clutter estimate)
//!
//! **Algorithm:**
//! ```text
//! For each pixel:
//! 1. s(t) = slow-time signal
//! 2. p(t) = polynomial fit (order 2-6)
//! 3. s_filtered(t) = s(t) - p(t)
//! ```
//!
//! **Advantages:**
//! - Simple and fast
//! - Deterministic (no parameters to tune)
//!
//! **Disadvantages:**
//! - Poor rejection of complex motion (~20 dB)
//! - Can distort blood signal
//!
//! ## 3. IIR High-Pass Filter
//!
//! **Principle:**
//! - Apply high-pass filter to remove low-frequency tissue motion
//! - Cutoff frequency: 50-100 Hz (below blood flow Doppler shift)
//!
//! **Algorithm:**
//! ```text
//! 1. Design Butterworth filter (fc = 100 Hz, order 4)
//! 2. Apply forward-backward for zero phase
//! 3. s_filtered = filtfilt(b, a, s)
//! ```
//!
//! **Advantages:**
//! - Real-time capable
//! - Linear phase (with filtfilt)
//!
//! **Disadvantages:**
//! - Fixed cutoff (not adaptive)
//! - Can remove slow blood flow
//!
//! # Performance Comparison
//!
//! From Demené et al. (2015):
//!
//! | Method | Clutter Rejection | Computation | Blood Preservation |
//! |--------|------------------|-------------|-------------------|
//! | SVD | ~40 dB | Slow | Excellent |
//! | Polynomial | ~20 dB | Fast | Good |
//! | IIR | ~25 dB | Very Fast | Fair |
//!
//! # Implementation Considerations
//!
//! ## Block Size Selection
//!
//! - Too small: Insufficient rank separation, poor filtering
//! - Too large: Reduced temporal resolution, motion artifacts
//! - **Recommended**: 100-200 frames @ 500 Hz = 200-400 ms blocks
//!
//! ## Rank Selection (SVD)
//!
//! **Manual Threshold:**
//! - Retain K=2-5 largest singular values as clutter
//! - Remove these components, keep rest as blood
//!
//! **Automatic Selection:**
//! - Energy threshold: Keep components until 99.9% clutter energy removed
//! - Knee detection: Find elbow in singular value spectrum
//!
//! **Adaptive:**
//! - Different K for different spatial regions
//! - Based on local tissue motion estimation
//!
//! ## Computational Optimization
//!
//! **Randomized SVD:**
//! - For large matrices, use randomized algorithms
//! - Approximation error controlled, much faster
//! - See: Halko et al. (2011) "Finding structure with randomness"
//!
//! **GPU Acceleration:**
//! - SVD highly parallelizable
//! - cuSOLVER for CUDA-based SVD
//! - ~10-100× speedup possible
//!
//! # Literature References
//!
//! **Spatiotemporal SVD:**
//! - Demené, C., et al. (2015). "Spatiotemporal clutter filtering of ultrafast ultrasound data highly increases Doppler and fUltrasound sensitivity."
//!   *Scientific Reports*, 5, 11203. DOI: 10.1038/srep11203
//!
//! - Baranger, J., et al. (2018). "Adaptive spatiotemporal SVD clutter filtering for ultrafast Doppler imaging using similarity of spatial singular vectors."
//!   *IEEE Transactions on Medical Imaging*, 37(7), 1574-1586. DOI: 10.1109/TMI.2018.2789499
//!
//! **Classical Filters:**
//! - Bjaerum, S., Torp, H., & Kristoffersen, K. (2002). "Clutter filter design for ultrasound color flow imaging."
//!   *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control*, 49(2), 204-216.
//!   DOI: 10.1109/58.985705
//!
//! - Ledoux, L. A. F., Brands, P. J., & Hoeks, A. P. G. (1997). "Reduction of the clutter component in Doppler ultrasound signals based on singular value decomposition."
//!   *IEEE Transactions on Biomedical Engineering*, 44(11), 1120-1129. DOI: 10.1109/10.641344
//!
//! **Textbook:**
//! - Jensen, J. A. (1996). "Estimation of Blood Velocities Using Ultrasound: A Signal Processing Approach."
//!   Cambridge University Press. ISBN: 0-521-46484-4
//!
//! # Module Organization
//!
//! All filters are fully implemented and production-ready:
//!
//! - `svd_filter`: Spatiotemporal SVD clutter filtering (444 lines)
//! - `polynomial_filter`: Polynomial regression filtering (390 lines)
//! - `iir_filter`: IIR high-pass filtering (411 lines)
//! - `adaptive_filter`: Adaptive clutter rejection (571 lines)
//!
//! Total: 1,816 lines of production code + comprehensive documentation

// Implemented clutter filters
pub mod adaptive_filter;
pub mod iir_filter;
pub mod polynomial_filter;
pub mod svd_filter;

// Public API re-exports
pub use adaptive_filter::{
    AdaptiveFilter, AdaptiveFilterConfig, CbrEstimationMethod, SubspaceSeparationMethod,
};
pub use iir_filter::{IirFilter, IirFilterConfig};
pub use polynomial_filter::{PolynomialFilter, PolynomialFilterConfig};
pub use svd_filter::{SvdClutterFilter, SvdClutterFilterConfig};
