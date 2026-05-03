//! Feature extraction utilities for neural beamforming.
//!
//! This module provides functions to extract meaningful features from
//! beamformed images or raw RF data for neural network processing.
//!
//! ## Feature Types
//!
//! 1. **Intensity Features**: Raw image values, log-compressed, normalized
//! 2. **Texture Features**: Local standard deviation, entropy
//! 3. **Structural Features**: Gradients, Laplacian, edges
//! 4. **Statistical Features**: Moments, coherence, speckle metrics
//!
//! ## Mathematical Foundation
//!
//! ### Local Standard Deviation (Texture)
//!
//! For a 3×3 local neighborhood:
//! ```text
//! σ(x,y) = √[ (1/9)∑ᵢⱼ I²(x+i, y+j) - μ²(x,y) ]
//! ```
//!
//! ### Gradient Magnitude (Edges)
//!
//! Sobel operator for edge detection:
//! ```text
//! Gₓ = [-1  0  1]      Gᵧ = [-1 -2 -1]
//!      [-2  0  2]           [ 0  0  0]
//!      [-1  0  1]           [ 1  2  1]
//!
//! |∇I| = √(Gₓ² + Gᵧ²)
//! ```
//!
//! ### Laplacian (Structural)
//!
//! Second derivative for detecting rapid intensity changes:
//! ```text
//! ∇²I = [ 0  1  0]     ∇²I(x,y) = 4I(x,y) - I(x±1,y) - I(x,y±1)
//!       [ 1 -4  1]
//!       [ 0  1  0]
//! ```
//!
//! ### Local Entropy (Information)
//!
//! Information content in local neighborhood:
//! ```text
//! H = -∑ᵢ p(i) log₂ p(i)
//! ```
//! where p(i) is the normalized histogram of the local patch.

mod aggregation;
mod local_ops;

pub use aggregation::{concatenate_features, extract_all_features, normalize_features};
pub use local_ops::{
    compute_laplacian, compute_local_entropy, compute_local_std, compute_spatial_gradient,
};
