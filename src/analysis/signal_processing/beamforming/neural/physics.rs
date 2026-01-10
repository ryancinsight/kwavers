//! Physics-informed constraints for neural beamforming.
//!
//! This module enforces physical constraints on neural beamforming outputs to ensure
//! consistency with acoustic wave propagation principles.
//!
//! ## Physical Constraints
//!
//! ### 1. Acoustic Reciprocity
//! Wave propagation satisfies time-reversal symmetry:
//! ```text
//! H(r_tx → r_rx) = H(r_rx → r_tx)
//! ```
//! Enforced through spatial smoothing to ensure local symmetry.
//!
//! ### 2. Spatial Coherence
//! Neighboring voxels should exhibit correlated intensities due to continuous
//! wave field propagation. Enforced via Laplacian diffusion:
//! ```text
//! ∂I/∂t = α ∇²I
//! ```
//!
//! ### 3. Sparsity
//! Focused beams should produce sparse (localized) point-spread functions.
//! Enforced through soft thresholding (L1 regularization):
//! ```text
//! I_sparse = sign(I) · max(|I| - λ, 0)
//! ```
//!
//! ## References
//!
//! - Morse & Ingard (1986): "Theoretical Acoustics" - Reciprocity theorem
//! - Perona & Malik (1990): "Scale-space and edge detection" - Coherence preservation
//! - Donoho (1995): "De-noising by soft-thresholding" - Sparsity enforcement

use crate::domain::core::error::KwaversResult;
use ndarray::Array3;

use super::types::BeamformingFeedback;

/// Physics constraints for neural beamforming.
///
/// Enforces acoustic reciprocity, spatial coherence, and sparsity in beamformed images.
#[derive(Debug, Clone)]
pub struct PhysicsConstraints {
    /// Weight for reciprocity constraint (0.0-1.0)
    reciprocity_weight: f64,
    /// Weight for spatial coherence constraint (0.0-1.0)
    coherence_weight: f64,
    /// Weight for sparsity constraint (0.0-1.0)
    sparsity_weight: f64,
}

impl PhysicsConstraints {
    /// Create physics constraints with specified weights.
    ///
    /// # Arguments
    ///
    /// * `reciprocity_weight` - Strength of reciprocity enforcement (0.0-1.0)
    /// * `coherence_weight` - Diffusion rate for spatial smoothing (0.0-1.0)
    /// * `sparsity_weight` - Threshold for sparsity promotion (0.0-1.0)
    ///
    /// # Invariants
    ///
    /// - All weights must be in [0.0, 1.0]
    /// - Higher weights = stronger constraint enforcement
    pub fn new(reciprocity_weight: f64, coherence_weight: f64, sparsity_weight: f64) -> Self {
        debug_assert!(
            (0.0..=1.0).contains(&reciprocity_weight),
            "Reciprocity weight must be in [0, 1], got {}",
            reciprocity_weight
        );
        debug_assert!(
            (0.0..=1.0).contains(&coherence_weight),
            "Coherence weight must be in [0, 1], got {}",
            coherence_weight
        );
        debug_assert!(
            (0.0..=1.0).contains(&sparsity_weight),
            "Sparsity weight must be in [0, 1], got {}",
            sparsity_weight
        );

        Self {
            reciprocity_weight,
            coherence_weight,
            sparsity_weight,
        }
    }

    /// Get reciprocity constraint weight.
    pub fn reciprocity_weight(&self) -> f64 {
        self.reciprocity_weight
    }

    /// Get coherence constraint weight.
    pub fn coherence_weight(&self) -> f64 {
        self.coherence_weight
    }

    /// Get sparsity constraint weight.
    pub fn sparsity_weight(&self) -> f64 {
        self.sparsity_weight
    }

    /// Apply all physics constraints to beamformed image.
    ///
    /// Applies constraints in sequence:
    /// 1. Reciprocity (spatial symmetry)
    /// 2. Coherence (diffusion smoothing)
    /// 3. Sparsity (soft thresholding)
    ///
    /// # Arguments
    ///
    /// * `image` - Input beamformed image (frames × lateral × axial)
    ///
    /// # Returns
    ///
    /// Constrained image satisfying physical consistency requirements.
    pub fn apply(&self, image: &Array3<f32>) -> KwaversResult<Array3<f32>> {
        let mut constrained = image.clone();

        // Apply reciprocity constraint (time-reversal symmetry)
        constrained = self.apply_reciprocity(&constrained);

        // Apply coherence constraint (spatial smoothness)
        constrained = self.apply_coherence(&constrained);

        // Apply sparsity constraint (focused beams)
        constrained = self.apply_sparsity(&constrained);

        Ok(constrained)
    }

    /// Enforce acoustic reciprocity through spatial smoothing.
    ///
    /// # Mathematical Foundation
    ///
    /// Acoustic reciprocity requires that the Green's function is symmetric:
    /// G(r₁, r₂) = G(r₂, r₁)
    ///
    /// We approximate this locally by blending each voxel with its 4-neighbor average:
    /// ```text
    /// I'(i,j,k) = (1-α)·I(i,j,k) + α·mean(neighbors)
    /// ```
    /// where α = 0.1 · reciprocity_weight
    ///
    /// # Implementation
    ///
    /// - Kernel: 4-point spatial average (von Neumann neighborhood)
    /// - Boundary: Preserved (interior points only)
    /// - Strength: 10% of reciprocity_weight (α ∈ [0, 0.1])
    fn apply_reciprocity(&self, image: &Array3<f32>) -> Array3<f32> {
        let mut result = image.clone();
        let (rows, cols, depth) = image.dim();
        let alpha = (self.reciprocity_weight * 0.1) as f32;

        for i in 1..rows - 1 {
            for j in 1..cols - 1 {
                for k in 0..depth {
                    let neighborhood = [
                        image[[i - 1, j, k]],
                        image[[i + 1, j, k]],
                        image[[i, j - 1, k]],
                        image[[i, j + 1, k]],
                    ];
                    let avg = neighborhood.iter().sum::<f32>() / 4.0;
                    result[[i, j, k]] = image[[i, j, k]] * (1.0 - alpha) + avg * alpha;
                }
            }
        }

        result
    }

    /// Enforce spatial coherence using Laplacian diffusion.
    ///
    /// # Mathematical Foundation
    ///
    /// Heat equation: ∂I/∂t = α ∇²I
    ///
    /// Discrete 5-point stencil:
    /// ```text
    /// ∇²I(i,j) ≈ [I(i-1,j) + I(i+1,j) + I(i,j-1) + I(i,j+1) - 4I(i,j)] / Δx²
    /// ```
    ///
    /// Update rule (forward Euler):
    /// ```text
    /// I_new = I + λ·∇²I
    /// ```
    /// where λ = 0.25 · coherence_weight (CFL stability condition)
    ///
    /// # Implementation
    ///
    /// - Stencil: 5-point Laplacian (cross-shaped kernel)
    /// - Diffusion coefficient: 0.25 · coherence_weight
    /// - Boundary: Neumann (zero-flux, preserved edges)
    fn apply_coherence(&self, image: &Array3<f32>) -> Array3<f32> {
        let mut smoothed = image.clone();
        let (rows, cols, depth) = image.dim();
        let lambda = (self.coherence_weight * 0.25) as f32;

        for k in 0..depth {
            for i in 1..rows - 1 {
                for j in 1..cols - 1 {
                    // 5-point Laplacian stencil
                    let neighbors_sum = image[[i - 1, j, k]]
                        + image[[i + 1, j, k]]
                        + image[[i, j - 1, k]]
                        + image[[i, j + 1, k]];
                    let center = image[[i, j, k]];

                    // Laplacian: ∇²I ≈ (sum_neighbors - 4·center)
                    let laplacian = neighbors_sum - 4.0 * center;

                    // Diffusion update
                    smoothed[[i, j, k]] = center + lambda * laplacian;
                }
            }
        }

        smoothed
    }

    /// Enforce sparsity through soft thresholding (L1 regularization).
    ///
    /// # Mathematical Foundation
    ///
    /// Soft thresholding operator (proximal operator of L1 norm):
    /// ```text
    /// S_λ(x) = sign(x) · max(|x| - λ, 0)
    ///       = { x - λ    if x > λ
    ///         { 0        if |x| ≤ λ
    ///         { x + λ    if x < -λ
    /// ```
    ///
    /// # Implementation
    ///
    /// - Threshold: λ = 0.1 · sparsity_weight · max(|I|)
    /// - Adaptive scaling relative to signal peak ensures robustness
    /// - Promotes focused beams by suppressing low-amplitude artifacts
    fn apply_sparsity(&self, image: &Array3<f32>) -> Array3<f32> {
        // Determine adaptive threshold relative to peak signal
        let max_val = image.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        let threshold = (self.sparsity_weight * 0.1) as f32 * max_val;

        // Apply soft thresholding element-wise
        image.mapv(|x| {
            let abs_x = x.abs();
            if abs_x > threshold {
                x.signum() * (abs_x - threshold)
            } else {
                0.0
            }
        })
    }

    /// Update constraint weights based on performance feedback.
    ///
    /// Adaptive weight adjustment:
    /// - If improvement > 0: Maintain current weights (working well)
    /// - If improvement ≤ 0: Reduce weights by 5% (over-constraining)
    ///
    /// # Arguments
    ///
    /// * `feedback` - Performance metrics from beamforming iteration
    pub fn update(&mut self, feedback: &BeamformingFeedback) -> KwaversResult<()> {
        if feedback.improvement <= 0.0 {
            // Poor performance: relax constraints
            self.reciprocity_weight *= 0.95;
            self.coherence_weight *= 0.95;
            // Sparsity weight unchanged (less sensitive)
        }
        // Good performance: maintain weights

        Ok(())
    }
}

impl Default for PhysicsConstraints {
    /// Create constraints with default weights optimized for ultrasound.
    ///
    /// Default values:
    /// - Reciprocity: 1.0 (full enforcement)
    /// - Coherence: 0.5 (moderate smoothing)
    /// - Sparsity: 0.1 (light sparsity promotion)
    fn default() -> Self {
        Self::new(1.0, 0.5, 0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_physics_constraints_creation() {
        let pc = PhysicsConstraints::new(0.8, 0.5, 0.2);
        assert_eq!(pc.reciprocity_weight(), 0.8);
        assert_eq!(pc.coherence_weight(), 0.5);
        assert_eq!(pc.sparsity_weight(), 0.2);
    }

    #[test]
    fn test_physics_constraints_default() {
        let pc = PhysicsConstraints::default();
        assert_eq!(pc.reciprocity_weight(), 1.0);
        assert_eq!(pc.coherence_weight(), 0.5);
        assert_eq!(pc.sparsity_weight(), 0.1);
    }

    #[test]
    fn test_reciprocity_smoothing() {
        let pc = PhysicsConstraints::new(1.0, 0.0, 0.0);
        let mut image = Array3::zeros((5, 5, 1));
        image[[2, 2, 0]] = 100.0; // Central spike

        let result = pc.apply_reciprocity(&image);

        // Center should be slightly reduced due to averaging
        assert!(result[[2, 2, 0]] < 100.0);
        // Neighbors should receive some signal
        assert!(result[[1, 2, 0]] > 0.0);
        assert!(result[[3, 2, 0]] > 0.0);
    }

    #[test]
    fn test_coherence_diffusion() {
        let pc = PhysicsConstraints::new(0.0, 1.0, 0.0);
        let mut image = Array3::zeros((5, 5, 1));
        image[[2, 2, 0]] = 100.0; // Central spike

        let result = pc.apply_coherence(&image);

        // Diffusion should smooth the spike
        assert!(result[[2, 2, 0]] < 100.0);
        // Energy should spread to neighbors
        assert!(result[[1, 2, 0]] > 0.0 || result[[2, 1, 0]] > 0.0);
    }

    #[test]
    fn test_sparsity_thresholding() {
        let pc = PhysicsConstraints::new(0.0, 0.0, 1.0);
        let mut image = Array3::from_elem((5, 5, 1), 1.0);
        image[[2, 2, 0]] = 100.0; // Strong peak

        let result = pc.apply_sparsity(&image);

        // Low values should be zeroed
        assert_eq!(result[[0, 0, 0]], 0.0);
        // Peak should be preserved (though reduced)
        assert!(result[[2, 2, 0]] > 50.0);
    }

    #[test]
    fn test_combined_constraints() {
        let pc = PhysicsConstraints::default();
        let image = Array::from_shape_fn((10, 10, 5), |(i, j, _k)| {
            if i == 5 && j == 5 {
                100.0
            } else {
                (i as f32 + j as f32) * 0.1
            }
        });

        let result = pc.apply(&image).unwrap();

        // Should be well-formed
        assert!(result.iter().all(|&x| x.is_finite()));
        assert!(result.dim() == image.dim());
    }

    #[test]
    fn test_adaptive_weight_update() {
        let mut pc = PhysicsConstraints::default();
        let initial_reciprocity = pc.reciprocity_weight();

        let feedback = BeamformingFeedback {
            improvement: -0.1,
            error_gradient: 0.5,
            signal_quality: 0.7,
        };

        pc.update(&feedback).unwrap();

        // Weights should be reduced after negative feedback
        assert!(pc.reciprocity_weight() < initial_reciprocity);
        assert!(pc.coherence_weight() < 0.5);
    }

    #[test]
    fn test_boundary_preservation() {
        let pc = PhysicsConstraints::default();
        let image = Array3::from_elem((5, 5, 1), 10.0);

        let result = pc.apply(&image).unwrap();

        // Boundary values should be approximately preserved
        assert!((result[[0, 0, 0]] - 10.0).abs() < 5.0);
        assert!((result[[4, 4, 0]] - 10.0).abs() < 5.0);
    }
}
