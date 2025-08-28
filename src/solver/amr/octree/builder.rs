// octree/builder.rs - Octree construction utilities

use super::{Octree, OctreeOperations};
use crate::error::KwaversResult;
use ndarray::Array3;

/// Builder for octree construction
pub struct OctreeBuilder {
    nx: usize,
    ny: usize,
    nz: usize,
    max_level: usize,
    refinement_criterion: RefinementCriterion,
}

/// Criterion for adaptive refinement
#[derive(Debug, Clone)]
pub enum RefinementCriterion {
    /// Refine based on error threshold
    ErrorThreshold(f64),
    /// Refine based on gradient threshold
    GradientThreshold(f64),
    /// Refine based on feature detection
    FeatureBased,
}

impl OctreeBuilder {
    /// Create new builder
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            nx,
            ny,
            nz,
            max_level: 5,
            refinement_criterion: RefinementCriterion::ErrorThreshold(0.01),
        }
    }

    /// Set maximum refinement level
    pub fn with_max_level(mut self, max_level: usize) -> Self {
        self.max_level = max_level;
        self
    }

    /// Set refinement criterion
    pub fn with_criterion(mut self, criterion: RefinementCriterion) -> Self {
        self.refinement_criterion = criterion;
        self
    }

    /// Build octree
    pub fn build(self) -> KwaversResult<Octree> {
        let octree = Octree::new(self.nx, self.ny, self.nz, self.max_level);
        Ok(octree)
    }

    /// Build adaptive octree from field
    pub fn build_adaptive(self, field: &Array3<f64>) -> KwaversResult<Octree> {
        let mut octree = Octree::new(self.nx, self.ny, self.nz, self.max_level);

        // Adaptively refine based on field
        self.refine_adaptive(&mut octree, field)?;

        // Balance the octree
        OctreeOperations::balance(&mut octree)?;

        Ok(octree)
    }

    /// Perform adaptive refinement
    fn refine_adaptive(&self, octree: &mut Octree, field: &Array3<f64>) -> KwaversResult<()> {
        let mut nodes_to_refine = vec![0]; // Start with root

        while !nodes_to_refine.is_empty() {
            let node_idx = nodes_to_refine.pop().unwrap();

            if self.should_refine(octree, node_idx, field)? {
                OctreeOperations::refine_node(octree, node_idx)?;

                // Add children to refinement queue
                if let Some(node) = octree.node(node_idx) {
                    if let Some(children) = node.children() {
                        for &child_idx in children {
                            nodes_to_refine.push(child_idx);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if node should be refined
    fn should_refine(
        &self,
        octree: &Octree,
        node_idx: usize,
        field: &Array3<f64>,
    ) -> KwaversResult<bool> {
        let node = octree.node(node_idx).ok_or_else(|| {
            crate::error::KwaversError::InvalidParameter(format!("Node {} not found", node_idx))
        })?;

        // Don't refine beyond max level
        if node.level() >= self.max_level as i32 {
            return Ok(false);
        }

        // Don't refine already refined nodes
        if node.is_refined() {
            return Ok(false);
        }

        // Apply refinement criterion
        match &self.refinement_criterion {
            RefinementCriterion::ErrorThreshold(threshold) => {
                let error = self.compute_error(node, field);
                Ok(error > *threshold)
            }
            RefinementCriterion::GradientThreshold(threshold) => {
                let gradient = self.compute_gradient(node, field);
                Ok(gradient > *threshold)
            }
            RefinementCriterion::FeatureBased => Ok(self.detect_feature(node, field)),
        }
    }

    /// Compute error estimate for node
    fn compute_error(&self, node: &super::OctreeNode, field: &Array3<f64>) -> f64 {
        let (i_min, j_min, k_min) = node.bounds_min();
        let (i_max, j_max, k_max) = node.bounds_max();

        // Compute local variance as error estimate
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0;

        for k in k_min..k_max.min(field.shape()[2]) {
            for j in j_min..j_max.min(field.shape()[1]) {
                for i in i_min..i_max.min(field.shape()[0]) {
                    let val = field[[i, j, k]];
                    sum += val;
                    sum_sq += val * val;
                    count += 1;
                }
            }
        }

        if count > 0 {
            let mean = sum / count as f64;
            let variance = sum_sq / count as f64 - mean * mean;
            variance.sqrt()
        } else {
            0.0
        }
    }

    /// Compute gradient magnitude for node
    fn compute_gradient(&self, node: &super::OctreeNode, field: &Array3<f64>) -> f64 {
        let (i_min, j_min, k_min) = node.bounds_min();
        let (i_max, j_max, k_max) = node.bounds_max();

        let mut max_gradient = 0.0;

        for k in k_min..k_max.min(field.shape()[2]) {
            for j in j_min..j_max.min(field.shape()[1]) {
                for i in i_min..i_max.min(field.shape()[0]) {
                    if i > 0 && i < field.shape()[0] - 1 {
                        let grad_x = (field[[i + 1, j, k]] - field[[i - 1, j, k]]).abs() / 2.0;
                        max_gradient = max_gradient.max(grad_x);
                    }
                    if j > 0 && j < field.shape()[1] - 1 {
                        let grad_y = (field[[i, j + 1, k]] - field[[i, j - 1, k]]).abs() / 2.0;
                        max_gradient = max_gradient.max(grad_y);
                    }
                    if k > 0 && k < field.shape()[2] - 1 {
                        let grad_z = (field[[i, j, k + 1]] - field[[i, j, k - 1]]).abs() / 2.0;
                        max_gradient = max_gradient.max(grad_z);
                    }
                }
            }
        }

        max_gradient
    }

    /// Detect features in node
    fn detect_feature(&self, _node: &super::OctreeNode, _field: &Array3<f64>) -> bool {
        // Simplified feature detection
        false
    }
}
