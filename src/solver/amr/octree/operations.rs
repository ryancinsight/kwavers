// octree/operations.rs - Octree operations (refine, coarsen, balance)

use super::{Octree, OctreeNode};
use crate::error::{KwaversError, KwaversResult};

/// Operations on octree structure
pub struct OctreeOperations;

impl OctreeOperations {
    /// Refine a node into 8 children
    pub fn refine_node(octree: &mut Octree, node_index: usize) -> KwaversResult<()> {
        let node = octree.node(node_index).ok_or_else(|| {
            KwaversError::InvalidParameter(format!("Node {} not found", node_index))
        })?;

        if node.is_refined() {
            return Ok(()); // Already refined
        }

        if node.level() >= octree.max_level() as i32 {
            return Err(KwaversError::InvalidParameter(format!(
                "Cannot refine beyond max level {}",
                octree.max_level()
            )));
        }

        let (i_min, j_min, k_min) = node.bounds_min();
        let (i_max, j_max, k_max) = node.bounds_max();
        let level = node.level();

        // Compute midpoints
        let i_mid = (i_min + i_max) / 2;
        let j_mid = (j_min + j_max) / 2;
        let k_mid = (k_min + k_max) / 2;

        // Create 8 children
        let children_bounds = [
            ((i_min, j_min, k_min), (i_mid, j_mid, k_mid)), // 000
            ((i_mid, j_min, k_min), (i_max, j_mid, k_mid)), // 100
            ((i_min, j_mid, k_min), (i_mid, j_max, k_mid)), // 010
            ((i_mid, j_mid, k_min), (i_max, j_max, k_mid)), // 110
            ((i_min, j_min, k_mid), (i_mid, j_mid, k_max)), // 001
            ((i_mid, j_min, k_mid), (i_max, j_mid, k_max)), // 101
            ((i_min, j_mid, k_mid), (i_mid, j_max, k_max)), // 011
            ((i_mid, j_mid, k_mid), (i_max, j_max, k_max)), // 111
        ];

        let mut child_indices = [0; 8];

        for (idx, (bounds_min, bounds_max)) in children_bounds.iter().enumerate() {
            let child = OctreeNode::child(node_index, level, *bounds_min, *bounds_max);
            child_indices[idx] = octree.add_node(child);
        }

        // Update parent node
        if let Some(node) = octree.node_mut(node_index) {
            node.set_children(child_indices);
        }

        Ok(())
    }

    /// Coarsen by removing children
    pub fn coarsen_node(octree: &mut Octree, node_index: usize) -> KwaversResult<()> {
        let node = octree.node_mut(node_index).ok_or_else(|| {
            KwaversError::InvalidParameter(format!("Node {} not found", node_index))
        })?;

        if !node.is_refined() {
            return Ok(()); // Not refined
        }

        // Clear children
        node.clear_children();

        Ok(())
    }

    /// Balance octree to ensure 2:1 level ratio
    pub fn balance(octree: &mut Octree) -> KwaversResult<()> {
        let mut changed = true;
        const MAX_ITERATIONS: usize = 100;
        let mut iteration = 0;

        while changed && iteration < MAX_ITERATIONS {
            changed = false;
            iteration += 1;

            // Check each node for balance violations
            let node_count = octree.node_count();
            for i in 0..node_count {
                if Self::needs_refinement_for_balance(octree, i)? {
                    Self::refine_node(octree, i)?;
                    changed = true;
                }
            }
        }

        if iteration >= MAX_ITERATIONS {
            return Err(KwaversError::NumericalError(
                "Octree balancing did not converge".to_string(),
            ));
        }

        Ok(())
    }

    /// Check if node needs refinement for 2:1 balance
    fn needs_refinement_for_balance(octree: &Octree, node_index: usize) -> KwaversResult<bool> {
        let node = octree.node(node_index).ok_or_else(|| {
            KwaversError::InvalidParameter(format!("Node {} not found", node_index))
        })?;

        if node.is_refined() {
            return Ok(false);
        }

        let node_level = node.level();

        // Check all neighbors
        for neighbor_idx in Self::find_neighbors(octree, node_index) {
            if let Some(neighbor) = octree.node(neighbor_idx) {
                if neighbor.level() > node_level + 1 {
                    return Ok(true); // Violates 2:1 ratio
                }
            }
        }

        Ok(false)
    }

    /// Find neighbor indices
    fn find_neighbors(_octree: &Octree, _node_index: usize) -> Vec<usize> {
        // Simplified - would implement full neighbor search
        Vec::new()
    }
}
