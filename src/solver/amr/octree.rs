// src/solver/amr/octree.rs
//! Octree data structure for 3D adaptive mesh refinement
//! 
//! Implements an efficient octree for spatial hierarchy with:
//! - O(log n) insertion and lookup
//! - Memory-efficient storage
//! - Fast parent-child navigation

use crate::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

/// Octree node representing a spatial region
#[derive(Debug, Clone)]
struct OctreeNode {
    /// Node index in the octree
    index: usize,
    /// Refinement level (0 = root)
    level: usize,
    /// Spatial bounds (i, j, k) start
    bounds_min: (usize, usize, usize),
    /// Spatial bounds (i, j, k) end (exclusive)
    bounds_max: (usize, usize, usize),
    /// Parent node index (None for root)
    parent: Option<usize>,
    /// Child node indices (None if leaf)
    children: Option<[usize; 8]>,
    /// Whether this node is active (leaf)
    is_active: bool,
}

/// Octree for adaptive mesh refinement
#[derive(Debug)]
pub struct Octree {
    /// All nodes in the octree
    nodes: Vec<OctreeNode>,
    /// Mapping from spatial coordinates to node index
    coord_to_node: HashMap<(usize, usize, usize), usize>,
    /// Base grid dimensions
    base_dims: (usize, usize, usize),
    /// Maximum refinement level
    max_level: usize,
    /// Next available node index
    next_index: usize,
}

impl Octree {
    /// Create a new octree with given base dimensions
    pub fn new(nx: usize, ny: usize, nz: usize, max_level: usize) -> Self {
        let mut octree = Self {
            nodes: Vec::new(),
            coord_to_node: HashMap::new(),
            base_dims: (nx, ny, nz),
            max_level,
            next_index: 0,
        };
        
        // Initialize with root node
        octree.create_root();
        
        octree
    }
    
    /// Create the root node covering entire domain
    fn create_root(&mut self) {
        let root = OctreeNode {
            index: 0,
            level: 0,
            bounds_min: (0, 0, 0),
            bounds_max: self.base_dims,
            parent: None,
            children: None,
            is_active: true,
        };
        
        self.nodes.push(root);
        self.coord_to_node.insert((0, 0, 0), 0);
        self.next_index = 1;
    }
    
    /// Check if cell coordinates are valid
    pub fn is_valid_cell(&self, i: usize, j: usize, k: usize) -> bool {
        i < self.base_dims.0 && j < self.base_dims.1 && k < self.base_dims.2
    }
    
    /// Get the refinement level of a cell
    pub fn get_level(&self, i: usize, j: usize, k: usize) -> usize {
        if let Some(&node_idx) = self.coord_to_node.get(&(i, j, k)) {
            self.nodes[node_idx].level
        } else {
            0
        }
    }
    
    /// Get parent cell coordinates
    pub fn get_parent(&self, i: usize, j: usize, k: usize) -> Option<(usize, usize, usize)> {
        if let Some(&node_idx) = self.coord_to_node.get(&(i, j, k)) {
            if let Some(parent_idx) = self.nodes[node_idx].parent {
                let parent = &self.nodes[parent_idx];
                Some(parent.bounds_min)
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// Get children cell coordinates
    pub fn get_children(&self, i: usize, j: usize, k: usize) -> Vec<(usize, usize, usize)> {
        if let Some(&node_idx) = self.coord_to_node.get(&(i, j, k)) {
            if node_idx < self.nodes.len() {
                if let Some(children) = self.nodes[node_idx].children {
                    children.iter()
                        .filter_map(|&child_idx| {
                            if child_idx < self.nodes.len() {
                                Some(self.nodes[child_idx].bounds_min)
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }
    
    /// Refine a cell by creating 8 children
    pub fn refine_cell(&mut self, i: usize, j: usize, k: usize) -> KwaversResult<bool> {
        let node_idx = match self.coord_to_node.get(&(i, j, k)) {
            Some(&idx) => idx,
            None => return Ok(false),
        };
        
        // Check if already refined or at max level
        if self.nodes[node_idx].children.is_some() || self.nodes[node_idx].level >= self.max_level {
            return Ok(false);
        }
        
        // Create 8 children
        let parent = &self.nodes[node_idx];
        let level = parent.level + 1;
        let (min_i, min_j, min_k) = parent.bounds_min;
        let (max_i, max_j, max_k) = parent.bounds_max;
        
        let mid_i = (min_i + max_i) / 2;
        let mid_j = (min_j + max_j) / 2;
        let mid_k = (min_k + max_k) / 2;
        
        let child_bounds = [
            ((min_i, min_j, min_k), (mid_i, mid_j, mid_k)),
            ((mid_i, min_j, min_k), (max_i, mid_j, mid_k)),
            ((min_i, mid_j, min_k), (mid_i, max_j, mid_k)),
            ((mid_i, mid_j, min_k), (max_i, max_j, mid_k)),
            ((min_i, min_j, mid_k), (mid_i, mid_j, max_k)),
            ((mid_i, min_j, mid_k), (max_i, mid_j, max_k)),
            ((min_i, mid_j, mid_k), (mid_i, max_j, max_k)),
            ((mid_i, mid_j, mid_k), (max_i, max_j, max_k)),
        ];
        
        let mut child_indices = [0; 8];
        
        for (child_num, &(bounds_min, bounds_max)) in child_bounds.iter().enumerate() {
            let child = OctreeNode {
                index: self.next_index,
                level,
                bounds_min,
                bounds_max,
                parent: Some(node_idx),
                children: None,
                is_active: true,
            };
            
            child_indices[child_num] = self.next_index;
            self.coord_to_node.insert(bounds_min, self.next_index);
            self.nodes.push(child);
            self.next_index += 1;
        }
        
        // Update parent
        self.nodes[node_idx].children = Some(child_indices);
        self.nodes[node_idx].is_active = false;
        
        Ok(true)
    }
    
    /// Coarsen a cell by removing its children
    pub fn coarsen_cell(&mut self, i: usize, j: usize, k: usize) -> KwaversResult<bool> {
        let node_idx = match self.coord_to_node.get(&(i, j, k)) {
            Some(&idx) => idx,
            None => return Ok(false),
        };
        
        // Check if has children
        let children = match self.nodes[node_idx].children {
            Some(children) => children,
            None => return Ok(false),
        };
        
        // Remove children from coord mapping
        for &child_idx in &children {
            let child_bounds = self.nodes[child_idx].bounds_min;
            self.coord_to_node.remove(&child_bounds);
        }
        
        // Mark parent as leaf
        self.nodes[node_idx].children = None;
        self.nodes[node_idx].is_active = true;
        
        // Note: We don't actually remove nodes from the vector to avoid index invalidation
        // In a production implementation, we might use a free list or periodic compaction
        
        Ok(true)
    }
    
    /// Get total number of cells in octree
    pub fn total_cells(&self) -> usize {
        self.nodes.len()
    }
    
    /// Get number of base (uniform grid) cells
    pub fn base_cells(&self) -> usize {
        let (nx, ny, nz) = self.base_dims;
        nx * ny * nz
    }
    
    /// Find the leaf node containing a given point
    pub fn find_leaf_node(&self, i: usize, j: usize, k: usize) -> Option<&OctreeNode> {
        // Start from root
        let mut current = &self.nodes[0];
        
        // Traverse down until we find a leaf
        while let Some(children) = current.children {
            // Find which child contains the point
            let child_idx = self.get_child_index(current, i, j, k);
            current = &self.nodes[children[child_idx]];
        }
        
        Some(current)
    }
    
    /// Determine which child octant contains a point
    fn get_child_index(&self, node: &OctreeNode, i: usize, j: usize, k: usize) -> usize {
        let (min_i, min_j, min_k) = node.bounds_min;
        let (max_i, max_j, max_k) = node.bounds_max;
        
        let mid_i = (min_i + max_i) / 2;
        let mid_j = (min_j + max_j) / 2;
        let mid_k = (min_k + max_k) / 2;
        
        let mut index = 0;
        if i >= mid_i { index |= 1; }
        if j >= mid_j { index |= 2; }
        if k >= mid_k { index |= 4; }
        
        index
    }
    
    /// Get all active (leaf) nodes
    pub fn get_active_nodes(&self) -> Vec<&OctreeNode> {
        self.nodes.iter()
            .filter(|node| node.is_active)
            .collect()
    }
    
    /// Get refinement statistics
    pub fn get_stats(&self) -> OctreeStats {
        let mut level_counts = vec![0; self.max_level + 1];
        let mut active_counts = vec![0; self.max_level + 1];
        
        for node in &self.nodes {
            level_counts[node.level] += 1;
            if node.is_active {
                active_counts[node.level] += 1;
            }
        }
        
        let max_level_used = level_counts.iter().rposition(|&c| c > 0).unwrap_or(0);
        
        OctreeStats {
            total_nodes: self.nodes.len(),
            active_nodes: self.nodes.iter().filter(|n| n.is_active).count(),
            level_counts,
            active_counts,
            max_level_used,
        }
    }
}

/// Statistics about the octree structure
#[derive(Debug, Clone)]
pub struct OctreeStats {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of active (leaf) nodes
    pub active_nodes: usize,
    /// Number of nodes at each level
    pub level_counts: Vec<usize>,
    /// Number of active nodes at each level
    pub active_counts: Vec<usize>,
    /// Maximum level actually used
    pub max_level_used: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_octree_creation() {
        let octree = Octree::new(64, 64, 64, 5);
        assert_eq!(octree.total_cells(), 1);
        assert_eq!(octree.base_cells(), 64 * 64 * 64);
    }
    
    #[test]
    fn test_cell_refinement() {
        let mut octree = Octree::new(8, 8, 8, 3);
        
        // Refine root cell
        assert!(octree.refine_cell(0, 0, 0).unwrap());
        assert_eq!(octree.total_cells(), 9); // 1 root + 8 children
        
        // Check children exist
        let children = octree.get_children(0, 0, 0);
        assert_eq!(children.len(), 8);
    }
    
    #[test]
    fn test_cell_coarsening() {
        let mut octree = Octree::new(8, 8, 8, 3);
        
        // Refine then coarsen
        octree.refine_cell(0, 0, 0).unwrap();
        assert!(octree.coarsen_cell(0, 0, 0).unwrap());
        
        // Check children removed
        let children = octree.get_children(0, 0, 0);
        assert_eq!(children.len(), 0);
    }
    
    #[test]
    fn test_max_level_limit() {
        let mut octree = Octree::new(64, 64, 64, 2);
        
        // Refine to max level
        octree.refine_cell(0, 0, 0).unwrap();
        octree.refine_cell(0, 0, 0).unwrap(); // Now at level 1
        
        // Get a child at level 1
        let children = octree.get_children(0, 0, 0);
        let (ci, cj, ck) = children[0];
        
        // Refine child to level 2
        octree.refine_cell(ci, cj, ck).unwrap();
        
        // Try to refine beyond max level
        let grandchildren = octree.get_children(ci, cj, ck);
        let (gci, gcj, gck) = grandchildren[0];
        assert!(!octree.refine_cell(gci, gcj, gck).unwrap());
    }
}