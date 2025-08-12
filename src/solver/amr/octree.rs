// src/solver/amr/octree.rs
//! Octree data structure for 3D adaptive mesh refinement
//! 
//! Implements an efficient octree for spatial hierarchy with:
//! - O(log n) insertion and lookup
//! - Memory-efficient storage
//! - Fast parent-child navigation

use crate::error::KwaversResult;
use std::collections::HashMap;

/// Node in the octree structure
#[derive(Debug, Clone)]
pub struct OctreeNode {
    /// Node index in the octree
    index: usize,
    /// Refinement level (0 = root, positive = refined)
    level: i32,
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

impl OctreeNode {
    /// Check if this node is a leaf
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }
    
    /// Get the refinement level
    pub fn level(&self) -> i32 {
        self.level
    }
    
    /// Get iterator over child indices if they exist
    pub fn children_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.children
            .as_ref()
            .map(|c| c.iter().copied())
            .into_iter()
            .flatten()
    }
    
    /// Check if node has children
    pub fn has_children(&self) -> bool {
        self.children.is_some()
    }
    
    /// Get child count
    pub fn child_count(&self) -> usize {
        self.children.as_ref().map_or(0, |c| c.len())
    }
}

impl Default for OctreeNode {
    fn default() -> Self {
        Self {
            index: 0,
            level: 0,
            bounds_min: (0, 0, 0),
            bounds_max: (0, 0, 0),
            parent: None,
            children: None,
            is_active: true,
        }
    }
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
    /// Free list for recycling node indices
    free_nodes: Vec<usize>,
    /// Number of inactive nodes (for tracking memory efficiency)
    inactive_nodes: usize,
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
            free_nodes: Vec::new(),
            inactive_nodes: 0,
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
    }
    
    /// Check if cell coordinates are valid
    pub fn is_valid_cell(&self, i: usize, j: usize, k: usize) -> bool {
        i < self.base_dims.0 && j < self.base_dims.1 && k < self.base_dims.2
    }
    
    /// Get the refinement level of a cell
    pub fn get_level(&self, i: usize, j: usize, k: usize) -> usize {
        if let Some(&node_idx) = self.coord_to_node.get(&(i, j, k)) {
            self.nodes[node_idx].level as usize
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
    
    /// Get child nodes of a given node
    pub fn get_children(&self, node_idx: usize) -> impl Iterator<Item = &OctreeNode> + '_ {
        self.nodes.get(node_idx)
            .and_then(|node| node.children.as_ref())
            .map(|children| children.iter())
            .into_iter()
            .flatten()
            .filter_map(move |&child_idx| self.nodes.get(child_idx))
    }
    
    /// Get mutable child nodes of a given node
    pub fn get_children_mut(&mut self, node_idx: usize) -> Vec<&mut OctreeNode> {
        if let Some(node) = self.nodes.get(node_idx) {
            if let Some(children) = &node.children {
                let child_indices: Vec<usize> = children.to_vec();
                child_indices.into_iter()
                    .filter_map(|idx| {
                        // Safe because indices are unique
                        self.nodes.get_mut(idx)
                    })
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }
    
    /// Iterate over all leaf nodes
    pub fn leaf_nodes(&self) -> impl Iterator<Item = &OctreeNode> + '_ {
        self.nodes.iter()
            .filter(|node| node.is_leaf() && node.is_active)
    }
    
    /// Iterate over all active nodes at a given level
    pub fn nodes_at_level(&self, level: usize) -> impl Iterator<Item = &OctreeNode> + '_ {
        self.nodes.iter()
            .filter(move |node| node.level as usize == level && node.is_active)
    }
    
    /// Count active nodes
    pub fn active_node_count(&self) -> usize {
        self.nodes.iter()
            .filter(|node| node.is_active)
            .count()
    }
    
    /// Get children cell coordinates
    pub fn get_children_coords(&self, i: usize, j: usize, k: usize) -> Vec<(usize, usize, usize)> {
        self.coord_to_node.get(&(i, j, k))
            .and_then(|&node_idx| self.nodes.get(node_idx))
            .and_then(|node| node.children.as_ref())
            .map(|children| {
                children.iter()
                    .filter_map(|&child_idx| {
                        self.nodes.get(child_idx)
                            .map(|child| child.bounds_min)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
    
    /// Refine a cell by creating 8 children
    pub fn refine_cell(&mut self, i: usize, j: usize, k: usize) -> KwaversResult<bool> {
        let node_idx = match self.coord_to_node.get(&(i, j, k)) {
            Some(&idx) => idx,
            None => return Ok(false),
        };
        
        // Check if already refined or at max level
        if self.nodes[node_idx].children.is_some() || self.nodes[node_idx].level >= self.max_level as i32 {
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
            // Determine child index first
            let child_idx = if let Some(free_idx) = self.free_nodes.pop() {
                self.inactive_nodes -= 1;
                free_idx
            } else {
                let idx = self.nodes.len();
                self.nodes.push(OctreeNode::default()); // Placeholder
                idx
            };
            
            // Now create the child with correct index
            let child = OctreeNode {
                index: child_idx,
                level,
                bounds_min,
                bounds_max,
                parent: Some(node_idx),
                children: None,
                is_active: true,
            };
            
            // Update the node
            self.nodes[child_idx] = child;
            self.next_index += 1;
            
            child_indices[child_num] = child_idx;
            self.coord_to_node.insert(bounds_min, child_idx);
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
        
        // Check if this node has children (is a parent)
        let parent_idx = if self.nodes[node_idx].children.is_some() {
            // This is already a parent
            node_idx
        } else {
            // This is a child, find its parent
            // Search for a node that has this node as a child
            let mut found_parent = None;
            for (idx, node) in self.nodes.iter().enumerate() {
                if let Some(children) = node.children {
                    if children.contains(&node_idx) {
                        found_parent = Some(idx);
                        break;
                    }
                }
            }
            match found_parent {
                Some(idx) => idx,
                None => return Ok(false), // No parent found
            }
        };
        
        // Get children array
        let children = match self.nodes[parent_idx].children {
            Some(children) => children,
            None => return Ok(false),
        };
        
        // Check that all children are leaves (cannot coarsen if grandchildren exist)
        for &child_idx in &children {
            if self.nodes[child_idx].children.is_some() {
                return Ok(false); // Cannot coarsen - has grandchildren
            }
        }
        
        // Remove children from coord mapping and mark as free
        for &child_idx in &children {
            // Get all coordinates that map to this child
            let child_node = &self.nodes[child_idx];
            let child_bounds_min = child_node.bounds_min;
            let child_bounds_max = child_node.bounds_max;
            
            // Remove all coordinate mappings within child bounds
            let coords_to_remove: Vec<_> = self.coord_to_node.keys()
                .filter(|&&(ci, cj, ck)| {
                    ci >= child_bounds_min.0 && ci <= child_bounds_max.0 &&
                    cj >= child_bounds_min.1 && cj <= child_bounds_max.1 &&
                    ck >= child_bounds_min.2 && ck <= child_bounds_max.2 &&
                    self.coord_to_node[&(ci, cj, ck)] == child_idx
                })
                .cloned()
                .collect();
            
            for coord in coords_to_remove {
                self.coord_to_node.remove(&coord);
            }
            
            // Mark child as inactive and add to free list
            self.nodes[child_idx].is_active = false;
            self.free_nodes.push(child_idx);
            self.inactive_nodes += 1;
        }
        
        // Mark parent as leaf
        self.nodes[parent_idx].children = None;
        self.nodes[parent_idx].is_active = true;
        
        // Re-add parent coordinates to mapping
        let parent_bounds_min = self.nodes[parent_idx].bounds_min;
        let parent_bounds_max = self.nodes[parent_idx].bounds_max;
        for ci in parent_bounds_min.0..=parent_bounds_max.0 {
            for cj in parent_bounds_min.1..=parent_bounds_max.1 {
                for ck in parent_bounds_min.2..=parent_bounds_max.2 {
                    self.coord_to_node.insert((ci, cj, ck), parent_idx);
                }
            }
        }
        
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
    
    /// Compact the octree to remove inactive nodes
    /// This should be called periodically to prevent memory bloat
    pub fn compact(&mut self) {
        // Only compact if we have significant waste
        if self.inactive_nodes < self.nodes.len() / 4 {
            return;
        }
        
        // Create mapping from old to new indices
        let mut index_mapping: HashMap<usize, usize> = HashMap::new();
        let mut compacted_nodes = Vec::new();
        let mut current_index = 0;
        
        // Copy active nodes and build mapping
        for (old_idx, node) in self.nodes.iter().enumerate() {
            if node.is_active || (node.parent.is_some() && index_mapping.contains_key(&node.parent.unwrap())) {
                let mut compacted_node = node.clone();
                
                // Update parent index
                if let Some(parent_idx) = compacted_node.parent {
                    compacted_node.parent = index_mapping.get(&parent_idx).copied();
                }
                
                // Update index
                compacted_node.index = current_index;
                
                index_mapping.insert(old_idx, current_index);
                compacted_nodes.push(compacted_node);
                current_index += 1;
            }
        }
        
        // Update children indices
        for node in &mut compacted_nodes {
            if let Some(mut children) = node.children {
                for child in &mut children {
                    if let Some(&mapped_idx) = index_mapping.get(child) {
                        *child = mapped_idx;
                    }
                }
                node.children = Some(children);
            }
        }
        
        // Rebuild coord_to_node mapping
        self.coord_to_node.clear();
        for (idx, node) in compacted_nodes.iter().enumerate() {
            self.coord_to_node.insert(node.bounds_min, idx);
        }
        
        // Update state
        self.nodes = compacted_nodes;
        self.next_index = self.nodes.len();
        self.free_nodes.clear();
        self.inactive_nodes = 0;
    }
    
    /// Get memory efficiency ratio
    pub fn memory_efficiency(&self) -> f64 {
        let active_nodes = self.nodes.len() - self.inactive_nodes;
        active_nodes as f64 / self.nodes.len() as f64
    }
    
    /// Get statistics about the octree structure
    pub fn stats(&self) -> OctreeStats {
        let mut level_counts = vec![0; self.max_level + 1];
        let mut active_counts = vec![0; self.max_level + 1];
        
        for node in &self.nodes {
            level_counts[node.level as usize] += 1;
            if node.is_active {
                active_counts[node.level as usize] += 1;
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
    
    /// Get the root node
    pub fn root(&self) -> &OctreeNode {
        &self.nodes[0]
    }
    
    /// Get the base resolution
    pub fn base_resolution(&self) -> (usize, usize, usize) {
        self.base_dims
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
        
        // Store the root node index before refinement
        let root_idx = *octree.coord_to_node.get(&(0, 0, 0)).unwrap();
        
        // Refine root cell
        assert!(octree.refine_cell(0, 0, 0).unwrap());
        assert_eq!(octree.total_cells(), 9); // 1 root + 8 children
        
        // After refinement, the root node should have children
        assert!(octree.nodes[root_idx].children.is_some());
        let children_indices = octree.nodes[root_idx].children.unwrap();
        assert_eq!(children_indices.len(), 8);
        
        // The coordinate (0,0,0) now maps to the first child
        let new_node_at_origin = octree.coord_to_node.get(&(0, 0, 0)).unwrap();
        assert_eq!(*new_node_at_origin, children_indices[0]);
    }
    
    #[test]
    fn test_cell_coarsening() {
        let mut octree = Octree::new(8, 8, 8, 3);
        
        // Refine root cell
        assert!(octree.refine_cell(0, 0, 0).unwrap());
        
        // After refinement, (0,0,0) maps to a child
        // We can coarsen by using any coordinate that maps to one of the children
        // The coarsen_cell function should work on any child coordinate
        assert!(octree.coarsen_cell(0, 0, 0).unwrap());
        
        // After coarsening, (0,0,0) should map back to the parent
        let node_at_origin = octree.coord_to_node.get(&(0, 0, 0)).unwrap();
        
        // Check that the node at (0,0,0) has no children
        assert!(octree.nodes[*node_at_origin].children.is_none());
    }
    
    #[test]
    fn test_max_level_limit() {
        let mut octree = Octree::new(64, 64, 64, 2);
        
        // Refine to max level
        assert!(octree.refine_cell(0, 0, 0).unwrap());
        
        // After first refinement, (0,0,0) maps to a child
        // Find a child coordinate to refine further
        let child_idx = *octree.coord_to_node.get(&(0, 0, 0)).unwrap();
        let child_node = &octree.nodes[child_idx];
        assert_eq!(child_node.level, 1);
        
        // Refine the child (now at level 1)
        assert!(octree.refine_cell(0, 0, 0).unwrap());
        
        // Now (0,0,0) maps to a grandchild at level 2
        let grandchild_idx = *octree.coord_to_node.get(&(0, 0, 0)).unwrap();
        let grandchild_node = &octree.nodes[grandchild_idx];
        assert_eq!(grandchild_node.level, 2);
        
        // Try to refine beyond max level (should fail)
        assert!(!octree.refine_cell(0, 0, 0).unwrap());
    }
}