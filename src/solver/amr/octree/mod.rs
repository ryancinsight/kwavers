// solver/amr/octree/mod.rs - Unified octree implementation for AMR

pub mod builder;
pub mod node;
pub mod operations;
pub mod traversal;

use crate::error::KwaversResult;
use std::collections::HashMap;

// Re-export core types - single implementation
pub use builder::OctreeBuilder;
pub use node::{NodeStatus, OctreeNode};
pub use operations::OctreeOperations;
pub use traversal::{Traversal, TraversalOrder};

/// Octree for adaptive mesh refinement - single source of truth
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
    /// Create new octree with base dimensions
    pub fn new(nx: usize, ny: usize, nz: usize, max_level: usize) -> Self {
        let mut octree = Self {
            nodes: Vec::new(),
            coord_to_node: HashMap::new(),
            base_dims: (nx, ny, nz),
            max_level,
            next_index: 0,
        };

        // Create root node
        let root = OctreeNode::root(nx, ny, nz);
        octree.add_node(root);

        octree
    }

    /// Add node to octree
    fn add_node(&mut self, node: OctreeNode) -> usize {
        let index = self.next_index;
        self.next_index += 1;

        // Update coordinate mapping
        let (i_min, j_min, k_min) = node.bounds_min();
        self.coord_to_node.insert((i_min, j_min, k_min), index);

        self.nodes.push(node);
        index
    }

    /// Get node by index
    pub fn node(&self, index: usize) -> Option<&OctreeNode> {
        self.nodes.get(index)
    }

    /// Get mutable node by index
    pub fn node_mut(&mut self, index: usize) -> Option<&mut OctreeNode> {
        self.nodes.get_mut(index)
    }

    /// Get node at spatial coordinate
    pub fn node_at(&self, i: usize, j: usize, k: usize) -> Option<&OctreeNode> {
        self.coord_to_node
            .get(&(i, j, k))
            .and_then(|&idx| self.node(idx))
    }

    /// Get total node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get base resolution
    pub fn base_resolution(&self) -> (usize, usize, usize) {
        self.base_dims
    }

    /// Get refinement ratio (always 2 for octree)
    pub fn refinement_ratio(&self) -> usize {
        2
    }

    /// Get maximum level
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Get root node
    pub fn root(&self) -> &OctreeNode {
        &self.nodes[0]
    }
}

/// Statistics about the octree structure
#[derive(Debug, Clone)]
pub struct OctreeStats {
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub level_counts: Vec<usize>,
    pub max_level_used: usize,
}

impl Octree {
    /// Compute statistics
    pub fn compute_stats(&self) -> OctreeStats {
        let mut level_counts = vec![0; self.max_level + 1];
        let mut active_nodes = 0;
        let mut max_level_used = 0;

        for node in &self.nodes {
            let level = node.level() as usize;
            level_counts[level] += 1;

            if node.is_leaf() {
                active_nodes += 1;
            }

            max_level_used = max_level_used.max(level);
        }

        OctreeStats {
            total_nodes: self.nodes.len(),
            active_nodes,
            level_counts,
            max_level_used,
        }
    }
}
