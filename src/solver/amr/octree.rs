//! Octree data structure for 3D adaptive mesh refinement

use crate::error::KwaversResult;
use crate::grid::Bounds;
use ndarray::Array3;

/// Octree node for spatial subdivision
#[derive(Debug, Clone)]
pub struct OctreeNode {
    /// Node bounds
    pub bounds: Bounds,
    /// Refinement level (0 = root)
    pub level: usize,
    /// Node data
    pub data: NodeData,
    /// Child nodes (8 for octree)
    pub children: Option<Box<[OctreeNode; 8]>>,
}

/// Node data stored at each octree node
#[derive(Debug, Clone)]
pub enum NodeData {
    /// Leaf node with field values
    Leaf(Vec<f64>),
    /// Internal node (no data)
    Internal,
}

impl OctreeNode {
    /// Create a new octree node
    #[must_use]
    pub fn new(bounds: Bounds, level: usize) -> Self {
        Self {
            bounds,
            level,
            data: NodeData::Leaf(Vec::new()),
            children: None,
        }
    }

    /// Check if node is a leaf
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// Refine this node by creating children
    pub fn refine(&mut self) -> KwaversResult<()> {
        if self.children.is_some() {
            return Ok(()); // Already refined
        }

        let children = self.create_children()?;
        self.children = Some(Box::new(children));
        self.data = NodeData::Internal;

        Ok(())
    }

    /// Create child nodes
    fn create_children(&self) -> KwaversResult<[OctreeNode; 8]> {
        let mid = self.bounds.center();
        let min = self.bounds.min;
        let max = self.bounds.max;

        Ok([
            // Bottom layer (z = min)
            OctreeNode::new(
                Bounds::new([min[0], min[1], min[2]], [mid[0], mid[1], mid[2]]),
                self.level + 1,
            ),
            OctreeNode::new(
                Bounds::new([mid[0], min[1], min[2]], [max[0], mid[1], mid[2]]),
                self.level + 1,
            ),
            OctreeNode::new(
                Bounds::new([min[0], mid[1], min[2]], [mid[0], max[1], mid[2]]),
                self.level + 1,
            ),
            OctreeNode::new(
                Bounds::new([mid[0], mid[1], min[2]], [max[0], max[1], mid[2]]),
                self.level + 1,
            ),
            // Top layer (z = mid)
            OctreeNode::new(
                Bounds::new([min[0], min[1], mid[2]], [mid[0], mid[1], max[2]]),
                self.level + 1,
            ),
            OctreeNode::new(
                Bounds::new([mid[0], min[1], mid[2]], [max[0], mid[1], max[2]]),
                self.level + 1,
            ),
            OctreeNode::new(
                Bounds::new([min[0], mid[1], mid[2]], [mid[0], max[1], max[2]]),
                self.level + 1,
            ),
            OctreeNode::new(
                Bounds::new([mid[0], mid[1], mid[2]], [max[0], max[1], max[2]]),
                self.level + 1,
            ),
        ])
    }

    /// Coarsen by removing children
    pub fn coarsen(&mut self) {
        self.children = None;
        self.data = NodeData::Leaf(Vec::new());
    }
}

/// Octree structure for adaptive mesh refinement
#[derive(Debug)]
pub struct Octree {
    /// Root node
    root: OctreeNode,
    /// Maximum refinement level
    max_level: usize,
}

impl Octree {
    /// Create a new octree
    pub fn new(bounds: Bounds, max_level: usize) -> KwaversResult<Self> {
        Ok(Self {
            root: OctreeNode::new(bounds, 0),
            max_level,
        })
    }

    /// Get the bounds of the octree
    #[must_use]
    pub fn bounds(&self) -> &Bounds {
        &self.root.bounds
    }

    /// Update refinement based on markers
    pub fn update_refinement(&mut self, markers: &Array3<i8>) -> KwaversResult<()> {
        // Traverse tree and refine/coarsen based on markers
        // 1 = refine, -1 = coarsen, 0 = no change
        Self::update_node(&mut self.root, markers, self.max_level)
    }

    fn update_node(
        node: &mut OctreeNode,
        markers: &Array3<i8>,
        max_level: usize,
    ) -> KwaversResult<()> {
        // Check if this node should be refined
        let should_refine = Self::check_refinement_marker(node, markers);

        if should_refine && node.level < max_level {
            node.refine()?;
        } else if !should_refine && !node.is_leaf() {
            // Check if we can coarsen
            let can_coarsen = Self::check_coarsening(node, markers);
            if can_coarsen {
                node.coarsen();
            }
        }

        // Recursively update children
        if let Some(ref mut children) = node.children {
            for child in children.iter_mut() {
                Self::update_node(child, markers, max_level)?;
            }
        }

        Ok(())
    }

    fn check_refinement_marker(node: &OctreeNode, markers: &Array3<i8>) -> bool {
        // Check if any cell in this node's region is marked for refinement
        // This is a simplified check - real implementation would map bounds to indices
        false // Placeholder
    }

    fn check_coarsening(node: &OctreeNode, markers: &Array3<i8>) -> bool {
        // Check if all children can be coarsened
        true // Placeholder
    }

    /// Count total nodes
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.count_nodes(&self.root)
    }

    fn count_nodes(&self, node: &OctreeNode) -> usize {
        let mut count = 1;
        if let Some(ref children) = node.children {
            for child in children.iter() {
                count += self.count_nodes(child);
            }
        }
        count
    }

    /// Count leaf nodes
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.count_leaves(&self.root)
    }

    fn count_leaves(&self, node: &OctreeNode) -> usize {
        if node.is_leaf() {
            1
        } else if let Some(ref children) = node.children {
            children.iter().map(|c| self.count_leaves(c)).sum()
        } else {
            0
        }
    }

    /// Estimate memory usage
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.node_count() * std::mem::size_of::<OctreeNode>()
    }
}
