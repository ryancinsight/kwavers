//! Octree data structure for 3D adaptive mesh refinement

use crate::domain::core::error::KwaversResult;
use crate::domain::grid::Bounds;
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

    /// Get a reference to the root node
    #[must_use]
    pub fn root(&self) -> &OctreeNode {
        &self.root
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

    /// Check if node should be refined based on marker array
    ///
    /// A node is marked for refinement if any cell within its bounds
    /// has a positive marker value (1 = refine)
    ///
    /// References:
    /// - Berger & Oliger (1984): "Adaptive mesh refinement for hyperbolic PDEs"
    /// - Lohner (1987): "Adaptive remeshing for transient problems"
    fn check_refinement_marker(node: &OctreeNode, markers: &Array3<i8>) -> bool {
        let (mx, my, mz) = markers.dim();
        let bounds = &node.bounds;

        // Extract bounds components
        let (x_min, y_min, z_min) = (bounds.min[0], bounds.min[1], bounds.min[2]);
        let (x_max, y_max, z_max) = (bounds.max[0], bounds.max[1], bounds.max[2]);

        // Compute domain extent for normalization
        let domain_x = x_max - x_min;
        let domain_y = y_max - y_min;
        let domain_z = z_max - z_min;

        if domain_x <= 0.0 || domain_y <= 0.0 || domain_z <= 0.0 {
            return false;
        }

        // Map spatial bounds to grid indices
        let i_min = ((x_min / domain_x) * mx as f64).max(0.0) as usize;
        let i_max = ((x_max / domain_x) * mx as f64).min(mx as f64) as usize;
        let j_min = ((y_min / domain_y) * my as f64).max(0.0) as usize;
        let j_max = ((y_max / domain_y) * my as f64).min(my as f64) as usize;
        let k_min = ((z_min / domain_z) * mz as f64).max(0.0) as usize;
        let k_max = ((z_max / domain_z) * mz as f64).min(mz as f64) as usize;

        // Check if any cell in region is marked for refinement (> 0)
        for i in i_min..i_max.min(mx) {
            for j in j_min..j_max.min(my) {
                for k in k_min..k_max.min(mz) {
                    if markers[[i, j, k]] > 0 {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Check if node can be coarsened based on marker array
    ///
    /// A node can be coarsened if all cells within its bounds
    /// have non-positive markers (≤ 0) and all children (if any)
    /// can also be coarsened
    ///
    /// References:
    /// - Berger & Colella (1989): "Local adaptive mesh refinement for shock hydrodynamics"
    fn check_coarsening(node: &OctreeNode, markers: &Array3<i8>) -> bool {
        let (mx, my, mz) = markers.dim();
        let bounds = &node.bounds;

        // Extract bounds components
        let (x_min, y_min, z_min) = (bounds.min[0], bounds.min[1], bounds.min[2]);
        let (x_max, y_max, z_max) = (bounds.max[0], bounds.max[1], bounds.max[2]);

        // Compute domain extent
        let domain_x = x_max - x_min;
        let domain_y = y_max - y_min;
        let domain_z = z_max - z_min;

        if domain_x <= 0.0 || domain_y <= 0.0 || domain_z <= 0.0 {
            return true; // Empty domain can be coarsened
        }

        // Map spatial bounds to grid indices
        let i_min = ((x_min / domain_x) * mx as f64).max(0.0) as usize;
        let i_max = ((x_max / domain_x) * mx as f64).min(mx as f64) as usize;
        let j_min = ((y_min / domain_y) * my as f64).max(0.0) as usize;
        let j_max = ((y_max / domain_y) * my as f64).min(my as f64) as usize;
        let k_min = ((z_min / domain_z) * mz as f64).max(0.0) as usize;
        let k_max = ((z_max / domain_z) * mz as f64).min(mz as f64) as usize;

        // Check if all cells in region are marked for coarsening (< 0) or unchanged (0)
        // A positive marker anywhere prevents coarsening
        for i in i_min..i_max.min(mx) {
            for j in j_min..j_max.min(my) {
                for k in k_min..k_max.min(mz) {
                    if markers[[i, j, k]] > 0 {
                        return false; // Cannot coarsen if any cell needs refinement
                    }
                }
            }
        }

        // Additionally, check that children are all leaves if they exist
        if let Some(ref children) = node.children {
            for child in children.iter() {
                if !child.is_leaf() {
                    return false; // Cannot coarsen if children have grandchildren
                }
            }
        }

        true
    }

    /// Count total nodes
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.count_nodes(&self.root)
    }

    fn count_nodes(&self, node: &OctreeNode) -> usize {
        Self::count_nodes_recursive(node)
    }

    fn count_nodes_recursive(node: &OctreeNode) -> usize {
        let mut count = 1;
        if let Some(ref children) = node.children {
            for child in children.iter() {
                count += Self::count_nodes_recursive(child);
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
        Self::count_leaves_recursive(node)
    }

    fn count_leaves_recursive(node: &OctreeNode) -> usize {
        if node.is_leaf() {
            1
        } else if let Some(ref children) = node.children {
            children.iter().map(Self::count_leaves_recursive).sum()
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
