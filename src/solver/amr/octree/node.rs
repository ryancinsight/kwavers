// octree/node.rs - Octree node implementation

/// Node in the octree structure
#[derive(Debug, Clone, Default)]
pub struct OctreeNode {
    /// Node index in the octree
    pub(super) index: usize,
    /// Refinement level (0 = root, positive = refined)
    level: i32,
    /// Spatial bounds (i, j, k) start
    bounds_min: (usize, usize, usize),
    /// Spatial bounds (i, j, k) end (exclusive)
    bounds_max: (usize, usize, usize),
    /// Parent node index (None for root)
    parent: Option<usize>,
    /// Child node indices (8 for full refinement)
    children: Option<[usize; 8]>,
    /// Node status
    status: NodeStatus,
}

/// Node status in the octree
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum NodeStatus {
    #[default]
    Active,
    Refined,
    Coarsened,
    Boundary,
}

impl OctreeNode {
    /// Create root node
    pub fn root(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            index: 0,
            level: 0,
            bounds_min: (0, 0, 0),
            bounds_max: (nx, ny, nz),
            parent: None,
            children: None,
            status: NodeStatus::Active,
        }
    }

    /// Create child node
    pub fn child(
        parent_index: usize,
        parent_level: i32,
        bounds_min: (usize, usize, usize),
        bounds_max: (usize, usize, usize),
    ) -> Self {
        Self {
            index: 0, // Will be set when added to octree
            level: parent_level + 1,
            bounds_min,
            bounds_max,
            parent: Some(parent_index),
            children: None,
            status: NodeStatus::Active,
        }
    }

    /// Check if node is a leaf
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// Check if node is refined
    pub fn is_refined(&self) -> bool {
        self.children.is_some()
    }

    /// Get node level
    pub fn level(&self) -> i32 {
        self.level
    }

    /// Get bounds
    pub fn bounds_min(&self) -> (usize, usize, usize) {
        self.bounds_min
    }

    pub fn bounds_max(&self) -> (usize, usize, usize) {
        self.bounds_max
    }

    /// Get size
    pub fn size(&self) -> (usize, usize, usize) {
        let (i_min, j_min, k_min) = self.bounds_min;
        let (i_max, j_max, k_max) = self.bounds_max;
        (i_max - i_min, j_max - j_min, k_max - k_min)
    }

    /// Check if point is inside node
    pub fn contains(&self, i: usize, j: usize, k: usize) -> bool {
        let (i_min, j_min, k_min) = self.bounds_min;
        let (i_max, j_max, k_max) = self.bounds_max;

        i >= i_min && i < i_max && j >= j_min && j < j_max && k >= k_min && k < k_max
    }

    /// Set children indices
    pub fn set_children(&mut self, children: [usize; 8]) {
        self.children = Some(children);
        self.status = NodeStatus::Refined;
    }

    /// Clear children
    pub fn clear_children(&mut self) {
        self.children = None;
        self.status = NodeStatus::Active;
    }

    /// Get children
    pub fn children(&self) -> Option<&[usize; 8]> {
        self.children.as_ref()
    }
}
