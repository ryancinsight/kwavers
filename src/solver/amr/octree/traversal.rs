// octree/traversal.rs - Octree traversal algorithms

use super::{Octree, OctreeNode};

/// Traversal order for octree
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TraversalOrder {
    PreOrder,
    PostOrder,
    LevelOrder,
}

/// Octree traversal iterator
pub struct Traversal<'a> {
    octree: &'a Octree,
    order: TraversalOrder,
    stack: Vec<usize>,
    visited: Vec<bool>,
}

impl<'a> Traversal<'a> {
    /// Create new traversal
    pub fn new(octree: &'a Octree, order: TraversalOrder) -> Self {
        let mut stack = Vec::new();
        stack.push(0); // Start with root

        Self {
            octree,
            order,
            stack,
            visited: vec![false; octree.node_count()],
        }
    }

    /// Visit leaves only
    pub fn leaves_only(octree: &'a Octree) -> impl Iterator<Item = &'a OctreeNode> + 'a {
        (0..octree.node_count()).filter_map(move |i| octree.node(i).filter(|n| n.is_leaf()))
    }

    /// Visit nodes at specific level
    pub fn at_level(octree: &'a Octree, level: i32) -> impl Iterator<Item = &'a OctreeNode> + 'a {
        (0..octree.node_count()).filter_map(move |i| octree.node(i).filter(|n| n.level() == level))
    }
}

impl<'a> Iterator for Traversal<'a> {
    type Item = &'a OctreeNode;

    fn next(&mut self) -> Option<Self::Item> {
        match self.order {
            TraversalOrder::PreOrder => self.next_preorder(),
            TraversalOrder::PostOrder => self.next_postorder(),
            TraversalOrder::LevelOrder => self.next_levelorder(),
        }
    }
}

impl<'a> Traversal<'a> {
    fn next_preorder(&mut self) -> Option<&'a OctreeNode> {
        while let Some(index) = self.stack.pop() {
            if self.visited[index] {
                continue;
            }

            self.visited[index] = true;

            if let Some(node) = self.octree.node(index) {
                // Add children to stack in reverse order
                if let Some(children) = node.children() {
                    for &child_idx in children.iter().rev() {
                        self.stack.push(child_idx);
                    }
                }

                return Some(node);
            }
        }

        None
    }

    fn next_postorder(&mut self) -> Option<&'a OctreeNode> {
        // Simplified postorder - would implement full algorithm
        self.next_preorder()
    }

    fn next_levelorder(&mut self) -> Option<&'a OctreeNode> {
        // Simplified level-order - would use queue instead of stack
        self.next_preorder()
    }
}
