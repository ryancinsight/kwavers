//! Transducer geometry abstraction for the linear Born inversion solver.
//!
//! # Why a trait
//!
//! Linear Born inversion is geometry-driven: the sensitivity matrix is
//! `A[(source,offset), voxel] = K(source_position, voxel, receiver_position)`.
//! Implementing kernels in terms of a concrete geometry (a transcranial bowl,
//! a multi-row ring, a linear array, a planar array) blocks reuse. The
//! [`TransducerGeometry`] trait lets the generic Born + PCG primitives accept
//! any geometry that exposes Cartesian element positions and a source-to-
//! receiver mapping.
//!
//! Geometry-specific shapes (hemisphere, ring, linear, planar) live in their
//! respective clinical adapters (`clinical::imaging::reconstruction::*`) and
//! `impl TransducerGeometry` for those types.
//!
//! # Default `receiver_indices` policy
//!
//! Many acquisition geometries do not have rotational symmetry; for those a
//! cyclic offset mapping `(source_idx + offset) % len()` is the natural
//! default. Geometries with stronger symmetry (e.g. a transcranial bowl that
//! supports azimuthal rotation about the cap axis) override this method.

use std::fmt::Debug;

/// Cartesian position of one transducer element [m].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementPosition {
    pub x_m: f64,
    pub y_m: f64,
    pub z_m: f64,
}

/// Abstract acquisition geometry consumed by the generic linear Born + PCG
/// kernels.
///
/// Implementations are stateless value types (or `Arc`-shared `Vec`-of-
/// positions wrappers); they expose the Cartesian element layout and the
/// source-to-receiver offset semantics specific to the array topology.
pub trait TransducerGeometry: Debug + Send + Sync {
    /// Cartesian element positions [m]. Indexing convention is implementor-
    /// defined; all sensitivity / Born kernels iterate over this slice
    /// directly.
    fn elements(&self) -> &[ElementPosition];

    /// Number of elements.
    fn len(&self) -> usize {
        self.elements().len()
    }

    /// `true` iff the geometry has no elements.
    fn is_empty(&self) -> bool {
        self.elements().is_empty()
    }

    /// Receiver indices for the cross-product `source × offsets`, in
    /// row-major order `(source_idx, offset_idx)`.
    ///
    /// Default: cyclic offset mapping `(source_idx + offset) % len()` —
    /// appropriate for ring arrays and any geometry with index-space
    /// translational symmetry. Override for geometries with continuous
    /// rotational symmetry (e.g. transcranial bowls) where the natural
    /// receiver mapping is "nearest element after rotating the source
    /// position by an angle proportional to `offset`".
    fn receiver_indices(&self, offsets: &[usize]) -> Vec<usize> {
        let n = self.len();
        let mut indices = Vec::with_capacity(n * offsets.len());
        for source_idx in 0..n {
            for offset in offsets {
                indices.push((source_idx + offset) % n.max(1));
            }
        }
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct LinearStrip {
        elements: Vec<ElementPosition>,
    }

    impl LinearStrip {
        fn new(n: usize, dx: f64) -> Self {
            let elements = (0..n)
                .map(|i| ElementPosition {
                    x_m: (i as f64) * dx,
                    y_m: 0.0,
                    z_m: 0.0,
                })
                .collect();
            Self { elements }
        }
    }

    impl TransducerGeometry for LinearStrip {
        fn elements(&self) -> &[ElementPosition] {
            &self.elements
        }
    }

    #[test]
    fn default_receiver_indices_is_cyclic_offset() {
        let strip = LinearStrip::new(4, 1.0);
        let indices = strip.receiver_indices(&[0, 1]);
        // For each source (4 sources) × each offset (2 offsets) = 8 entries.
        // (s=0,off=0)=0  (s=0,off=1)=1
        // (s=1,off=0)=1  (s=1,off=1)=2
        // (s=2,off=0)=2  (s=2,off=1)=3
        // (s=3,off=0)=3  (s=3,off=1)=0
        assert_eq!(indices, vec![0, 1, 1, 2, 2, 3, 3, 0]);
    }

    #[test]
    fn empty_geometry_reports_empty_and_zero_length() {
        #[derive(Debug)]
        struct Empty;
        impl TransducerGeometry for Empty {
            fn elements(&self) -> &[ElementPosition] {
                &[]
            }
        }
        assert!(Empty.is_empty());
        assert_eq!(Empty.len(), 0);
    }
}
