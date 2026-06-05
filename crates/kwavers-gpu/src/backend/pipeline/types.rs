//! Pipeline type enumeration for compute shader dispatch.

/// Types of compute pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineType {
    /// Element-wise multiply
    ElementWiseMultiply,
    /// Spatial derivative (k-space operator)
    SpatialDerivative,
}
