//! Pipeline type enumeration for compute shader dispatch.

/// Types of compute pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineType {
    /// FFT 3D (forward)
    FFT3D,
    /// Inverse FFT 3D
    IFFT3D,
    /// Element-wise multiply
    ElementWiseMultiply,
    /// Spatial derivative (k-space operator)
    SpatialDerivative,
}
