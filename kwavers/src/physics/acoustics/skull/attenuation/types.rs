//! Types for skull attenuation modeling

/// Bone type for material properties
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoneType {
    /// Outer dense bone layer
    Cortical,
    /// Inner porous bone structure
    Cancellous,
    /// Mixed/transitional regions
    Mixed { cortical_fraction: f64 },
}
