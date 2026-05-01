/// Cavitation detection methods.
#[derive(Debug, Clone, Copy)]
pub enum CavitationDetectionMethod {
    /// Pressure threshold: voxel cavitates iff |p| > P_Blake.
    PressureThreshold,
    /// Resonance-enhanced threshold: P_eff = P_Blake / E(f) per voxel.
    Spectral,
    /// Combined: spectral method, encompassing threshold as E → 1 off-resonance.
    Combined,
}

/// Cavitation detector for therapeutic ultrasound.
///
/// Provides per-voxel and aggregate cavitation metrics for HIFU/FUS therapy
/// planning and safety monitoring. The Blake threshold is computed at
/// construction for a standard 1 µm air nucleus, adjustable via `new_with_radius`.
#[derive(Debug)]
pub struct TherapyCavitationDetector {
    /// Driving frequency (Hz).
    pub(crate) frequency: f64,
    /// Blake threshold pressure magnitude (Pa), stored as a positive value.
    pub blake_threshold: f64,
    /// Detection method.
    pub(crate) method: CavitationDetectionMethod,
}
