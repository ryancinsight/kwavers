/// Skull layer classification derived from bone volume fraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkullLayer {
    /// Free water / soft tissue (φ < 0.15).
    SoftTissue,
    /// Diploe — cancellous (trabecular) bone (0.15 ≤ φ < 0.75).
    Diploe,
    /// Cortical bone (φ ≥ 0.75).
    Cortical,
}
