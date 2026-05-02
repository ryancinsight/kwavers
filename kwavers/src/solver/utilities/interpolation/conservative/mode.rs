//! Conservation mode selector for conservative interpolation.

/// Physical quantity to conserve during grid-to-grid field transfer.
///
/// Constrains the transfer operator to satisfy discrete conservation
/// laws up to machine precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConservationMode {
    /// Conserve mass: ∫ρ dV = const.
    Mass,
    /// Conserve energy: ∫E dV = const.
    Energy,
    /// Conserve momentum: ∫ρu dV = const.
    Momentum,
    /// Conserve all quantities (mass, energy, momentum).
    All,
    /// No conservation enforcement — standard interpolation.
    None,
}
