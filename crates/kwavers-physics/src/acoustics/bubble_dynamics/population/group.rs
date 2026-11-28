/// A monodisperse slice of a bubble population: `count` bubbles,
/// each of equilibrium radius `radius` [m] holding
/// `gas_per_bubble` molecules.
///
/// Counts are real-valued (ensemble weights), not integers — the
/// operators are ensemble bookkeeping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BubbleGroup {
    /// Equilibrium radius of each bubble [m].
    pub radius: f64,
    /// Number of bubbles in the group (ensemble weight).
    pub count: f64,
    /// Gas content of each bubble [molecules].
    pub gas_per_bubble: f64,
}

impl BubbleGroup {
    /// Total gas held by the group [molecules].
    #[must_use]
    pub fn total_gas(&self) -> f64 {
        self.count * self.gas_per_bubble
    }
}
