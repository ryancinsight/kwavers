use kwavers_physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters;
use kwavers_physics::factory::models::BubbleModel;

/// Construction-time configuration for `BubbleDynamicsPlugin`.
#[derive(Debug, Clone)]
pub struct BubbleDynamicsConfig {
    /// Which ODE model to use for the bubble-wall motion.
    pub model: BubbleModel,
    /// When `true`, seed eight additional bubbles at ±¼-domain offsets from
    /// the grid centre (focal-zone nucleation cloud).
    pub nucleation: bool,
    /// Physical parameters shared by all seeded bubbles.
    pub params: BubbleParameters,
}

impl Default for BubbleDynamicsConfig {
    fn default() -> Self {
        Self {
            model: BubbleModel::KellerMiksis,
            nucleation: false,
            params: BubbleParameters::default(),
        }
    }
}
