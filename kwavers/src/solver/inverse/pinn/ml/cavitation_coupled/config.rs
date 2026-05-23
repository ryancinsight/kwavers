use crate::physics::bubble_dynamics::BubbleParameters;

/// Cavitation coupling configuration
#[derive(Debug, Clone)]
pub struct CavitationCouplingConfig {
    /// Enable bubble-acoustic coupling
    pub enable_coupling: bool,
    /// Coupling strength (0 = no coupling, 1 = full coupling)
    pub coupling_strength: f64,
    /// Bubble parameters for cavitation
    pub bubble_params: BubbleParameters,
    /// Number of bubbles per coupling point
    pub bubbles_per_point: usize,
    /// Enable multi-bubble interactions
    pub multi_bubble_effects: bool,
    /// Enable nonlinear acoustic effects from bubbles
    pub nonlinear_acoustic: bool,
    /// Center frequency for resonance calculation (Hz)
    pub center_frequency: f64,
    /// Speed of sound in the medium (m/s)
    pub sound_speed: f64,
    /// Domain size for bubble field [Lx, Ly, Lz]
    pub domain_size: Vec<f64>,
}

impl Default for CavitationCouplingConfig {
    fn default() -> Self {
        Self {
            enable_coupling: true,
            coupling_strength: 0.5,
            bubble_params: BubbleParameters::default(),
            bubbles_per_point: 1,
            multi_bubble_effects: false,
            nonlinear_acoustic: true,
            center_frequency: 2.5e6, // 2.5 MHz default
            sound_speed: crate::core::constants::fundamental::SOUND_SPEED_TISSUE, // Water/Tissue default
            domain_size: vec![1e-2, 1e-2, 1e-2],                                  // 1cm³ domain
        }
    }
}

/// Cavitation-acoustic coupling problem type
#[derive(Debug, Clone, PartialEq)]
pub enum CavitationCouplingType {
    /// Weak coupling: acoustic drives bubbles, no back-coupling
    Weak,
    /// Strong coupling: mutual interaction with scattering
    Strong,
    /// Multi-bubble coupling with collective effects
    MultiBubble,
}
