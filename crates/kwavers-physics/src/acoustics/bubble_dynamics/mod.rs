//! Bubble Dynamics Module
//!
//! This module provides the core bubble dynamics calculations that are used by:
//! - Mechanics: for cavitation damage and erosion
//! - Optics: for sonoluminescence light emission
//! - Chemistry: for ROS generation and sonochemistry
//!
//! Based on the Keller-Miksis equation and extended models from literature.
//!
//! # ARCHITECTURE NOTE (SOC / SRP debt)
//!
//! The `adaptive_integration` and `imex_integration` sub-modules implement ODE
//! **time-stepping** logic (adaptive Runge-Kutta, IMEX schemes), which
//! architecturally belongs in the **solver layer** (e.g., `solver/forward/ode/`),
//! not the physics layer.  The physics layer should only define equations of
//! motion (Keller-Miksis, Rayleigh-Plesset, etc.).
//!
//! Tracked for future refactor — see `docs/IMPLEMENTATION_ROADMAP.md` §Layer Boundaries.
//!
//! ## Migration Path
//! 1. Create `solver/forward/ode/` with `AdaptiveRkSolver<E: BubbleOde>` trait
//! 2. Move `adaptive_integration` and `imex_integration` to `solver/forward/ode/`
//! 3. Physics layer keeps only the equation structs implementing `BubbleOde`
//! 4. Update all callers to the new solver-layer path

pub mod adaptive_integration; // NEW: Adaptive time-stepping for stiff ODEs
pub mod bjerknes_forces; // NEW: Bjerknes forces for bubble-bubble interactions
pub mod bubble_field;
pub mod bubble_state;
pub mod bubbly_medium; // Void-fraction → sound speed (Wood) + attenuation (Commander–Prosperetti)
pub mod cavitation_control;
pub mod dissolution; // Gas-diffusion dissolution (Epstein–Plesset 1950 + shelled extension)
pub mod encapsulated; // NEW: Encapsulated bubbles with shell dynamics (Church, Marmottant)
pub mod energy;
pub mod epstein_plesset; // Epstein-Plesset oscillation-STABILITY theorem (distinct from dissolution)
pub mod gilmore; // Gilmore equation for violent collapse // NEW: Comprehensive energy balance model
pub mod heterogeneous_nucleation;
pub mod imex_integration;

pub use energy::{update_temperature_energy_balance, EnergyBalanceCalculator};
pub mod integration; // NEW: Stable integration utilities extracted from monolithic file
pub mod interactions;
pub mod keller_miksis; // NEW: Extracted Keller-Miksis solver for modularity
pub mod rayleigh_plesset;
pub mod symplectic_integration; // Störmer-Verlet / Yoshida symplectic ODE (relocated from solver)
pub mod thermodynamics;

pub use adaptive_integration::{
    integrate_bubble_dynamics_adaptive, AdaptiveBubbleConfig, AdaptiveBubbleIntegrator,
    IntegrationStatistics,
};
pub use bjerknes_forces::{BjerknesCalculator, BjerknesConfig, BjerknesInteractionType}; // NEW: Bubble-bubble interaction forces
pub use bubbly_medium::{
    commander_prosperetti_attenuation, commander_prosperetti_phase_velocity,
    commander_prosperetti_wavenumber, mixture_density, wood_sound_speed,
};
pub use dissolution::{
    dissolution_time_numeric, integrate_dissolution, DissolutionModel, DissolutionTrajectory,
    EpsteinPlessetDissolution, GasDiffusionParams, ShellPermeationDissolution,
};
pub use bubble_field::{BubbleCloud, BubbleField, BubbleStateFields};
pub use bubble_state::{BubbleParameters, BubbleState, GasSpecies};
pub use cavitation_control::{
    CavitationMetrics, ControlOutput, ControlStrategy, FeedbackConfig, FeedbackController,
};
pub use encapsulated::{ChurchModel, MarmottantModel, ShellProperties}; // NEW: Encapsulated bubble models
pub use epstein_plesset::EpsteinPlessetStabilitySolver; // NEW: Epstein-Plesset stability analysis
pub use heterogeneous_nucleation::{
    ClassicalHeterogeneousNucleation, HeterogeneousNucleationModel,
};
pub use imex_integration::{
    integrate_bubble_dynamics_imex, BubbleIMEXConfig, BubbleIMEXIntegrator,
};
pub use integration::integrate_bubble_dynamics_stable; // NEW: Extracted integration utilities
pub use symplectic_integration::{
    integrate_bubble_dynamics_symplectic, stormer_verlet_step, yoshida4_step,
    BubbleSymplecticIntegrator, SymplecticConfig,
};
pub use interactions::{BjerknesForceComputer, BubbleInteractions, CollectiveEffects};
pub use keller_miksis::KellerMiksisModel; // NEW: Modular Keller-Miksis solver
pub use rayleigh_plesset::RayleighPlessetSolver;
