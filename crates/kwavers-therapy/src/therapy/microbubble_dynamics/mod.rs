//! Microbubble Dynamics Application Layer
//!
//! Application services and use cases for therapeutic microbubble simulation.
//!
//! ## Architecture - Clean Architecture Application Layer
//!
//! This module implements the **Application Layer** which orchestrates domain
//! entities and coordinates infrastructure services to fulfill use cases.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │               Presentation Layer (API/CLI)                  │
//! └─────────────────────────────────────────────────────────────┘
//!                          │
//!                          ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │          Application Layer (This Module)                    │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
//! │  │   Service   │  │   Command   │  │    Query    │        │
//! │  │ Orchestrate │  │   Handlers  │  │   Handlers  │        │
//! │  └─────────────┘  └─────────────┘  └─────────────┘        │
//! └─────────────────────────────────────────────────────────────┘
//!        │                    │                    │
//!        ▼                    ▼                    ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Domain Layer                             │
//! │  - MicrobubbleState  - MarmottantShellProperties            │
//! │  - DrugPayload       - RadiationForce                       │
//! └─────────────────────────────────────────────────────────────┘
//!        │                    │
//!        ▼                    ▼
//! ┌─────────────────┐  ┌─────────────────┐
//! │ Infrastructure  │  │   Core/Error    │
//! │ - ODE Solver    │  │ - Result Types  │
//! │ - Field Access  │  │ - Validation    │
//! └─────────────────┘  └─────────────────┘
//! ```
//!
//! ## Responsibilities
//!
//! The application layer is responsible for:
//!
//! 1. **Use Case Orchestration**
//!    - Coordinate domain entities (MicrobubbleState, MarmottantShellProperties)
//!    - Integrate infrastructure (Keller-Miksis solver, acoustic field access)
//!    - Enforce transaction boundaries and business workflows
//!
//! 2. **Domain-Infrastructure Mapping**
//!    - Convert between domain models and infrastructure types
//!    - Map `MicrobubbleState` ↔ `BubbleState` (Keller-Miksis)
//!    - Extract field data from grids at bubble positions
//!
//! 3. **Event Coordination** (Future)
//!    - Emit domain events (bubble rupture, cavitation, drug release)
//!    - Handle event subscribers and side effects
//!
//! ## Module Structure
//!
//! ```text
//! microbubble_dynamics/
//! ├── service.rs           - MicrobubbleDynamicsService (main orchestrator)
//! ├── tests/
//! │   ├── integration_tests.rs  - Full dynamics integration tests
//! │   └── validation_tests.rs   - Analytical validation tests
//! └── mod.rs               - This file
//! ```
//!
//! ## Key Components
//!
//! ### MicrobubbleDynamicsService
//!
//! Primary application service coordinating:
//! - Keller-Miksis ODE integration
//! - Marmottant shell state updates
//! - Radiation force calculations
//! - Drug release kinetics
//! - Cavitation detection
//!
//! ### Helper Functions
//!
//! - `sample_acoustic_field_at_position`: Extract local field properties
//! - Field gradient calculations using finite differences
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use kwavers_therapy::therapy::microbubble_dynamics::{
//!     MicrobubbleDynamicsService, DrugPayload, DrugLoadingMode,
//! };
//! use kwavers_physics::therapy::microbubble::{
//!     MicrobubbleState, MarmottantShellProperties, Position3D, PressureGradient3D,
//! };
//! use aequitas::systems::si::quantities::{
//!     Length, MassDensity, Pressure, PressureGradient, PressureRate, Time,
//! };
//!
//! // Create microbubble with drug payload
//! let position = Position3D::new(
//!     Length::from_base(0.01),
//!     Length::from_base(0.02),
//!     Length::from_base(0.03),
//! );
//! let mut bubble = MicrobubbleState::drug_loaded(
//!     Length::from_base(2.0e-6),
//!     MassDensity::from_base(50.0),
//!     position,
//! )
//! .unwrap();
//! let mut shell = MarmottantShellProperties::drug_delivery(bubble.radius_equilibrium).unwrap();
//! let mut drug = DrugPayload::doxorubicin(bubble.volume().into_base()).unwrap();
//!
//! // Create dynamics service
//! let service = MicrobubbleDynamicsService::from_microbubble_state(&bubble).unwrap();
//!
//! // Simulate dynamics
//! let acoustic_pressure = Pressure::from_base(1.0e5); // 100 kPa
//! let pressure_gradient = PressureGradient3D::new(
//!     PressureGradient::from_base(1.0e5),
//!     PressureGradient::from_base(0.0),
//!     PressureGradient::from_base(0.0),
//! );
//! let dt = Time::from_base(1.0e-6); // 1 microsecond timestep
//!
//! service.update_bubble_dynamics(
//!     &mut bubble,
//!     &mut shell,
//!     &mut drug,
//!     acoustic_pressure,
//!     pressure_gradient,
//!     PressureRate::from_base(0.0), // slowly-varying approximation
//!     Time::from_base(0.0),         // time `s`
//!     dt,
//! ).unwrap();
//!
//! println!("Bubble radius: {:.2} μm", bubble.radius.into_base() * 1e6);
//! println!("Drug released: {:.1}%", drug.release_fraction() * 100.0);
//! ```
//!
//! ## Design Patterns
//!
//! - **Application Service Pattern**: Service layer coordinates domain logic
//! - **Adapter Pattern**: Maps between domain and infrastructure types
//! - **Command Pattern** (Future): Commands for bubble dynamics operations
//! - **Repository Pattern** (Future): Persistence of bubble populations
//!
//! ## Testing Strategy
//!
//! - **Unit Tests**: Individual service methods (in `service.rs`)
//! - **Integration Tests**: Full dynamics simulation cycles
//! - **Validation Tests**: Compare against analytical solutions
//! - **Property Tests**: Invariants (energy conservation, mass conservation)
//!
//! ## References
//!
//! - Clean Architecture (Robert C. Martin, 2017)
//! - Domain-Driven Design (Eric Evans, 2003)
//! - Patterns of Enterprise Application Architecture (Martin Fowler, 2002)

pub mod drug_payload;
pub mod service;

// Tests are inline in service.rs for now
// #[cfg(test)]
// pub mod tests;

// Drug-payload value objects (therapy-delivery concern; moved here from the
// former kwavers-domain therapy module).
pub use drug_payload::{DrugLoadingMode, DrugPayload};
// Re-export main service for convenience
pub use service::{sample_acoustic_field_at_position, MicrobubbleDynamicsService};
