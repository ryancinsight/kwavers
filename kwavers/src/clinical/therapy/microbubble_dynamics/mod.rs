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
//! use kwavers::clinical::therapy::microbubble_dynamics::MicrobubbleDynamicsService;
//! use kwavers::domain::therapy::microbubble::{
//!     MicrobubbleState, MarmottantShellProperties, DrugPayload,
//!     DrugLoadingMode, Position3D,
//! };
//!
//! // Create microbubble with drug payload
//! let position = Position3D::new(0.01, 0.02, 0.03);
//! let mut bubble = MicrobubbleState::drug_loaded(2.0, 50.0, position).unwrap();
//! let mut shell = MarmottantShellProperties::drug_delivery(bubble.radius_equilibrium).unwrap();
//! let mut drug = DrugPayload::doxorubicin(bubble.volume()).unwrap();
//!
//! // Create dynamics service
//! let service = MicrobubbleDynamicsService::from_microbubble_state(&bubble).unwrap();
//!
//! // Simulate dynamics
//! let acoustic_pressure = 1e5; // 100 kPa
//! let pressure_gradient = (1e5, 0.0, 0.0); // Pressure gradient [Pa/m]
//! let dt = 1e-6; // 1 microsecond timestep
//!
//! service.update_bubble_dynamics(
//!     &mut bubble,
//!     &mut shell,
//!     &mut drug,
//!     acoustic_pressure,
//!     pressure_gradient,
//!     0.0, // time
//!     dt,
//! ).unwrap();
//!
//! println!("Bubble radius: {:.2} μm", bubble.radius * 1e6);
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

pub mod service;

// Tests are inline in service.rs for now
// #[cfg(test)]
// pub mod tests;

// Re-export main service for convenience
pub use service::{sample_acoustic_field_at_position, MicrobubbleDynamicsService};
