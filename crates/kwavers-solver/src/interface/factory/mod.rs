// Solver Factory trait abstraction
//
// Defines the abstract factory interface for solver creation,
// following the Dependency Inversion Principle.

#[cfg(test)]
mod tests;
pub mod traits;
pub mod types;

pub use traits::{
    ApolloFourierBackend, FactoryGridParameters, FactoryMediumParameters, FactorySourceParameters,
    FourierBackend, MeshProvider, RegistrationEngine, SolverComponent, SolverFactoryTrait,
};
pub use types::{FactoryConfiguration, FactoryError};
