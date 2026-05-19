// Solver Factory trait abstraction
//
// Defines the abstract factory interface for solver creation,
// following the Dependency Inversion Principle.

#[cfg(test)]
mod tests;
pub mod traits;
pub mod types;

pub use traits::{
    ApolloFourierBackend, FourierBackend, FactoryGridParameters, FactoryMediumParameters, MeshProvider,
    RegistrationEngine, SolverFactoryTrait, FactorySourceParameters,
};
pub use types::{FactoryConfiguration, FactoryError};
