// Solver Factory trait abstraction
//
// Defines the abstract factory interface for solver creation,
// following the Dependency Inversion Principle.

#[cfg(test)]
mod tests;
pub mod traits;
pub mod types;

pub use traits::{
    ApolloFourierBackend, FourierBackend, GridParameters, MediumParameters, MeshProvider,
    RegistrationEngine, SolverFactory, SourceParameters,
};
pub use types::{FactoryConfiguration, FactoryError};
