use thiserror::Error;

/// Input or arithmetic failure from a transient gas diffusion step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum GasTransferError {
    /// Input violates the finite, positive geometry or concentration contract.
    #[error("invalid spherical gas diffusion input: {0}")]
    InvalidInput(&'static str),
    /// Grid, time step, or solution cannot be represented at this precision.
    #[error("spherical gas diffusion state is not representable")]
    Unrepresentable,
    /// The time step would produce a negative dissolved concentration.
    #[error("spherical gas diffusion step produces a negative concentration")]
    NegativeConcentration,
}
