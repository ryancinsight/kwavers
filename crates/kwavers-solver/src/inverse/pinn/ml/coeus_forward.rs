//! Error conversion at the Coeus-to-Kwavers PINN boundary.

use coeus_autograd::Var;
use coeus_nn::ModuleError;
use kwavers_core::error::{KwaversError, KwaversResult};

/// Convert a fallible Coeus module evaluation into the solver error contract.
pub(crate) fn map_forward<T, B>(
    result: Result<Var<T, B>, ModuleError<B::Error>>,
    context: &'static str,
) -> KwaversResult<Var<T, B>>
where
    T: coeus_core::Scalar,
    B: coeus_ops::BackendOps<T> + Default,
{
    result.map_err(|error| KwaversError::Other(anyhow::Error::new(error).context(context)))
}

/// Convert a fallible Coeus backend operation into the solver error contract.
pub(crate) fn map_backend<T, E>(result: Result<T, E>, context: &'static str) -> KwaversResult<T>
where
    E: std::error::Error + Send + Sync + 'static,
{
    result.map_err(|error| KwaversError::Other(anyhow::Error::new(error).context(context)))
}
