//! A runtime restart width bridged onto Athena's compile-time `Gmres`.
//!
//! Athena fixes the restart width as a const generic so the Arnoldi basis and
//! the Hessenberg workspace are sized at compile time and every recurrence step
//! monomorphizes. kwavers chooses the width at runtime — it arrives as
//! [`GMRESConfig::krylov_dim`](super::GMRESConfig) or as the boundary-element
//! assembler's `restart` argument — so this module carries a fixed ladder of
//! instantiations and rounds a request up to the smallest width covering it.
//!
//! Rounding up costs no correctness. GMRES(m) minimises the residual over the
//! Krylov subspace `K_m(A, r₀)`, which contains `K_m'(A, r₀)` for every
//! `m' <= m` (Saad & Schultz 1986, §2), so a wider restart searches a superset
//! of the requested subspace. The trade is basis memory for subspace depth.

use athena_core::{
    ConvergencePolicy, Gmres, GmresWorkspace, IterationObserver, LinearOperator, NoObserver,
    Preconditioner, SolveError, SolveReport,
};
use athena_leto::{LetoBackend, LetoBackendError};
use core::fmt;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::Array1;

/// The CPU backend every kwavers Krylov solve runs on.
type Backend = LetoBackend<f64>;

/// Reusable restarted-GMRES workspace for a fixed dimension and restart width.
///
/// Construction performs every allocation and prepares Athena's reductions, so
/// repeated solves at the same dimension — successive Newton steps, successive
/// coupled time steps — reuse the Krylov basis instead of reallocating it.
pub struct KrylovWorkspace {
    inner: Ladder,
    dimension: usize,
}

/// The restart widths Athena is instantiated at.
///
/// The ladder is geometric, so any request lands within a factor of two of its
/// width. That bounds both the memory a solve reserves and the number of
/// `Gmres` monomorphisations this crate carries.
enum Ladder {
    W8(GmresWorkspace<Backend, 8>),
    W16(GmresWorkspace<Backend, 16>),
    W32(GmresWorkspace<Backend, 32>),
    W64(GmresWorkspace<Backend, 64>),
    W128(GmresWorkspace<Backend, 128>),
    W256(GmresWorkspace<Backend, 256>),
}

/// Ladder rung selected for a requested restart width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RestartWidth {
    W8,
    W16,
    W32,
    W64,
    W128,
    W256,
}

impl RestartWidth {
    /// Smallest rung at least as wide as `requested`, saturating at the top.
    ///
    /// A request above the ceiling is served by the ceiling rather than
    /// rejected: the restart is a tuning parameter, and capping it costs
    /// subspace depth per cycle, not correctness.
    pub(super) const fn covering(requested: usize) -> Self {
        if requested <= 8 {
            Self::W8
        } else if requested <= 16 {
            Self::W16
        } else if requested <= 32 {
            Self::W32
        } else if requested <= 64 {
            Self::W64
        } else if requested <= 128 {
            Self::W128
        } else {
            Self::W256
        }
    }
}

impl KrylovWorkspace {
    /// Allocate a workspace for `dimension` unknowns and a requested restart.
    ///
    /// # Errors
    ///
    /// Returns [`NumericalError::SolverFailed`] when the backend cannot
    /// allocate the Krylov basis or prepare its reductions.
    pub fn new(restart: usize, dimension: usize) -> KwaversResult<Self> {
        let backend = Backend::default();
        let inner = match RestartWidth::covering(restart) {
            RestartWidth::W8 => {
                GmresWorkspace::<Backend, 8>::new(&backend, dimension).map(Ladder::W8)
            }
            RestartWidth::W16 => {
                GmresWorkspace::<Backend, 16>::new(&backend, dimension).map(Ladder::W16)
            }
            RestartWidth::W32 => {
                GmresWorkspace::<Backend, 32>::new(&backend, dimension).map(Ladder::W32)
            }
            RestartWidth::W64 => {
                GmresWorkspace::<Backend, 64>::new(&backend, dimension).map(Ladder::W64)
            }
            RestartWidth::W128 => {
                GmresWorkspace::<Backend, 128>::new(&backend, dimension).map(Ladder::W128)
            }
            RestartWidth::W256 => {
                GmresWorkspace::<Backend, 256>::new(&backend, dimension).map(Ladder::W256)
            }
        }
        .map_err(|error| backend_failure("GMRES workspace allocation", &error))?;
        Ok(Self { inner, dimension })
    }

    /// Unknowns this workspace was allocated for.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Restart width this workspace solves at.
    #[must_use]
    const fn width(&self) -> usize {
        match self.inner {
            Ladder::W8(_) => 8,
            Ladder::W16(_) => 16,
            Ladder::W32(_) => 32,
            Ladder::W64(_) => 64,
            Ladder::W128(_) => 128,
            Ladder::W256(_) => 256,
        }
    }

    /// Solve `A·x = b` with restarted GMRES.
    ///
    /// `solution` carries the initial iterate in and the final iterate out. A
    /// solve that stalls, stagnates, or breaks down is reported in the returned
    /// [`SolveReport`] rather than as an error: the last iterate is still
    /// present, and whether it is usable is the caller's judgement.
    ///
    /// # Errors
    ///
    /// Returns [`NumericalError::MatrixDimension`] when the operator, the
    /// vectors, and the workspace disagree on the system dimension, and
    /// [`NumericalError::SolverFailed`] for a backend failure.
    pub fn solve<O, P>(
        &mut self,
        operator: &O,
        preconditioner: &P,
        right_hand_side: &Array1<f64>,
        solution: &mut Array1<f64>,
        policy: ConvergencePolicy<f64>,
    ) -> KwaversResult<SolveReport<f64>>
    where
        O: LinearOperator<Backend>,
        P: Preconditioner<Backend>,
    {
        self.solve_with_observer(
            operator,
            preconditioner,
            right_hand_side,
            solution,
            policy,
            &mut NoObserver,
        )
    }

    /// Solve `A·x = b` while reporting every checked residual to `observer`.
    ///
    /// Athena accumulates no residual history itself; a caller that wants one
    /// supplies the observer that records it. [`Self::solve`] is this call with
    /// the discarding observer.
    ///
    /// # Errors
    ///
    /// See [`Self::solve`].
    pub fn solve_with_observer<O, P, Obs>(
        &mut self,
        operator: &O,
        preconditioner: &P,
        right_hand_side: &Array1<f64>,
        solution: &mut Array1<f64>,
        policy: ConvergencePolicy<f64>,
        observer: &mut Obs,
    ) -> KwaversResult<SolveReport<f64>>
    where
        O: LinearOperator<Backend>,
        P: Preconditioner<Backend>,
        Obs: IterationObserver<f64>,
    {
        let backend = Backend::default();
        macro_rules! run {
            ($width:literal, $workspace:expr) => {
                Gmres::<Backend, $width>::solve_with_observer(
                    &backend,
                    operator,
                    preconditioner,
                    right_hand_side,
                    solution,
                    $workspace,
                    policy,
                    observer,
                )
            };
        }

        let outcome = match &mut self.inner {
            Ladder::W8(workspace) => run!(8, workspace),
            Ladder::W16(workspace) => run!(16, workspace),
            Ladder::W32(workspace) => run!(32, workspace),
            Ladder::W64(workspace) => run!(64, workspace),
            Ladder::W128(workspace) => run!(128, workspace),
            Ladder::W256(workspace) => run!(256, workspace),
        };
        outcome.map_err(|error| solve_failure(&error))
    }
}

impl fmt::Debug for KrylovWorkspace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KrylovWorkspace")
            .field("dimension", &self.dimension)
            .field("restart", &self.width())
            .finish()
    }
}

/// Convert an Athena solve failure into kwavers' typed numerical vocabulary.
fn solve_failure(error: &SolveError<LetoBackendError>) -> KwaversError {
    match error {
        SolveError::DimensionMismatch {
            context,
            expected,
            actual,
        } => KwaversError::Numerical(NumericalError::MatrixDimension {
            operation: format!("GMRES {context}"),
            expected: expected.to_string(),
            actual: actual.to_string(),
        }),
        SolveError::Backend(backend) => backend_failure("GMRES solve", backend),
        // `SolveError` is `#[non_exhaustive]`: a variant Athena adds later must
        // surface as a solver failure rather than silently taking a branch that
        // was written for a different condition.
        other => KwaversError::Numerical(NumericalError::SolverFailed {
            method: "GMRES solve".to_owned(),
            reason: other.to_string(),
        }),
    }
}

/// Convert a backend allocation or arithmetic failure into a typed error.
fn backend_failure(operation: &str, error: &LetoBackendError) -> KwaversError {
    KwaversError::Numerical(NumericalError::SolverFailed {
        method: operation.to_owned(),
        reason: error.to_string(),
    })
}
