//! Eigenspace PAM spectral fixtures.
//!
//! The routines in this module encode theorem-level passive acoustic mapping
//! relationships used by documentation and bindings. They do not simulate
//! stochastic receive snapshots; callers that need sample covariance behavior
//! should use the beamforming subspace kernels.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Invalid parameters for the deterministic eigenspace spectrum.
#[derive(Debug, Clone, PartialEq)]
pub enum EigenspaceSpectrumError {
    /// The receive aperture must contain at least one element.
    EmptyAperture,
    /// The signal subspace rank must satisfy `0 < n_sources < n_elements`.
    InvalidSourceRank {
        /// Number of receive elements.
        n_elements: usize,
        /// Number of incoherent point sources.
        n_sources: usize,
    },
    /// The signal power must be finite and strictly positive.
    InvalidSignalPower {
        /// Supplied signal power.
        signal_power: f64,
    },
    /// The noise power must be finite and strictly positive.
    InvalidNoisePower {
        /// Supplied noise power.
        noise_power: f64,
    },
}

impl Display for EigenspaceSpectrumError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyAperture => write!(f, "n_elements must be greater than zero"),
            Self::InvalidSourceRank {
                n_elements,
                n_sources,
            } => write!(
                f,
                "n_sources must satisfy 0 < n_sources < n_elements; got n_sources={n_sources}, n_elements={n_elements}"
            ),
            Self::InvalidSignalPower { signal_power } => write!(
                f,
                "signal_power must be finite and greater than zero; got {signal_power}"
            ),
            Self::InvalidNoisePower { noise_power } => write!(
                f,
                "noise_power must be finite and greater than zero; got {noise_power}"
            ),
        }
    }
}

impl Error for EigenspaceSpectrumError {}

/// Deterministic eigenspace PAM covariance eigenvalue spectrum.
///
/// For `K` incoherent point sources with equal signal power `signal_power` and
/// spatially white noise power `noise_power`, Theorem 22.2 gives a rank-`K`
/// perturbation of `noise_power * I`: the `K` signal eigenvalues are
/// `signal_power + noise_power`, while the remaining `N-K` eigenvalues are
/// `noise_power`.
///
/// Returns the eigenvalues sorted descending. For a Hermitian positive
/// semidefinite cross-spectral matrix these eigenvalues are also the singular
/// values.
///
/// # Errors
///
/// Returns [`EigenspaceSpectrumError`] when the aperture is empty, the source
/// rank is not in `(0, n_elements)`, or either power is non-finite or
/// non-positive.
pub fn eigenspace_covariance_eigenvalues(
    n_elements: usize,
    n_sources: usize,
    signal_power: f64,
    noise_power: f64,
) -> Result<Vec<f64>, EigenspaceSpectrumError> {
    if n_elements == 0 {
        return Err(EigenspaceSpectrumError::EmptyAperture);
    }
    if n_sources == 0 || n_sources >= n_elements {
        return Err(EigenspaceSpectrumError::InvalidSourceRank {
            n_elements,
            n_sources,
        });
    }
    if !signal_power.is_finite() || signal_power <= 0.0 {
        return Err(EigenspaceSpectrumError::InvalidSignalPower { signal_power });
    }
    if !noise_power.is_finite() || noise_power <= 0.0 {
        return Err(EigenspaceSpectrumError::InvalidNoisePower { noise_power });
    }

    let signal_eigenvalue = signal_power + noise_power;
    let mut eigenvalues = Vec::with_capacity(n_elements);
    eigenvalues.extend(std::iter::repeat_n(signal_eigenvalue, n_sources));
    eigenvalues.extend(std::iter::repeat_n(noise_power, n_elements - n_sources));
    Ok(eigenvalues)
}
