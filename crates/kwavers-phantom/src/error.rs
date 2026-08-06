use hyperion::TransportError;
use std::fmt;

/// Errors returned while constructing an optical phantom.
#[derive(Debug)]
#[non_exhaustive]
pub enum PhantomError {
    /// Hyperion rejected a wavelength or another optical coefficient input.
    Hyperion(TransportError<f64>),
    /// The computed optical properties violate the medium contract.
    InvalidOpticalProperties(String),
    /// A builder was used before its grid dimensions were configured.
    MissingDimensions,
}

impl fmt::Display for PhantomError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hyperion(error) => {
                write!(
                    formatter,
                    "Hyperion rejected an optical coefficient input: {error}"
                )
            }
            Self::InvalidOpticalProperties(error) => {
                write!(formatter, "invalid optical properties: {error}")
            }
            Self::MissingDimensions => {
                formatter.write_str("phantom dimensions must be set before building")
            }
        }
    }
}

impl std::error::Error for PhantomError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Hyperion(error) => Some(error),
            Self::InvalidOpticalProperties(_) | Self::MissingDimensions => None,
        }
    }
}

impl From<TransportError<f64>> for PhantomError {
    fn from(error: TransportError<f64>) -> Self {
        Self::Hyperion(error)
    }
}
