//! Wavelet basis types.

/// Wavelet basis functions
#[derive(Debug, Clone, Copy)]
pub enum WaveletBasis {
    /// Haar wavelet (simplest)
    Haar,
    /// Daubechies wavelets
    Daubechies(usize),
    /// Cohen-Daubechies-Feauveau wavelets
    CDF(usize, usize),
}
