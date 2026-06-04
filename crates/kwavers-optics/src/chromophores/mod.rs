//! Chromophore Spectral Database

pub mod hemoglobin;
pub mod spectrum;

#[cfg(test)]
mod tests;

pub use hemoglobin::HemoglobinDatabase;
pub use spectrum::ExtinctionSpectrum;
