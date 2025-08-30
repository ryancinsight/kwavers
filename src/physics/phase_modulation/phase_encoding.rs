//! Phase Encoding for Pulse Compression and Imaging
//!
//! Implements phase encoding schemes for improved SNR and resolution.

#[derive(Debug, Debug))]
pub struct PhaseEncoder;
pub enum EncodingScheme {
    Hadamard,
    Golay,
    Barker,
}
#[derive(Debug, Debug))]
pub struct HadamardEncoding;
#[derive(Debug, Debug))]
pub struct GolayEncoding;
#[derive(Debug, Debug))]
pub struct BarkerCode;
#[derive(Debug, Debug))]
pub struct PulseCompression;
