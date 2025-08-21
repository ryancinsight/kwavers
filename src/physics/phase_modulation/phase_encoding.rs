//! Phase Encoding for Pulse Compression and Imaging
//!
//! Implements phase encoding schemes for improved SNR and resolution.

pub struct PhaseEncoder;
pub enum EncodingScheme {
    Hadamard,
    Golay,
    Barker,
}
pub struct HadamardEncoding;
pub struct GolayEncoding;
pub struct BarkerCode;
pub struct PulseCompression;
