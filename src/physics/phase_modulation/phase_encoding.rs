//! Phase Encoding for Pulse Compression and Imaging
//!
//! Implements phase encoding schemes for improved SNR and resolution.

#[derive(Debug)]
pub struct PhaseEncoder;
#[derive(Debug)]
pub enum EncodingScheme {
    Hadamard,
    Golay,
    Barker,
}
#[derive(Debug)]
pub struct HadamardEncoding;
#[derive(Debug)]
pub struct GolayEncoding;
#[derive(Debug)]
pub struct BarkerCode;
#[derive(Debug)]
pub struct PulseCompression;
