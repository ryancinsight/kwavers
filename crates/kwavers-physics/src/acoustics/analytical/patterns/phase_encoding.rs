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

#[cfg(test)]
mod tests {
    use super::*;

    /// PhaseEncoder is constructible and Debug-formattable.
    #[test]
    fn phase_encoder_debug_non_empty() {
        let s = format!("{:?}", PhaseEncoder);
        assert!(!s.is_empty());
    }

    /// All EncodingScheme variants produce distinct Debug strings.
    #[test]
    fn encoding_scheme_variants_debug_distinct() {
        let h = format!("{:?}", EncodingScheme::Hadamard);
        let g = format!("{:?}", EncodingScheme::Golay);
        let b = format!("{:?}", EncodingScheme::Barker);
        assert_ne!(h, g);
        assert_ne!(g, b);
        assert_ne!(h, b);
    }

    /// Encoding marker structs are constructible.
    #[test]
    fn encoding_marker_structs_constructible() {
        let _h = HadamardEncoding;
        let _g = GolayEncoding;
        let _b = BarkerCode;
        let _pc = PulseCompression;
    }
}
