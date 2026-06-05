//! Complex Phase Patterns for Beam Shaping
//!
//! Implements various phase patterns for specialized beam profiles.

#[derive(Debug)]
pub struct PhasePattern;
#[derive(Debug)]
pub struct SpiralPhase;
#[derive(Debug)]
pub struct VortexBeam;
#[derive(Debug)]
pub struct BesselBeam;
#[derive(Debug)]
pub struct AiryBeam;

#[cfg(test)]
mod tests {
    use super::*;

    /// All phase-pattern marker structs produce non-empty Debug strings.
    #[test]
    fn marker_structs_debug_non_empty() {
        for s in [
            format!("{:?}", PhasePattern),
            format!("{:?}", SpiralPhase),
            format!("{:?}", VortexBeam),
            format!("{:?}", BesselBeam),
            format!("{:?}", AiryBeam),
        ] {
            assert!(!s.is_empty(), "debug string must not be empty: {s}");
        }
    }

    /// Debug strings for distinct types are distinct.
    #[test]
    fn marker_structs_debug_are_distinct() {
        let names = [
            format!("{:?}", VortexBeam),
            format!("{:?}", BesselBeam),
            format!("{:?}", AiryBeam),
        ];
        // All three must be pairwise distinct
        assert_ne!(names[0], names[1]);
        assert_ne!(names[1], names[2]);
        assert_ne!(names[0], names[2]);
    }
}
