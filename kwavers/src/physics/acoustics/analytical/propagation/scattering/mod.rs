//! Wave scattering physics module

#[derive(Debug)]
pub struct ScatteringCalculator;

impl ScatteringCalculator {
    #[must_use]
    pub fn new(_frequency: f64, _wave_speed: f64) -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use crate::core::constants::numerical::MHZ_TO_HZ;

    /// ScatteringCalculator::new constructs without panic for physical parameters.
    #[test]
    fn new_constructs_for_physical_parameters() {
        let _sc = ScatteringCalculator::new(MHZ_TO_HZ, SOUND_SPEED_WATER_SIM);
    }

    /// Debug output is non-empty.
    #[test]
    fn debug_non_empty() {
        let sc = ScatteringCalculator::new(500e3, 3400.0);
        let s = format!("{sc:?}");
        assert!(
            !s.is_empty(),
            "ScatteringCalculator debug must not be empty"
        );
    }
}
