//! Optical source types
//!
//! This module contains optical source implementations for light propagation
//! simulations, including laser sources, LED arrays, and fiber optics.

pub mod fiber;
pub mod laser;
pub mod led;

pub use fiber::{FiberConfig, FiberSource};
pub use laser::{GaussianLaser, LaserConfig, LaserSource};
pub use led::{LEDConfig, LEDSource};

/// Optical source trait
pub trait OpticalSource {
    /// Get the optical power at a given time (W)
    fn optical_power(&self, t: f64) -> f64;

    /// Get the wavelength (m)
    fn wavelength(&self) -> f64;

    /// Get the beam profile
    fn beam_profile(&self, x: f64, y: f64, z: f64) -> f64;

    /// Get source positions
    fn positions(&self) -> Vec<(f64, f64, f64)>;

    /// Visit source positions through a callback.
    ///
    /// The default implementation preserves compatibility by adapting
    /// [`OpticalSource::positions`], and therefore may allocate. Built-in
    /// fiber, laser, and LED sources override this method so position
    /// consumers can stream positions without constructing a temporary `Vec`.
    /// The visitor is called once for every position in the same order as
    /// [`OpticalSource::positions`].
    fn for_each_position(&self, visitor: &mut dyn FnMut((f64, f64, f64))) {
        for position in self.positions() {
            visitor(position);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fiber_position_visitor_matches_owned_positions() {
        let source = FiberSource::new(FiberConfig::default(), (1.0, 2.0, 3.0));
        let mut visited = Vec::new();

        source.for_each_position(&mut |position| visited.push(position));

        assert_eq!(visited, source.positions());
        assert_eq!(visited.len(), 1);
    }

    #[test]
    fn laser_position_visitor_matches_owned_positions() {
        let source = GaussianLaser::new(LaserConfig::default(), (1.0, 2.0, 3.0), (0.0, 0.0, 1.0));
        let mut visited = Vec::new();

        source.for_each_position(&mut |position| visited.push(position));

        assert_eq!(visited, source.positions());
        assert_eq!(visited.len(), 1);
    }

    #[test]
    fn led_position_visitor_preserves_order() {
        let positions = vec![(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)];
        let source = LEDSource::new(LEDConfig::default(), positions);
        let mut visited = Vec::new();

        source.for_each_position(&mut |position| visited.push(position));

        assert_eq!(visited, source.positions());
        assert_eq!(
            visited,
            vec![(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
        );
    }
}
