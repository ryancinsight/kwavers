//! Main heterogeneous media handler
//!
//! Coordinates interface detection, smoothing, and split formulation

use super::config::{HeterogeneousConfig, SmoothingMethod};
use super::interface_detection::InterfaceDetector;
use super::pressure_velocity_split::PressureVelocitySplit;
use super::smoothing::Smoother;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::KwaversResult;
use ndarray::Array3;

/// Heterogeneous media handler
#[derive(Debug)]
pub struct HeterogeneousHandler {
    config: HeterogeneousConfig,
    grid: Grid,
    /// Interface detector
    detector: InterfaceDetector,
    /// Smoothing processor
    smoother: Smoother,
    /// Pressure-velocity split handler
    pv_split: Option<PressureVelocitySplit>,
    /// Detected interface locations
    interface_mask: Option<Array3<bool>>,
    /// Smoothed density field
    density_smooth: Option<Array3<f64>>,
    /// Smoothed sound speed field
    sound_speed_smooth: Option<Array3<f64>>,
    /// Interface sharpness map
    sharpness_map: Option<Array3<f64>>,
}

impl HeterogeneousHandler {
    /// Create a new heterogeneous media handler
    pub fn new(config: HeterogeneousConfig, grid: Grid) -> Self {
        let detector = InterfaceDetector::new(config.interface_threshold, grid.clone());
        let smoother = Smoother::new(
            config.smoothing_method,
            config.smoothing_width,
            grid.clone(),
        );

        Self {
            config,
            grid,
            detector,
            smoother,
            pv_split: None,
            interface_mask: None,
            density_smooth: None,
            sound_speed_smooth: None,
            sharpness_map: None,
        }
    }

    /// Initialize with medium properties
    pub fn initialize(&mut self, medium: &dyn Medium, grid: &Grid) -> KwaversResult<()> {
        // Get medium properties as arrays
        let density = medium.density_array(grid);
        let sound_speed = medium.sound_speed_array(grid);

        // Detect interfaces
        self.interface_mask = Some(self.detector.detect(&density, &sound_speed)?);

        // Compute sharpness map for adaptive treatment
        if self.config.adaptive_treatment {
            self.sharpness_map = Some(self.detector.compute_sharpness(&density, &sound_speed));
        }

        // Apply smoothing if enabled
        if self.config.mitigate_gibbs {
            let (density_smooth, sound_speed_smooth) = self.smoother.smooth(
                &density,
                &sound_speed,
                self.interface_mask.as_ref().unwrap(),
            )?;
            self.density_smooth = Some(density_smooth.clone());
            self.sound_speed_smooth = Some(sound_speed_smooth.clone());

            // Initialize pressure-velocity split if enabled
            if self.config.use_pv_split {
                self.pv_split = Some(PressureVelocitySplit::new(
                    self.grid.clone(),
                    &density_smooth,
                    &sound_speed_smooth,
                ));
            }
        } else {
            self.density_smooth = Some(density);
            self.sound_speed_smooth = Some(sound_speed);
        }

        Ok(())
    }

    /// Get smoothed density field
    pub fn density(&self) -> Option<&Array3<f64>> {
        self.density_smooth.as_ref()
    }

    /// Get smoothed sound speed field
    pub fn sound_speed(&self) -> Option<&Array3<f64>> {
        self.sound_speed_smooth.as_ref()
    }

    /// Get interface mask
    pub fn interface_mask(&self) -> Option<&Array3<bool>> {
        self.interface_mask.as_ref()
    }

    /// Get sharpness map
    pub fn sharpness_map(&self) -> Option<&Array3<f64>> {
        self.sharpness_map.as_ref()
    }

    /// Update pressure field using split formulation
    pub fn update_pressure(
        &mut self,
        pressure: &mut Array3<f64>,
        velocity: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        if let Some(ref mut pv_split) = self.pv_split {
            pv_split.update_pressure(pressure, velocity, dt)?;

            // Apply interface correction if available
            if let Some(ref mask) = self.interface_mask {
                pv_split.apply_interface_correction(pressure, mask)?;
            }
        }
        Ok(())
    }

    /// Update velocity field using split formulation
    pub fn update_velocity(
        &mut self,
        velocity: &mut Array3<f64>,
        pressure: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        if let Some(ref mut pv_split) = self.pv_split {
            pv_split.update_velocity(velocity, pressure, dt)?;
        }
        Ok(())
    }

    /// Check if Gibbs mitigation is enabled
    pub fn is_gibbs_mitigation_enabled(&self) -> bool {
        self.config.mitigate_gibbs
    }

    /// Check if pressure-velocity split is enabled
    pub fn is_pv_split_enabled(&self) -> bool {
        self.config.use_pv_split
    }

    /// Get the smoothing method being used
    pub fn smoothing_method(&self) -> SmoothingMethod {
        self.config.smoothing_method
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;

    #[test]
    fn test_heterogeneous_handler_creation() {
        let config = HeterogeneousConfig::default();
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let handler = HeterogeneousHandler::new(config, grid);

        assert!(handler.is_gibbs_mitigation_enabled());
        assert!(handler.is_pv_split_enabled());
        assert_eq!(handler.smoothing_method(), SmoothingMethod::Gaussian);
    }

    #[test]
    fn test_initialization_with_homogeneous_medium() {
        let config = HeterogeneousConfig::default();
        let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3);
        let mut handler = HeterogeneousHandler::new(config, grid.clone());

        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);

        assert!(handler.initialize(&medium, &grid).is_ok());
        assert!(handler.density().is_some());
        assert!(handler.sound_speed().is_some());
        assert!(handler.interface_mask().is_some());
    }
}
