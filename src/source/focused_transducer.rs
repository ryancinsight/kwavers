use super::{Source, Apodization};
use crate::grid::Grid;
use crate::signal::Signal;
use std::sync::Arc;

#[derive(Debug)]
pub struct FocusedTransducer {
    x0: f64,                    // x position of transducer center
    y0: f64,                    // y position of transducer center
    z0: f64,                    // z position of transducer center
    focus_x: f64,              // x position of focal point
    focus_y: f64,              // y position of focal point
    focus_z: f64,              // z position of focal point
    radius: f64,               // radius of transducer
    signal: Box<dyn Signal>,    // input signal
    apodization: Box<dyn Apodization>, // apodization function
    medium: Arc<dyn crate::medium::Medium>, // reference to medium for sound speed
}

impl FocusedTransducer {
    pub fn new(
        x0: f64,
        y0: f64,
        z0: f64,
        focus_x: f64,
        focus_y: f64,
        focus_z: f64,
        radius: f64,
        signal: Box<dyn Signal>,
        apodization: Box<dyn Apodization>,
        medium: Arc<dyn crate::medium::Medium>,
    ) -> Self {
        FocusedTransducer {
            x0,
            y0,
            z0,
            focus_x,
            focus_y,
            focus_z,
            radius,
            signal,
            apodization,
            medium,
        }
    }
    
    // Calculate geometric focus delay for a given point
    fn calculate_delay(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let dx = x - self.x0;
        let dy = y - self.y0;
        let dz = z - self.z0;
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        
        let fx = self.focus_x - self.x0;
        let fy = self.focus_y - self.y0;
        let fz = self.focus_z - self.z0;
        let focal_length = (fx * fx + fy * fy + fz * fz).sqrt();
        
        // Calculate delay based on path length difference and local sound speed
        let sound_speed = self.medium.sound_speed(x, y, z, grid);
        (focal_length - r) / sound_speed
    }
    
    // Check if a point lies within the transducer surface
    fn is_on_surface(&self, x: f64, y: f64, z: f64) -> bool {
        // Project point onto transducer plane
        let dx = x - self.x0;
        let dy = y - self.y0;
        let dz = z - self.z0;
        
        // Calculate focal direction vector
        let fx = self.focus_x - self.x0;
        let fy = self.focus_y - self.y0;
        let fz = self.focus_z - self.z0;
        let focal_length = (fx * fx + fy * fy + fz * fz).sqrt();
        let normal_x = fx / focal_length;
        let normal_y = fy / focal_length;
        let normal_z = fz / focal_length;
        
        // Calculate distance from point to transducer plane
        let dist_to_plane = dx * normal_x + dy * normal_y + dz * normal_z;
        
        // Calculate radial distance from focal axis
        let radial_dist = ((dx - dist_to_plane * normal_x).powi(2) +
                          (dy - dist_to_plane * normal_y).powi(2) +
                          (dz - dist_to_plane * normal_z).powi(2)).sqrt();
        
        dist_to_plane.abs() < 1e-6 && radial_dist <= self.radius
    }
}

use crate::medium::homogeneous::HomogeneousMedium;

impl Source for FocusedTransducer {
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if !self.is_on_surface(x, y, z) {
            return 0.0;
        }
        
        // Calculate radial position for apodization
        let dx = x - self.x0;
        let dy = y - self.y0;
        let dz = z - self.z0;
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        
        // Get local sound speed for delay calculation
        let local_speed = self.medium.sound_speed(x, y, z, grid);
        
        // Calculate delay and apply focusing
        let delay = self.calculate_delay(x, y, z, grid);
        let delayed_t = t - delay;
        
        // Apply apodization
        let apodization = self.apodization.get_apodization(r / self.radius);
        
        // Get signal value with delay and apodization
        self.signal.value(delayed_t) * apodization
    }
    
    fn positions(&self) -> Vec<(f64, f64, f64)> {
        // Return a discretized set of points on the transducer surface
        // This is a simplified implementation - in practice, you'd want to
        // generate points based on the grid resolution
        let mut positions = Vec::new();
        let num_points = 100; // Number of points to approximate the surface
        
        for i in 0..num_points {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (num_points as f64);
            let x = self.x0;
            let y = self.y0 + self.radius * theta.cos();
            let z = self.z0 + self.radius * theta.sin();
            positions.push((x, y, z));
        }
        
        positions
    }
    
    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::sine_wave::SineWave;
    use crate::source::apodization::HanningApodization;
    
    #[test]
    fn test_focused_transducer_creation() {
        let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001);
        let medium = Arc::new(HomogeneousMedium::new(
            1000.0,  // density
            1500.0,  // sound speed
            &grid,
            0.0,     // absorption coefficient
            0.0,     // scattering coefficient
        ));
        let signal = Box::new(SineWave::new(1.0e6, 1.0e5, 0.0));
        let apodization = Box::new(HanningApodization);
        
        let transducer = FocusedTransducer::new(
            0.0, 0.0, 0.0,  // position
            0.03, 0.0, 0.0, // focus point
            0.01,           // radius
            signal,
            apodization,
            medium,
        );
        
        assert!((transducer.focus_x - 0.03).abs() < 1e-10);
        assert!(transducer.radius - 0.01 < 1e-10);
    }
    
    #[test]
    fn test_source_term_focusing() {
        let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001);
        let medium = Arc::new(HomogeneousMedium::new(
            1000.0,  // density
            1500.0,  // sound speed
            &grid,
            0.0,     // absorption coefficient
            0.0,     // scattering coefficient
        ));
        let signal = Box::new(SineWave::new(1.0e6, 1.0e5, 0.0));
        let apodization = Box::new(HanningApodization);
        
        let transducer = FocusedTransducer::new(
            0.0, 0.0, 0.0,  // position
            0.03, 0.0, 0.0, // focus point
            0.01,           // radius
            signal,
            apodization,
            medium,
        );
        
        // Check that source term is zero outside transducer surface
        assert_eq!(transducer.get_source_term(0.0, 0.02, 0.02, 0.02, &grid), 0.0);
        
        // Check that source term is non-zero on transducer surface
        let term = transducer.get_source_term(0.0, 0.0, 0.01, 0.0, &grid);
        assert!(term != 0.0);
    }
}
