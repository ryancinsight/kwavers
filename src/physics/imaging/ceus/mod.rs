//! Contrast-Enhanced Ultrasound (CEUS) Simulation Framework
//!
//! Implements microbubble dynamics, nonlinear scattering, and perfusion modeling
//! for contrast-enhanced ultrasound imaging.
//!
//! ## Physics Overview
//!
//! CEUS utilizes gas-filled microbubbles as contrast agents that resonate nonlinearly
//! under ultrasound excitation, providing enhanced sensitivity to blood flow and
//! tissue perfusion.
//!
//! ## Key Components
//!
//! - **Microbubble Dynamics**: Radial oscillations, shell viscoelasticity
//! - **Nonlinear Scattering**: Harmonic generation, subharmonic emission
//! - **Perfusion Modeling**: Blood flow kinetics, tissue uptake
//! - **Imaging Reconstruction**: Nonlinear beamforming, perfusion maps
//!
//! ## Clinical Applications
//!
//! - Liver lesion characterization
//! - Myocardial perfusion imaging
//! - Tumor vascularity assessment
//! - Renal blood flow analysis

pub mod microbubble;
pub mod scattering;
pub mod perfusion;
pub mod reconstruction;

pub use microbubble::{Microbubble, MicrobubblePopulation, BubbleDynamics};
pub use scattering::{NonlinearScattering, HarmonicImaging};
pub use perfusion::{PerfusionModel, FlowKinetics, TissueUptake};
pub use reconstruction::{CEUSReconstruction, ContrastImage};

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array3;

/// Contrast-Enhanced Ultrasound simulation workflow
#[derive(Debug)]
pub struct ContrastEnhancedUltrasound {
    /// Microbubble population
    microbubbles: MicrobubblePopulation,
    /// Nonlinear scattering model
    scattering: NonlinearScattering,
    /// Perfusion kinetics model
    perfusion: PerfusionModel,
    /// Image reconstruction
    reconstruction: CEUSReconstruction,
    /// Computational grid
    grid: Grid,
}

impl ContrastEnhancedUltrasound {
    /// Create new CEUS simulation workflow
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `medium` - Tissue medium properties
    /// * `bubble_concentration` - Initial microbubble concentration (bubbles/mL)
    /// * `bubble_size` - Mean bubble diameter (μm)
    ///
    /// # Returns
    ///
    /// Configured CEUS simulation ready for imaging
    pub fn new(
        grid: &Grid,
        medium: &dyn Medium,
        bubble_concentration: f64,
        bubble_size: f64,
    ) -> KwaversResult<Self> {
        let microbubbles = MicrobubblePopulation::new(bubble_concentration, bubble_size)?;
        let scattering = NonlinearScattering::new()?;
        let perfusion = PerfusionModel::new(grid, medium)?;
        let reconstruction = CEUSReconstruction::new(grid)?;

        Ok(Self {
            microbubbles,
            scattering,
            perfusion,
            reconstruction,
            grid: grid.clone(),
        })
    }

    /// Simulate contrast agent injection and imaging sequence
    ///
    /// # Arguments
    ///
    /// * `injection_rate` - Bolus injection rate (mL/s)
    /// * `total_time` - Total imaging time (s)
    /// * `frame_rate` - Imaging frame rate (Hz)
    /// * `acoustic_pressure` - Peak acoustic pressure (Pa)
    /// * `frequency` - Transmit frequency (Hz)
    ///
    /// # Returns
    ///
    /// Time series of contrast-enhanced ultrasound images
    pub fn simulate_imaging_sequence(
        &mut self,
        injection_rate: f64,
        total_time: f64,
        frame_rate: f64,
        acoustic_pressure: f64,
        frequency: f64,
    ) -> KwaversResult<Vec<ContrastImage>> {
        let n_frames = (total_time * frame_rate) as usize;
        let dt = 1.0 / frame_rate;

        let mut images = Vec::with_capacity(n_frames);

        // Bolus injection profile (gamma variate fit)
        let bolus_profile = self.create_bolus_profile(injection_rate, total_time, n_frames);

        for (frame, &current_concentration) in bolus_profile.iter().enumerate().take(n_frames) {
            let time = frame as f64 * dt;

            // Update microbubble distribution
            self.perfusion.update_concentration(current_concentration, dt)?;

            // Simulate acoustic excitation
            let scattered_signals = self.simulate_acoustic_response(
                acoustic_pressure,
                frequency,
                time,
            )?;

            // Reconstruct contrast image
            let contrast_image = self.reconstruction.process_frame(&scattered_signals)?;

            images.push(contrast_image);
        }

        Ok(images)
    }

    /// Simulate acoustic response of microbubbles
    fn simulate_acoustic_response(
        &self,
        acoustic_pressure: f64,
        frequency: f64,
        time: f64,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut scattered_signals = Array3::zeros((nx, ny, nz));

        // Get local microbubble concentration and acoustic field
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let _x = i as f64 * self.grid.dx;
                    let _y = j as f64 * self.grid.dy;
                    let _z = k as f64 * self.grid.dz;

                    // Local concentration from perfusion model
                    let concentration = self.perfusion.concentration(i, j, k);

                    // Acoustic pressure at this location (simplified)
                    let local_pressure = acoustic_pressure *
                        (2.0 * std::f64::consts::PI * frequency * time).cos();

                    // Microbubble scattering response
                    let scattering_response = self.scattering.compute_scattering(
                        &self.microbubbles,
                        concentration,
                        local_pressure,
                        frequency,
                    )?;

                    scattered_signals[[i, j, k]] = scattering_response;
                }
            }
        }

        Ok(scattered_signals)
    }

    /// Create bolus injection profile using gamma variate model
    fn create_bolus_profile(
        &self,
        injection_rate: f64,
        total_time: f64,
        n_frames: usize,
    ) -> Vec<f64> {
        // Gamma variate bolus profile: C(t) = A * (t/τ)^α * exp(-(t-τ)/β)
        // Typical parameters for contrast agents
        let alpha = 3.0;  // Shape parameter
        let beta = 1.5;   // Scale parameter (s)
        let tau = 0.5;    // Time to peak (s)

        let amplitude = injection_rate * 1000.0; // Convert to concentration units
        let dt = total_time / n_frames as f64;

        (0..n_frames)
            .map(|i| {
                let t = i as f64 * dt;
                if t > 0.0 {
                    amplitude * (t / tau).powf(alpha) * (-(t - tau) / beta).exp()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Generate perfusion map from time-intensity curves
    ///
    /// # Arguments
    ///
    /// * `images` - Time series of contrast images
    /// * `start_frame` - Start frame for analysis
    /// * `end_frame` - End frame for analysis
    ///
    /// # Returns
    ///
    /// Perfusion parameters: peak intensity, time-to-peak, area-under-curve
    pub fn compute_perfusion_map(
        &self,
        images: &[ContrastImage],
        start_frame: usize,
        end_frame: usize,
    ) -> KwaversResult<PerfusionMap> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut peak_intensity = Array3::zeros((nx, ny, nz));
        let mut time_to_peak = Array3::zeros((nx, ny, nz));
        let mut area_under_curve = Array3::zeros((nx, ny, nz));

        let analysis_frames = &images[start_frame..end_frame.min(images.len())];
        let frame_rate = 10.0; // Hz (typical for CEUS)

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Extract time-intensity curve for this voxel
                    let tic: Vec<f64> = analysis_frames.iter()
                        .map(|img| img.intensity[[i, j, k]])
                        .collect();

                    if let Some((peak_idx, peak_value)) = find_peak(&tic) {
                        peak_intensity[[i, j, k]] = peak_value;
                        time_to_peak[[i, j, k]] = peak_idx as f64 / frame_rate;

                        // Compute area under curve (trapezoidal integration)
                        area_under_curve[[i, j, k]] = compute_auc(&tic, 1.0 / frame_rate);
                    }
                }
            }
        }

        Ok(PerfusionMap {
            peak_intensity,
            time_to_peak,
            area_under_curve,
        })
    }

    /// Get current microbubble concentration field
    #[must_use]
    pub fn concentration_field(&self) -> &Array3<f64> {
        self.perfusion.concentration_field()
    }

    /// Get imaging parameters
    #[must_use]
    pub fn imaging_parameters(&self) -> &CEUSImagingParameters {
        &self.reconstruction.parameters
    }
}

/// Perfusion map containing quantitative perfusion parameters
#[derive(Debug, Clone)]
pub struct PerfusionMap {
    /// Peak intensity (dB)
    pub peak_intensity: Array3<f64>,
    /// Time to peak intensity (s)
    pub time_to_peak: Array3<f64>,
    /// Area under the time-intensity curve (dB·s)
    pub area_under_curve: Array3<f64>,
}

impl PerfusionMap {
    /// Get perfusion statistics for a region of interest
    ///
    /// # Arguments
    ///
    /// * `x_range` - X-coordinate range (inclusive)
    /// * `y_range` - Y-coordinate range (inclusive)
    /// * `z_range` - Z-coordinate range (inclusive)
    pub fn roi_statistics(
        &self,
        x_range: (usize, usize),
        y_range: (usize, usize),
        z_range: (usize, usize),
    ) -> PerfusionStatistics {
        let mut peak_values = Vec::new();
        let mut ttp_values = Vec::new();
        let mut auc_values = Vec::new();

        for i in x_range.0..=x_range.1 {
            for j in y_range.0..=y_range.1 {
                for k in z_range.0..=z_range.1 {
                    if self.peak_intensity[[i, j, k]] > 0.0 {
                        peak_values.push(self.peak_intensity[[i, j, k]]);
                        ttp_values.push(self.time_to_peak[[i, j, k]]);
                        auc_values.push(self.area_under_curve[[i, j, k]]);
                    }
                }
            }
        }

        PerfusionStatistics::from_samples(&peak_values, &ttp_values, &auc_values)
    }
}

/// Perfusion statistics for a region of interest
#[derive(Debug, Clone)]
pub struct PerfusionStatistics {
    /// Mean peak intensity (dB)
    pub mean_peak: f64,
    /// Standard deviation of peak intensity
    pub std_peak: f64,
    /// Mean time to peak (s)
    pub mean_ttp: f64,
    /// Standard deviation of time to peak
    pub std_ttp: f64,
    /// Mean area under curve (dB·s)
    pub mean_auc: f64,
    /// Standard deviation of area under curve
    pub std_auc: f64,
}

impl PerfusionStatistics {
    fn from_samples(peaks: &[f64], ttp: &[f64], auc: &[f64]) -> Self {
        let n = peaks.len() as f64;

        let mean_peak = peaks.iter().sum::<f64>() / n;
        let std_peak = (peaks.iter().map(|x| (x - mean_peak).powi(2)).sum::<f64>() / n).sqrt();

        let mean_ttp = ttp.iter().sum::<f64>() / n;
        let std_ttp = (ttp.iter().map(|x| (x - mean_ttp).powi(2)).sum::<f64>() / n).sqrt();

        let mean_auc = auc.iter().sum::<f64>() / n;
        let std_auc = (auc.iter().map(|x| (x - mean_auc).powi(2)).sum::<f64>() / n).sqrt();

        Self {
            mean_peak,
            std_peak,
            mean_ttp,
            std_ttp,
            mean_auc,
            std_auc,
        }
    }
}

/// CEUS imaging parameters
#[derive(Debug, Clone)]
pub struct CEUSImagingParameters {
    /// Transmit frequency (Hz)
    pub frequency: f64,
    /// Mechanical index
    pub mechanical_index: f64,
    /// Frame rate (Hz)
    pub frame_rate: f64,
    /// Dynamic range (dB)
    pub dynamic_range: f64,
    /// Field of view (mm)
    pub fov: (f64, f64),
    /// Imaging depth (mm)
    pub depth: f64,
}

impl Default for CEUSImagingParameters {
    fn default() -> Self {
        Self {
            frequency: 3.0e6,        // 3 MHz
            mechanical_index: 0.1,   // Low MI for CEUS
            frame_rate: 10.0,        // 10 fps
            dynamic_range: 60.0,     // 60 dB
            fov: (80.0, 60.0),       // 80x60 mm
            depth: 150.0,            // 150 mm
        }
    }
}

// Helper functions

/// Find peak in time-intensity curve
fn find_peak(tic: &[f64]) -> Option<(usize, f64)> {
    tic.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, &val)| (idx, val))
}

/// Compute area under curve using trapezoidal rule
fn compute_auc(tic: &[f64], dt: f64) -> f64 {
    let mut auc = 0.0;
    for i in 0..tic.len().saturating_sub(1) {
        auc += (tic[i] + tic[i + 1]) * dt / 2.0;
    }
    auc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::homogeneous::HomogeneousMedium;

    #[test]
    fn test_ceus_initialization() {
        let grid = Grid::new(50, 50, 30, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid);

        let mut ceus = ContrastEnhancedUltrasound::new(
            &grid,
            &medium,
            1e6,   // 1 million bubbles/mL
            2.5,   // 2.5 μm diameter
        ).unwrap();

        // Initially concentration should be zero, but after simulation it should have values
        assert!(ceus.concentration_field().iter().all(|&x| x >= 0.0));

        // Test a short simulation
        let images = ceus.simulate_imaging_sequence(
            1.0, 1.0, 1.0, 1e5, 2e6,
        ).unwrap();
        assert_eq!(images.len(), 1);
    }

    #[test]
    fn test_imaging_sequence() {
        let grid = Grid::new(32, 32, 20, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid);

        let mut ceus = ContrastEnhancedUltrasound::new(
            &grid,
            &medium,
            5e5,   // 500k bubbles/mL
            2.0,   // 2.0 μm diameter
        ).unwrap();

        let images = ceus.simulate_imaging_sequence(
            1.0,      // 1 mL/s injection rate
            2.0,      // 2 seconds total
            5.0,      // 5 fps
            100_000.0, // 100 kPa acoustic pressure
            3e6,      // 3 MHz
        ).unwrap();

        assert_eq!(images.len(), 10); // 2s * 5fps = 10 frames
        assert!(images.iter().all(|img| img.intensity.iter().all(|&x| x >= 0.0)));
    }

    #[test]
    fn test_perfusion_analysis() {
        let grid = Grid::new(20, 20, 10, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.0, &grid);

        let ceus = ContrastEnhancedUltrasound::new(
            &grid,
            &medium,
            1e6,
            2.5,
        ).unwrap();

        // Create mock images with increasing intensity
        let mut images = Vec::new();
        for i in 0..20 {
            let mut intensity = Array3::zeros((20, 20, 10));
            intensity.fill(i as f64 * 0.1); // Linear increase
            images.push(ContrastImage { intensity });
        }

        let perfusion_map = ceus.compute_perfusion_map(&images, 0, 20).unwrap();

        // Check that perfusion parameters are computed
        assert!(perfusion_map.peak_intensity.iter().any(|&x| x > 0.0));
        assert!(perfusion_map.time_to_peak.iter().any(|&x| x >= 0.0));
        assert!(perfusion_map.area_under_curve.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn test_roi_statistics() {
        let perfusion_map = PerfusionMap {
            peak_intensity: Array3::from_elem((10, 10, 5), 10.0),
            time_to_peak: Array3::from_elem((10, 10, 5), 15.0),
            area_under_curve: Array3::from_elem((10, 10, 5), 100.0),
        };

        let stats = perfusion_map.roi_statistics((0, 4), (0, 4), (0, 2));

        assert!((stats.mean_peak - 10.0).abs() < 1e-6);
        assert!((stats.mean_ttp - 15.0).abs() < 1e-6);
        assert!((stats.mean_auc - 100.0).abs() < 1e-6);
    }
}
