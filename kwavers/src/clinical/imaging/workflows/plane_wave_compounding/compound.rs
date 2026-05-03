//! `PlaneWaveCompound` — multi-angle insonification with coherent compounding.
//!
//! ## Physics Model
//!
//! Plane wave field: p(r,t) = P₀ cos(ωt − k·r)
//!
//! Coherent compound intensity:
//! ```text
//! I(r) = |Σᵢ Aᵢ(r) exp(jφᵢ(r))|²
//! ```
//! SNR improvement ∝ √N_angles (Montaldo et al. 2009).
//!
//! ## References
//! - Montaldo et al. (2009): "Coherent plane-wave compounding for very high frame rate."
//!   *IEEE UFFC*, 56(3), 489–506.

use super::config::PlaneWaveConfig;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array2;
use num_complex::Complex;
use std::f64::consts::PI;

/// Plane wave compounding processor.
#[derive(Debug, Clone)]
pub struct PlaneWaveCompound {
    pub(super) config: PlaneWaveConfig,
    /// Plane wave angles (radians).
    pub(super) angles: Vec<f64>,
    /// Beamformed complex images per angle.
    pub(super) angle_images: Vec<Array2<Complex<f64>>>,
    /// Coherently compounded complex image.
    pub(super) compounded_image: Array2<Complex<f64>>,
    /// Log-compressed display image (normalized to [0,1]).
    pub(super) display_image: Array2<f64>,
    /// Number of lateral image columns.
    pub(super) num_lateral: usize,
    /// Number of axial image rows.
    pub(super) num_axial: usize,
    _wavelength: f64,
    pub(super) wavenumber: f64,
    _omega: f64,
}

impl PlaneWaveCompound {
    /// Construct and validate a plane wave compounding processor.
    pub fn new(config: PlaneWaveConfig) -> KwaversResult<Self> {
        if config.num_angles == 0 {
            return Err(KwaversError::InvalidInput(
                "num_angles must be > 0".to_string(),
            ));
        }
        if config.sound_speed <= 0.0 || config.frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "sound_speed and frequency must be positive".to_string(),
            ));
        }
        if config.depth <= 0.0 || config.axial_step <= 0.0 || config.lateral_step <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "depth and steps must be positive".to_string(),
            ));
        }

        let wavelength = config.sound_speed / config.frequency;
        let wavenumber = 2.0 * PI / wavelength;
        let omega = 2.0 * PI * config.frequency;

        let num_axial = (config.depth / config.axial_step).ceil() as usize;
        let num_lateral = (config.aperture_size / config.lateral_step).ceil() as usize;

        let mut angles = Vec::with_capacity(config.num_angles);
        if config.num_angles == 1 {
            angles.push(0.0);
        } else {
            for i in 0..config.num_angles {
                let angle_deg = -config.angle_range
                    + (2.0 * config.angle_range) * i as f64 / (config.num_angles - 1) as f64;
                angles.push(angle_deg.to_radians());
            }
        }

        let angle_images = vec![Array2::zeros((num_axial, num_lateral)); config.num_angles];

        Ok(Self {
            config,
            angles,
            angle_images,
            compounded_image: Array2::zeros((num_axial, num_lateral)),
            display_image: Array2::zeros((num_axial, num_lateral)),
            num_lateral,
            num_axial,
            _wavelength: wavelength,
            wavenumber,
            _omega: omega,
        })
    }

    /// Generate a monochromatic plane wave pressure field at `angle_idx`.
    ///
    /// Field: p(x,z) = A(x) exp(j·k·x·sin(θ)) · exp(j·k·z)
    /// where A(x) is the aperture apodization.
    #[allow(dead_code)]
    pub(super) fn generate_plane_wave(
        &self,
        angle_idx: usize,
    ) -> KwaversResult<Array2<Complex<f64>>> {
        if angle_idx >= self.config.num_angles {
            return Err(KwaversError::InvalidInput(format!(
                "angle_idx {} out of range",
                angle_idx
            )));
        }

        let angle = self.angles[angle_idx];
        let mut field = Array2::zeros((self.num_axial, self.num_lateral));
        let apod = self.compute_apodization();

        for (idx, elem) in field.indexed_iter_mut() {
            let (axial_idx, lateral_idx) = idx;
            let x = lateral_idx as f64 * self.config.lateral_step;
            let z = axial_idx as f64 * self.config.axial_step;

            let phase = self.wavenumber * x * angle.sin();
            let amplitude = if lateral_idx < apod.len() {
                apod[lateral_idx]
            } else {
                0.0
            };

            *elem = Complex::new(amplitude * phase.cos(), amplitude * phase.sin());

            if z > 0.0 {
                let prop_phase = self.wavenumber * z;
                let prop = Complex::new(prop_phase.cos(), prop_phase.sin());
                *elem *= prop;
            }
        }

        Ok(field)
    }

    /// Compute normalized aperture apodization weights.
    ///
    /// Supported window types: `"hann"`, `"hamming"`, `"blackman"`, `"rect"` (default).
    #[allow(dead_code)]
    pub(super) fn compute_apodization(&self) -> Vec<f64> {
        let n = self.config.num_elements;
        let mut apod = vec![0.0_f64; n];

        match self.config.apodization.as_str() {
            "hann" => {
                for (i, w) in apod.iter_mut().enumerate() {
                    *w = 0.5 - 0.5 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos();
                }
            }
            "hamming" => {
                for (i, w) in apod.iter_mut().enumerate() {
                    *w = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos();
                }
            }
            "blackman" => {
                for (i, w) in apod.iter_mut().enumerate() {
                    let nn = i as f64 / (n as f64 - 1.0);
                    *w = 0.42 - 0.5 * (2.0 * PI * nn).cos() + 0.08 * (4.0 * PI * nn).cos();
                }
            }
            _ => {
                // "rect" or any other — uniform window
                for w in apod.iter_mut() {
                    *w = 1.0;
                }
            }
        }

        let max_w = apod.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_w > 0.0 {
            for w in &mut apod {
                *w = (*w / max_w).clamp(0.0, 1.0);
            }
        }

        apod
    }

    /// Delay-and-sum beamform `received_field` for the specified plane wave angle.
    ///
    /// Phase correction: −k x sin(θ) reverses transmit steering; −k z removes depth phase.
    pub(super) fn beamform_angle(
        &self,
        angle_idx: usize,
        received_field: &Array2<Complex<f64>>,
    ) -> KwaversResult<Array2<Complex<f64>>> {
        let angle = self.angles[angle_idx];
        let mut beamformed = Array2::zeros((self.num_axial, self.num_lateral));

        for (idx, elem) in beamformed.indexed_iter_mut() {
            let (ai, li) = idx;
            if ai >= received_field.nrows() || li >= received_field.ncols() {
                continue;
            }

            let signal = received_field[[ai, li]];
            let z = ai as f64 * self.config.axial_step;
            let x = li as f64 * self.config.lateral_step;

            let phase_corr = -self.wavenumber * x * angle.sin();
            let corr = Complex::new(phase_corr.cos(), phase_corr.sin());

            *elem = if z > 0.0 {
                let dp = -self.wavenumber * z;
                signal * corr * Complex::new(dp.cos(), dp.sin())
            } else {
                signal * corr
            };
        }

        Ok(beamformed)
    }

    /// Coherently compound all angle images and apply log-compression.
    ///
    /// Compound: Σᵢ image_i(r), then dB = 10·log₁₀(|·|²).
    pub fn compound(&mut self) -> KwaversResult<()> {
        for elem in self.compounded_image.iter_mut() {
            *elem = Complex::new(0.0, 0.0);
        }

        for angle_idx in 0..self.config.num_angles {
            if angle_idx >= self.angle_images.len() {
                continue;
            }
            for ((i, j), &value) in self.angle_images[angle_idx].indexed_iter() {
                self.compounded_image[[i, j]] += value;
            }
        }

        for ((i, j), elem) in self.compounded_image.indexed_iter() {
            let intensity = elem.norm_sqr();
            let db = if intensity > 1e-12 {
                10.0 * intensity.log10()
            } else {
                -120.0
            };
            self.display_image[[i, j]] =
                ((db + self.config.dynamic_range) / self.config.dynamic_range).clamp(0.0, 1.0);
        }

        Ok(())
    }

    /// Beamform one received field per angle, then coherently compound.
    ///
    /// Returns the normalized log-compressed display image.
    pub fn process_frame(
        &mut self,
        received_fields: &[Array2<Complex<f64>>],
    ) -> KwaversResult<Array2<f64>> {
        if received_fields.len() != self.config.num_angles {
            return Err(KwaversError::InvalidInput(format!(
                "Expected {} received fields, got {}",
                self.config.num_angles,
                received_fields.len()
            )));
        }

        for (angle_idx, received_field) in received_fields.iter().enumerate() {
            self.angle_images[angle_idx] = self.beamform_angle(angle_idx, received_field)?;
        }

        self.compound()?;
        Ok(self.display_image.clone())
    }

    /// Returns a thermal-acoustic grid configuration consistent with this image geometry.
    ///
    /// The plane-wave image grid is two-dimensional `(lateral, axial)`. The
    /// thermal-acoustic solver is volumetric, so this method embeds the image
    /// as a one-cell-thick volume with `(nx, ny, nz) = (lateral, 1, axial)`.
    /// The time step uses the same acoustic CFL factor as the coupled solver
    /// default: `dt = 0.3 min(dx, dy, dz) / c_ref`.
    pub fn config(&self) -> crate::solver::forward::coupled::ThermalAcousticConfig {
        let mut thermal = crate::solver::forward::coupled::ThermalAcousticConfig::default();
        thermal.nx = self.num_lateral;
        thermal.ny = 1;
        thermal.nz = self.num_axial;
        thermal.dx = self.config.lateral_step;
        thermal.dy = self.config.element_spacing;
        thermal.dz = self.config.axial_step;
        thermal.c_ref = self.config.sound_speed;
        thermal.dt = 0.3 * thermal.dx.min(thermal.dy).min(thermal.dz) / thermal.c_ref;
        thermal
    }

    /// Number of configured plane wave angles.
    pub fn num_angles(&self) -> usize {
        self.config.num_angles
    }

    /// Plane wave angles in degrees.
    pub fn get_angles(&self) -> Vec<f64> {
        self.angles.iter().map(|&a| a.to_degrees()).collect()
    }

    /// Log-compressed display image (values in [0,1]).
    pub fn display_image(&self) -> &Array2<f64> {
        &self.display_image
    }

    /// Raw coherently compounded complex image.
    pub fn compounded_image(&self) -> &Array2<Complex<f64>> {
        &self.compounded_image
    }

    /// Image grid dimensions: `(num_axial, num_lateral)`.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.num_axial, self.num_lateral)
    }

    /// Estimated frame rate: `(speedup_factor, practical_fps)`.
    ///
    /// Reference: 30 fps focused beam; speedup = N_angles.
    pub fn frame_rate_estimate(&self) -> (f64, f64) {
        let focused_fps = 30.0;
        let speedup = self.config.num_angles as f64;
        (speedup, focused_fps * speedup)
    }
}
