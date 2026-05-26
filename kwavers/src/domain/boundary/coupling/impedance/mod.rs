//! Impedance boundary condition
//!
//! Frequency-dependent absorption based on acoustic impedance matching.
//! Particularly useful for ultrasound transducers and tissue interfaces.

use crate::core::constants::fundamental::ACOUSTIC_IMPEDANCE_WATER_NOMINAL;
use crate::core::error::KwaversResult;
use crate::domain::boundary::traits::BoundaryCondition;
use crate::domain::grid::GridTopology;
use ndarray::ArrayViewMut3;

use super::types::{BoundaryDirections, FrequencyProfile};

#[cfg(test)]
mod tests;

/// Impedance boundary condition
///
/// Implements frequency-dependent absorption based on acoustic impedance matching.
/// This boundary condition is particularly useful for modeling:
///
/// - Ultrasound transducer surfaces
/// - Tissue-air interfaces
/// - Coupling layers and matching layers
/// - Frequency-selective absorption
///
/// # Physics
///
/// The reflection coefficient at an impedance boundary is given by:
///
/// ```text
/// R = (Z_target - Z_medium) / (Z_target + Z_medium)
/// ```
///
/// where:
/// - Z_target is the target impedance of the boundary
/// - Z_medium is the acoustic impedance of the propagating medium
///
/// The boundary can apply frequency-dependent profiles to model realistic
/// transducer responses or tissue frequency-dependent behavior.
///
/// # Example
///
/// ```no_run
/// use kwavers::domain::boundary::coupling::ImpedanceBoundary;
/// use kwavers::domain::boundary::traits::BoundaryDirections;
///
/// // Create impedance boundary matching water-tissue interface
/// let boundary = ImpedanceBoundary::new(
///     1.5e6,  // Target impedance (1.5 MRayl)
///     BoundaryDirections::all(),
/// );
///
/// // Add Gaussian frequency profile centered at 1 MHz with 0.5 MHz bandwidth
/// let boundary = boundary.with_gaussian_profile(1e6, 0.5e6);
/// ```
#[derive(Debug, Clone)]
pub struct ImpedanceBoundary {
    /// Target impedance Z_target (kg/m²s)
    pub target_impedance: f64,
    /// Medium impedance Z_medium (kg/m²s) — defaults to water.
    pub medium_impedance: f64,
    /// Representative frequency for spatial-domain reflection coefficient (Hz).
    /// Defaults to the Gaussian center frequency when set, else 1 MHz.
    pub representative_frequency_hz: f64,
    /// Frequency-dependent profile
    pub frequency_profile: FrequencyProfile,
    /// Boundary directions
    pub directions: BoundaryDirections,
}

impl ImpedanceBoundary {
    /// Create a new impedance boundary
    ///
    /// # Arguments
    ///
    /// * `target_impedance` - Target acoustic impedance in kg/(m²·s) or Rayl
    /// * `directions` - Directions in which to apply the boundary condition
    ///
    /// # Returns
    ///
    /// New `ImpedanceBoundary` with flat frequency response
    #[must_use]
    pub fn new(target_impedance: f64, directions: BoundaryDirections) -> Self {
        Self {
            target_impedance,
            medium_impedance: ACOUSTIC_IMPEDANCE_WATER_NOMINAL,
            representative_frequency_hz: 1.0e6,
            frequency_profile: FrequencyProfile::Flat,
            directions,
        }
    }

    /// Set the medium impedance Z_medium for reflection-coefficient computation.
    ///
    /// Defaults to `ACOUSTIC_IMPEDANCE_WATER_NOMINAL` (1.5e6 Rayl) when not set.
    #[must_use]
    pub fn with_medium_impedance(mut self, medium_impedance: f64) -> Self {
        self.medium_impedance = medium_impedance;
        self
    }

    /// Set Gaussian frequency profile
    ///
    /// # Arguments
    ///
    /// * `center_freq` - Center frequency in Hz
    /// * `bandwidth` - Bandwidth (FWHM) in Hz
    ///
    /// # Returns
    ///
    /// Self with Gaussian profile applied; representative frequency is set to `center_freq`.
    #[must_use]
    pub fn with_gaussian_profile(mut self, center_freq: f64, bandwidth: f64) -> Self {
        self.frequency_profile = FrequencyProfile::Gaussian {
            center_freq,
            bandwidth,
        };
        self.representative_frequency_hz = center_freq;
        self
    }

    /// Compute impedance ratio at given frequency
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz
    /// * `medium_impedance` - Impedance of the propagating medium in kg/(m²·s)
    ///
    /// # Returns
    ///
    /// Frequency-weighted impedance ratio Z_target / Z_medium
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[must_use]
    pub fn impedance_ratio(&self, frequency: f64, medium_impedance: f64) -> f64 {
        let z_ratio = self.target_impedance / medium_impedance;

        match &self.frequency_profile {
            FrequencyProfile::Flat => z_ratio,
            FrequencyProfile::Gaussian {
                center_freq,
                bandwidth,
            } => {
                let sigma = bandwidth / (2.0 * (2.0 * std::f64::consts::LN_2).sqrt()); // Convert FWHM to sigma
                let gaussian = (-0.5 * ((frequency - center_freq) / sigma).powi(2)).exp();
                z_ratio * gaussian
            }
            FrequencyProfile::Custom(pairs) => {
                // Simple linear interpolation
                if pairs.is_empty() {
                    z_ratio
                } else if frequency <= pairs[0].0 {
                    pairs[0].1 * z_ratio
                } else if frequency >= pairs.last().unwrap().0 {
                    pairs.last().unwrap().1 * z_ratio
                } else {
                    // Find interval and interpolate
                    for i in 0..pairs.len() - 1 {
                        if frequency >= pairs[i].0 && frequency <= pairs[i + 1].0 {
                            let f1 = pairs[i].0;
                            let f2 = pairs[i + 1].0;
                            let z1 = pairs[i].1;
                            let z2 = pairs[i + 1].1;

                            let ratio = z1 + (z2 - z1) * (frequency - f1) / (f2 - f1);
                            return ratio * z_ratio;
                        }
                    }
                    z_ratio
                }
            }
        }
    }

    /// Compute reflection coefficient from impedance mismatch
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz
    /// * `medium_impedance` - Impedance of the propagating medium in kg/(m²·s)
    ///
    /// # Returns
    ///
    /// Reflection coefficient R = (Z_target - Z_medium) / (Z_target + Z_medium)
    #[must_use]
    pub fn reflection_coefficient(&self, frequency: f64, medium_impedance: f64) -> f64 {
        let z_ratio = self.impedance_ratio(frequency, medium_impedance);
        (z_ratio - 1.0) / (z_ratio + 1.0)
    }
}

impl BoundaryCondition for ImpedanceBoundary {
    fn name(&self) -> &str {
        "ImpedanceBoundary"
    }

    fn active_directions(&self) -> BoundaryDirections {
        self.directions
    }

    fn apply_scalar_spatial(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Ghost-cell impedance boundary:
        //   field[boundary_cell] = R · field[interior_neighbour]
        // where R = (Z_target - Z_medium) / (Z_target + Z_medium) is the
        // pressure reflection coefficient (Pierce 1989, §3.6). R = +1 is
        // perfectly rigid (mirror), R = 0 is matched (perfect absorber),
        // R = -1 is pressure-release (free surface).
        //
        // For Flat frequency profile the coefficient is exact and frequency-
        // independent. For Gaussian/Custom profiles the coefficient is
        // evaluated at `representative_frequency_hz` (the spectral peak).
        let r = self.reflection_coefficient(
            self.representative_frequency_hz,
            self.medium_impedance,
        );
        let dims = grid.dimensions();
        let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
        if nx < 2 || ny < 2 || nz < 2 {
            return Ok(());
        }

        if self.directions.x_min {
            for j in 0..ny {
                for k in 0..nz {
                    field[[0, j, k]] = r * field[[1, j, k]];
                }
            }
        }
        if self.directions.x_max {
            for j in 0..ny {
                for k in 0..nz {
                    field[[nx - 1, j, k]] = r * field[[nx - 2, j, k]];
                }
            }
        }
        if self.directions.y_min {
            for i in 0..nx {
                for k in 0..nz {
                    field[[i, 0, k]] = r * field[[i, 1, k]];
                }
            }
        }
        if self.directions.y_max {
            for i in 0..nx {
                for k in 0..nz {
                    field[[i, ny - 1, k]] = r * field[[i, ny - 2, k]];
                }
            }
        }
        if self.directions.z_min {
            for i in 0..nx {
                for j in 0..ny {
                    field[[i, j, 0]] = r * field[[i, j, 1]];
                }
            }
        }
        if self.directions.z_max {
            for i in 0..nx {
                for j in 0..ny {
                    field[[i, j, nz - 1]] = r * field[[i, j, nz - 2]];
                }
            }
        }
        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        field: &mut ndarray::Array3<num_complex::Complex<f64>>,
        grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Frequency-domain ghost-cell impedance boundary.
        //
        // For a pseudospectral solver the field is the complex spatial-domain
        // representation; the reflection coefficient is applied identically to
        // the time-domain case at boundary cells using the representative
        // frequency. Frequency-bin-dependent profiles require k → ω mapping
        // (k = ω/c) which depends on the medium's sound speed and is therefore
        // applied at the model level where c is known, not here.
        let r = self.reflection_coefficient(
            self.representative_frequency_hz,
            self.medium_impedance,
        );
        let r_complex = num_complex::Complex::new(r, 0.0);
        let dims = grid.dimensions();
        let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
        if nx < 2 || ny < 2 || nz < 2 {
            return Ok(());
        }

        if self.directions.x_min {
            for j in 0..ny {
                for k in 0..nz {
                    field[[0, j, k]] = r_complex * field[[1, j, k]];
                }
            }
        }
        if self.directions.x_max {
            for j in 0..ny {
                for k in 0..nz {
                    field[[nx - 1, j, k]] = r_complex * field[[nx - 2, j, k]];
                }
            }
        }
        if self.directions.y_min {
            for i in 0..nx {
                for k in 0..nz {
                    field[[i, 0, k]] = r_complex * field[[i, 1, k]];
                }
            }
        }
        if self.directions.y_max {
            for i in 0..nx {
                for k in 0..nz {
                    field[[i, ny - 1, k]] = r_complex * field[[i, ny - 2, k]];
                }
            }
        }
        if self.directions.z_min {
            for i in 0..nx {
                for j in 0..ny {
                    field[[i, j, 0]] = r_complex * field[[i, j, 1]];
                }
            }
        }
        if self.directions.z_max {
            for i in 0..nx {
                for j in 0..ny {
                    field[[i, j, nz - 1]] = r_complex * field[[i, j, nz - 2]];
                }
            }
        }
        Ok(())
    }

    fn reset(&mut self) {}
}

