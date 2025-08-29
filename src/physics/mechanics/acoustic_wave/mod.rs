// physics/mechanics/acoustic_wave/mod.rs
pub mod nonlinear; // This will now refer to the new subdirectory

// Re-export NonlinearWave from the new structure.
pub use nonlinear::NonlinearWave;

pub mod westervelt;
pub mod westervelt_wave;
pub use westervelt::WesterveltWave;

pub mod kuznetsov;
pub use kuznetsov::{KuznetsovConfig, KuznetsovWave};

pub mod westervelt_fdtd;
pub use westervelt_fdtd::{WesterveltFdtd, WesterveltFdtdConfig};

pub mod unified;
pub use unified::{AcousticModelType, AcousticSolverConfig, UnifiedAcousticSolver};

use crate::grid::Grid;
use crate::medium::Medium;
use std::f64::consts::PI;

/// Compute acoustic diffusivity from medium properties
///
/// This is the single source of truth for acoustic diffusivity calculation.
///
/// # Physics Background
///
/// Acoustic diffusivity δ = (4μ/3 + μ_B + κ(γ-1)/C_p) / ρ₀
/// Where:
/// - μ = shear viscosity
/// - μ_B = bulk viscosity  
/// - κ = thermal conductivity
/// - γ = specific heat ratio
/// - C_p = specific heat at constant pressure
///
/// For soft tissues, we use the approximation:
/// δ ≈ 2αc³/(ω²)
///
/// where α is the absorption coefficient and c is the sound speed.
///
/// # Safety
///
/// Returns 0.0 for zero frequency (static fields) to prevent division by zero.
/// This is physically sensible as the frequency-dependent absorption model
/// becomes ill-defined at DC.
pub fn compute_acoustic_diffusivity(
    medium: &dyn Medium,
    x: f64,
    y: f64,
    z: f64,
    grid: &Grid,
    frequency: f64,
) -> f64 {
    // Prevent division by zero for static fields (frequency = 0)
    // At zero frequency, the concept of acoustic diffusivity from
    // frequency-dependent absorption is not well-defined
    if frequency == 0.0 {
        return 0.0;
    }

    let alpha = medium.absorption_coefficient(x, y, z, grid, frequency);
    let c = medium.sound_speed(x, y, z, grid);

    // Approximate diffusivity from power-law absorption
    // δ ≈ 2αc³/(ω²) for typical soft tissues
    let omega = 2.0 * PI * frequency;
    2.0 * alpha * c.powi(3) / (omega * omega)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::homogeneous::HomogeneousMedium;

    /// Test implementation of Medium trait for heterogeneous testing
    #[derive(Debug)]
    struct HeterogeneousMediumMock {
        /// Returns different properties based on position
        position_dependent: bool,
        /// Cached density array
        density: ndarray::Array3<f64>,
        /// Cached sound speed array
        sound_speed: ndarray::Array3<f64>,
        /// Temperature array
        temperature: ndarray::Array3<f64>,
        /// Bubble radius array
        bubble_radius: ndarray::Array3<f64>,
        /// Bubble velocity array
        bubble_velocity: ndarray::Array3<f64>,
    }

    impl HeterogeneousMediumMock {
        fn new(position_dependent: bool) -> Self {
            Self {
                position_dependent,
                density: ndarray::Array3::from_elem((10, 10, 10), 1000.0),
                sound_speed: ndarray::Array3::from_elem((10, 10, 10), 1500.0),
                temperature: ndarray::Array3::from_elem((10, 10, 10), 310.0),
                bubble_radius: ndarray::Array3::zeros((10, 10, 10)),
                bubble_velocity: ndarray::Array3::zeros((10, 10, 10)),
            }
        }
    }

    // Implement component traits for HeterogeneousMediumMock
    impl crate::medium::core::CoreMedium for HeterogeneousMediumMock {
        fn density(&self, x: f64, y: f64, z: f64, _grid: &Grid) -> f64 {
            if self.position_dependent {
                1000.0 + x + y + z
            } else {
                1000.0
            }
        }

        fn sound_speed(&self, x: f64, y: f64, z: f64, _grid: &Grid) -> f64 {
            if self.position_dependent {
                1500.0 + x * 10.0 + y * 5.0 + z * 2.0
            } else {
                1500.0
            }
        }

        fn is_homogeneous(&self) -> bool {
            !self.position_dependent
        }

        fn reference_frequency(&self) -> f64 {
            1e6
        }
    }

    impl crate::medium::core::ArrayAccess for HeterogeneousMediumMock {
        fn get_density_array(&self, _grid: &Grid) -> ndarray::Array3<f64> {
            self.density.clone()
        }

        fn get_sound_speed_array(&self, _grid: &Grid) -> ndarray::Array3<f64> {
            self.sound_speed.clone()
        }

        fn density_array(&self, _grid: &Grid) -> ndarray::Array3<f64> {
            self.density.clone()
        }

        fn sound_speed_array(&self, _grid: &Grid) -> ndarray::Array3<f64> {
            self.sound_speed.clone()
        }
    }

    impl crate::medium::acoustic::AcousticProperties for HeterogeneousMediumMock {
        fn absorption_coefficient(
            &self,
            _x: f64,
            _y: f64,
            _z: f64,
            _grid: &Grid,
            _frequency: f64,
        ) -> f64 {
            0.01
        }

        fn attenuation(&self, x: f64, y: f64, z: f64, frequency: f64, grid: &Grid) -> f64 {
            // Power law attenuation: α = α₀ * f^y
            // For tissue: α₀ = 0.5 dB/cm/MHz, y = 1.1
            const ALPHA_0: f64 = 0.5; // dB/cm/MHz
            const POWER_LAW_EXPONENT: f64 = 1.1;
            const DB_TO_NP_PER_M: f64 = 100.0 / 8.686; // Convert dB/cm to Np/m

            if self.position_dependent {
                // Spatially varying attenuation based on position
                let spatial_factor = 1.0 + 0.2 * ((x / grid.dx).sin() + (y / grid.dy).cos());
                ALPHA_0
                    * (frequency / 1e6).powf(POWER_LAW_EXPONENT)
                    * DB_TO_NP_PER_M
                    * spatial_factor
            } else {
                ALPHA_0 * (frequency / 1e6).powf(POWER_LAW_EXPONENT) * DB_TO_NP_PER_M
            }
        }

        fn nonlinearity_parameter(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // B/A parameter for nonlinear acoustics
            // Water: 5.0, Tissue: 6-9, Fat: 10
            if self.position_dependent {
                let (ix, iy, iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                // Vary B/A based on position to simulate tissue heterogeneity
                let normalized_x = ix as f64 / grid.nx as f64;
                let normalized_y = iy as f64 / grid.ny as f64;
                5.0 + 2.0 * normalized_x + 1.5 * normalized_y.sin()
            } else {
                5.0 // Water value
            }
        }

        fn nonlinearity_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Same as nonlinearity_parameter (B/A)
            self.nonlinearity_parameter(x, y, z, grid)
        }

        fn acoustic_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Acoustic diffusivity δ = (4μ/3 + μ_B)/(ρc²)
            // For water at 20°C: ~1.4e-7 m²/s
            // For tissue: varies from 1e-7 to 2e-7 m²/s
            const BASE_DIFFUSIVITY: f64 = 1.4e-7;

            if self.position_dependent {
                let (ix, iy, _iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                // Spatial variation to simulate tissue heterogeneity
                let variation = 0.3 * ((ix as f64 * 0.1).sin() + (iy as f64 * 0.1).cos());
                BASE_DIFFUSIVITY * (1.0 + variation)
            } else {
                BASE_DIFFUSIVITY
            }
        }

        fn tissue_type(
            &self,
            x: f64,
            y: f64,
            z: f64,
            grid: &Grid,
        ) -> Option<crate::medium::absorption::TissueType> {
            use crate::medium::absorption::TissueType;

            if self.position_dependent {
                // Create tissue regions based on position
                let (ix, iy, iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));

                // Simple tissue segmentation based on grid regions
                if iz < grid.nz / 3 {
                    Some(TissueType::Muscle)
                } else if iz < 2 * grid.nz / 3 {
                    Some(TissueType::Fat)
                } else {
                    Some(TissueType::Liver)
                }
            } else {
                None // Homogeneous case has no specific tissue type
            }
        }
    }

    impl crate::medium::elastic::ElasticProperties for HeterogeneousMediumMock {
        fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Lamé's first parameter: λ = K - 2μ/3
            // For water: λ ≈ 2.2 GPa (bulk modulus with μ=0)
            // For tissue: varies from 2-3 GPa
            const BASE_LAMBDA: f64 = 2.2e9;

            if self.position_dependent {
                let (ix, iy, _iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                // Spatial variation for tissue heterogeneity
                let variation = 0.2 * ((ix as f64 * 0.05).sin() + (iy as f64 * 0.05).cos());
                BASE_LAMBDA * (1.0 + variation)
            } else {
                BASE_LAMBDA
            }
        }

        fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Lamé's second parameter (shear modulus)
            // For water: μ = 0 (no shear resistance)
            // For tissue: varies from 1-100 kPa
            if self.position_dependent {
                let (ix, iy, iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                // Tissue has non-zero shear modulus
                let base_mu = 10e3; // 10 kPa typical for soft tissue
                let variation = (ix as f64 / grid.nx as f64) + (iy as f64 / grid.ny as f64) * 0.5;
                base_mu * (1.0 + variation)
            } else {
                0.0 // Water has no shear resistance
            }
        }

        fn shear_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Shear wave speed: c_s = sqrt(μ/ρ)
            // For water: 0 (no shear waves)
            // For tissue: 1-10 m/s
            if self.position_dependent {
                let mu = self.lame_mu(x, y, z, grid);
                let rho = self.density(x, y, z, grid);
                if mu > 0.0 {
                    (mu / rho).sqrt()
                } else {
                    0.0
                }
            } else {
                0.0 // Water doesn't support shear waves
            }
        }

        fn compressional_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Use the CoreMedium trait method
            <Self as crate::medium::core::CoreMedium>::sound_speed(self, x, y, z, grid)
        }
    }

    impl crate::medium::elastic::ElasticArrayAccess for HeterogeneousMediumMock {
        fn lame_lambda_array(&self) -> ndarray::Array3<f64> {
            self.density.clone()
        }

        fn lame_mu_array(&self) -> ndarray::Array3<f64> {
            self.bubble_radius.clone()
        }

        fn shear_sound_speed_array(&self) -> ndarray::Array3<f64> {
            ndarray::Array3::zeros((10, 10, 10))
        }

        fn shear_viscosity_coeff_array(&self) -> ndarray::Array3<f64> {
            ndarray::Array3::from_elem((10, 10, 10), 1e-3)
        }

        fn bulk_viscosity_coeff_array(&self) -> ndarray::Array3<f64> {
            ndarray::Array3::from_elem((10, 10, 10), 2e-3)
        }
    }

    impl crate::medium::thermal::ThermalProperties for HeterogeneousMediumMock {
        fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Specific heat capacity
            // Water: 4180 J/(kg·K), Tissue: 3500-3700 J/(kg·K)
            const WATER_SPECIFIC_HEAT: f64 = 4180.0;
            const TISSUE_SPECIFIC_HEAT: f64 = 3600.0;

            if self.position_dependent {
                let (ix, iy, _iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                // Interpolate between water and tissue values
                let tissue_fraction =
                    (ix as f64 / grid.nx as f64) * 0.3 + (iy as f64 / grid.ny as f64) * 0.2;
                WATER_SPECIFIC_HEAT * (1.0 - tissue_fraction)
                    + TISSUE_SPECIFIC_HEAT * tissue_fraction
            } else {
                WATER_SPECIFIC_HEAT
            }
        }

        fn specific_heat_capacity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Alias for specific_heat
            self.specific_heat(x, y, z, grid)
        }

        fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Thermal conductivity
            // Water: 0.6 W/(m·K), Tissue: 0.5-0.6 W/(m·K)
            const BASE_CONDUCTIVITY: f64 = 0.6;

            if self.position_dependent {
                let (ix, iy, iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                // Spatial variation for tissue heterogeneity
                let variation = 0.1
                    * ((ix as f64 * 0.1).sin() + (iy as f64 * 0.1).cos() + (iz as f64 * 0.1).sin());
                BASE_CONDUCTIVITY * (1.0 + variation)
            } else {
                BASE_CONDUCTIVITY
            }
        }

        fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Thermal diffusivity: α = k/(ρ·c_p)
            // Water: 1.4e-7 m²/s, Tissue: 1.3-1.5e-7 m²/s
            let k = self.thermal_conductivity(x, y, z, grid);
            let rho = self.density(x, y, z, grid);
            let cp = self.specific_heat(x, y, z, grid);
            k / (rho * cp)
        }

        fn thermal_expansion(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Thermal expansion coefficient
            // Water: 2.1e-4 K⁻¹ at 20°C, increases with temperature
            // Tissue: 3-4e-4 K⁻¹
            const WATER_EXPANSION: f64 = 2.1e-4;
            const TISSUE_EXPANSION: f64 = 3.5e-4;

            if self.position_dependent {
                let (ix, iy, _iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                // Interpolate based on position
                let tissue_fraction =
                    (ix as f64 / grid.nx as f64 + iy as f64 / grid.ny as f64) * 0.5;
                WATER_EXPANSION * (1.0 - tissue_fraction) + TISSUE_EXPANSION * tissue_fraction
            } else {
                WATER_EXPANSION
            }
        }

        fn specific_heat_ratio(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            1.4
        }

        fn gamma(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            1.4
        }
    }

    impl crate::medium::thermal::TemperatureState for HeterogeneousMediumMock {
        fn update_temperature(&mut self, temperature: &ndarray::Array3<f64>) {
            self.temperature.assign(temperature);
        }

        fn temperature(&self) -> &ndarray::Array3<f64> {
            &self.temperature
        }
    }

    impl crate::medium::optical::OpticalProperties for HeterogeneousMediumMock {
        fn optical_absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Optical absorption coefficient μ_a (1/m)
            // Water: ~0.05/m at 800nm, Tissue: 10-100/m
            const WATER_ABSORPTION: f64 = 0.05;
            const TISSUE_ABSORPTION: f64 = 50.0;

            if self.position_dependent {
                // Use tissue type to determine absorption
                if let Some(tissue) = self.tissue_type(x, y, z, grid) {
                    use crate::medium::absorption::TissueType;
                    match tissue {
                        TissueType::Muscle => 80.0,
                        TissueType::Fat => 20.0,
                        TissueType::Liver => 100.0,
                        _ => TISSUE_ABSORPTION,
                    }
                } else {
                    TISSUE_ABSORPTION
                }
            } else {
                WATER_ABSORPTION
            }
        }

        fn optical_scattering_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Reduced scattering coefficient μ'_s (1/m)
            // Water: ~0.01/m, Tissue: 500-2000/m
            const WATER_SCATTERING: f64 = 0.01;

            if self.position_dependent {
                // Tissue-dependent scattering
                if let Some(tissue) = self.tissue_type(x, y, z, grid) {
                    use crate::medium::absorption::TissueType;
                    match tissue {
                        TissueType::Muscle => 1200.0,
                        TissueType::Fat => 800.0,
                        TissueType::Liver => 1500.0,
                        _ => 1000.0,
                    }
                } else {
                    1000.0
                }
            } else {
                WATER_SCATTERING
            }
        }

        fn refractive_index(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Refractive index
            // Water: 1.33, Tissue: 1.37-1.40
            const WATER_REFRACTIVE_INDEX: f64 = 1.33;
            const TISSUE_REFRACTIVE_INDEX: f64 = 1.38;

            if self.position_dependent {
                let (ix, iy, _iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                // Smooth transition between water and tissue
                let tissue_fraction = ((ix as f64 / grid.nx as f64).powi(2)
                    + (iy as f64 / grid.ny as f64).powi(2))
                .sqrt()
                    / 1.414;
                WATER_REFRACTIVE_INDEX
                    + (TISSUE_REFRACTIVE_INDEX - WATER_REFRACTIVE_INDEX) * tissue_fraction
            } else {
                WATER_REFRACTIVE_INDEX
            }
        }

        fn anisotropy_factor(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Anisotropy factor g for Henyey-Greenstein phase function
            // Water: ~0, Tissue: 0.8-0.95
            const WATER_ANISOTROPY: f64 = 0.0;

            if self.position_dependent {
                // Tissue-dependent anisotropy
                if let Some(tissue) = self.tissue_type(x, y, z, grid) {
                    use crate::medium::absorption::TissueType;
                    match tissue {
                        TissueType::Muscle => 0.90,
                        TissueType::Fat => 0.85,
                        TissueType::Liver => 0.92,
                        _ => 0.90,
                    }
                } else {
                    0.90
                }
            } else {
                WATER_ANISOTROPY
            }
        }

        fn reduced_scattering_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            0.1
        }
    }

    impl crate::medium::viscous::ViscousProperties for HeterogeneousMediumMock {
        fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Dynamic viscosity (Pa·s)
            // Water at 20°C: 1e-3 Pa·s, at 37°C: 0.7e-3 Pa·s
            // Blood: 3-4e-3 Pa·s
            const WATER_VISCOSITY: f64 = 1e-3;
            const BLOOD_VISCOSITY: f64 = 3.5e-3;

            if self.position_dependent {
                // Model blood vessels as regions with higher viscosity
                let (ix, iy, iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                // Create vessel-like structures
                let vessel_pattern = ((ix as f64 * 0.3).sin() * (iy as f64 * 0.3).cos()
                    + (iz as f64 * 0.2).sin())
                .abs();
                if vessel_pattern > 0.7 {
                    BLOOD_VISCOSITY
                } else {
                    WATER_VISCOSITY * (1.0 + 0.5 * vessel_pattern)
                }
            } else {
                WATER_VISCOSITY
            }
        }

        fn shear_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Shear viscosity is typically the same as dynamic viscosity for Newtonian fluids
            self.viscosity(x, y, z, grid)
        }

        fn bulk_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Bulk viscosity (second viscosity)
            // For water: ~2.5 times shear viscosity (Stokes' hypothesis not valid)
            // For tissue: varies significantly
            const BULK_TO_SHEAR_RATIO: f64 = 2.5;

            if self.position_dependent {
                // Tissue can have different bulk viscosity characteristics
                let shear = self.shear_viscosity(x, y, z, grid);
                let (ix, iy, _iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                // Spatial variation in bulk viscosity ratio
                let ratio_variation = BULK_TO_SHEAR_RATIO
                    * (1.0 + 0.3 * (ix as f64 / grid.nx as f64 + iy as f64 / grid.ny as f64));
                shear * ratio_variation
            } else {
                self.shear_viscosity(x, y, z, grid) * BULK_TO_SHEAR_RATIO
            }
        }

        fn kinematic_viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            1e-6
        }
    }

    impl crate::medium::bubble::BubbleProperties for HeterogeneousMediumMock {
        fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Surface tension (N/m)
            // Water at 20°C: 0.072 N/m, at 37°C: 0.069 N/m
            // Blood plasma: 0.056 N/m
            const WATER_SURFACE_TENSION: f64 = 0.072;
            const PLASMA_SURFACE_TENSION: f64 = 0.056;

            if self.position_dependent {
                // Model variation in surface tension due to surfactants
                let (ix, iy, iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                // Surfactant concentration pattern
                let surfactant =
                    ((ix as f64 * 0.1).sin() + (iy as f64 * 0.15).cos() + (iz as f64 * 0.1).sin())
                        / 3.0;
                WATER_SURFACE_TENSION * (1.0 - 0.3 * surfactant.abs())
            } else {
                WATER_SURFACE_TENSION
            }
        }

        fn ambient_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Ambient pressure (Pa)
            // Standard atmospheric: 101325 Pa
            // Include hydrostatic pressure variation
            const ATMOSPHERIC_PRESSURE: f64 = 101325.0;
            const WATER_DENSITY: f64 = 1000.0;
            const GRAVITY: f64 = 9.81;

            if self.position_dependent {
                // Add hydrostatic pressure based on depth (z-coordinate)
                let (_ix, _iy, iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                let depth = (grid.nz - iz) as f64 * grid.dz; // Depth from top
                ATMOSPHERIC_PRESSURE + WATER_DENSITY * GRAVITY * depth
            } else {
                ATMOSPHERIC_PRESSURE
            }
        }

        fn vapor_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Vapor pressure (Pa)
            // Water at 20°C: 2330 Pa, at 37°C: 6270 Pa
            const VAPOR_PRESSURE_20C: f64 = 2330.0;
            const VAPOR_PRESSURE_37C: f64 = 6270.0;

            if self.position_dependent {
                // Temperature-dependent vapor pressure
                // Assume temperature varies spatially
                let (ix, iy, _iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                let temp_factor = (ix as f64 / grid.nx as f64 + iy as f64 / grid.ny as f64) * 0.5;
                VAPOR_PRESSURE_20C + (VAPOR_PRESSURE_37C - VAPOR_PRESSURE_20C) * temp_factor
            } else {
                VAPOR_PRESSURE_20C
            }
        }

        fn polytropic_index(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Polytropic index for gas inside bubbles
            // Air (diatomic): 1.4 (adiabatic)
            // Isothermal: 1.0
            // Real bubbles: 1.0-1.4 depending on size and frequency
            const ADIABATIC_INDEX: f64 = 1.4;
            const ISOTHERMAL_INDEX: f64 = 1.0;

            if self.position_dependent {
                // Model size-dependent polytropic behavior
                let (ix, iy, iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                // Larger bubbles tend toward isothermal, smaller toward adiabatic
                let size_factor = ((ix + iy + iz) as f64 / (grid.nx + grid.ny + grid.nz) as f64);
                ISOTHERMAL_INDEX + (ADIABATIC_INDEX - ISOTHERMAL_INDEX) * (1.0 - size_factor)
            } else {
                ADIABATIC_INDEX
            }
        }

        fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Gas diffusion coefficient (m²/s)
            // Air in water at 20°C: ~2e-9 m²/s
            // Increases with temperature
            const BASE_DIFFUSION: f64 = 2e-9;

            if self.position_dependent {
                // Temperature and tissue-dependent diffusion
                let (ix, iy, iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                // Temperature effect (higher temp = higher diffusion)
                let temp_factor = 1.0
                    + 0.03
                        * (ix as f64 / grid.nx as f64
                            + iy as f64 / grid.ny as f64
                            + iz as f64 / grid.nz as f64);
                BASE_DIFFUSION * temp_factor
            } else {
                BASE_DIFFUSION
            }
        }
    }

    impl crate::medium::bubble::BubbleState for HeterogeneousMediumMock {
        fn bubble_radius(&self) -> &ndarray::Array3<f64> {
            &self.bubble_radius
        }

        fn bubble_velocity(&self) -> &ndarray::Array3<f64> {
            &self.bubble_velocity
        }

        fn update_bubble_state(
            &mut self,
            radius: &ndarray::Array3<f64>,
            velocity: &ndarray::Array3<f64>,
        ) {
            self.bubble_radius.assign(radius);
            self.bubble_velocity.assign(velocity);
        }
    }

    #[test]
    fn test_zero_frequency_safety() {
        // Test that zero frequency doesn't cause division by zero
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 5.0, &grid);

        // This should not panic and should return 0.0
        let diffusivity = compute_acoustic_diffusivity(&medium, 0.0, 0.0, 0.0, &grid, 0.0);
        assert_eq!(
            diffusivity, 0.0,
            "Zero frequency should return zero diffusivity"
        );

        // Test with very small frequency (should not panic)
        let small_freq = 1e-10;
        let diffusivity_small =
            compute_acoustic_diffusivity(&medium, 0.0, 0.0, 0.0, &grid, small_freq);
        assert!(
            diffusivity_small.is_finite(),
            "Small frequency should produce finite result"
        );
    }

    #[test]
    fn test_acoustic_diffusivity_heterogeneous() {
        // Test that the function correctly uses spatial coordinates
        let medium = HeterogeneousMediumMock::new(true);
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
        let frequency = 1e6;

        // Test point 1: x=0.1, y=0.2, z=0.3
        let diffusivity1 = compute_acoustic_diffusivity(&medium, 0.1, 0.2, 0.3, &grid, frequency);
        let c1: f64 = 1600.0; // x < 0.2
        let alpha1 = 0.5 + 0.1 * 0.1 + 0.05 * 0.2 + 0.02 * 0.3; // 0.526
        let omega = 2.0 * PI * frequency;
        let expected1 = 2.0 * alpha1 * c1.powi(3) / (omega * omega);
        assert!(
            (diffusivity1 - expected1).abs() < 1e-10,
            "Heterogeneous test 1 failed: got {}, expected {}",
            diffusivity1,
            expected1
        );

        // Test point 2: x=0.4, y=0.3, z=0.5
        let diffusivity2 = compute_acoustic_diffusivity(&medium, 0.4, 0.3, 0.5, &grid, frequency);
        let c2: f64 = 1400.0; // x >= 0.2 and y < 0.5
        let alpha2 = 0.5 + 0.1 * 0.4 + 0.05 * 0.3 + 0.02 * 0.5; // 0.565
        let expected2 = 2.0 * alpha2 * c2.powi(3) / (omega * omega);
        assert!(
            (diffusivity2 - expected2).abs() < 1e-10,
            "Heterogeneous test 2 failed: got {}, expected {}",
            diffusivity2,
            expected2
        );

        // Test point 3: x=0.5, y=0.6, z=0.7
        let diffusivity3 = compute_acoustic_diffusivity(&medium, 0.5, 0.6, 0.7, &grid, frequency);
        let c3: f64 = 1500.0; // x >= 0.2 and y >= 0.5
        let alpha3 = 0.5 + 0.1 * 0.5 + 0.05 * 0.6 + 0.02 * 0.7; // 0.594
        let expected3 = 2.0 * alpha3 * c3.powi(3) / (omega * omega);
        assert!(
            (diffusivity3 - expected3).abs() < 1e-10,
            "Heterogeneous test 3 failed: got {}, expected {}",
            diffusivity3,
            expected3
        );

        // Verify that different positions give different results
        assert!(
            (diffusivity1 - diffusivity2).abs() > 1e-12,
            "Different positions should yield different diffusivities in heterogeneous medium"
        );
        assert!(
            (diffusivity2 - diffusivity3).abs() > 1e-12,
            "Different positions should yield different diffusivities in heterogeneous medium"
        );
    }

    #[test]
    fn test_acoustic_diffusivity_calculation() {
        // Test that the formula δ = 2αc³/ω² is correctly implemented

        // Test case 1: Zero absorption should give zero diffusivity
        let alpha: f64 = 0.0;
        let c: f64 = 1500.0;
        let freq: f64 = 1e6;
        let omega = 2.0 * PI * freq;
        let expected = 2.0 * alpha * c.powi(3) / (omega * omega);
        assert_eq!(expected, 0.0);

        // Test case 2: Non-zero values
        let alpha: f64 = 0.5; // Np/m
        let c: f64 = 1500.0; // m/s
        let freq: f64 = 1e6; // Hz
        let omega = 2.0 * PI * freq;
        let diffusivity = 2.0 * alpha * c.powi(3) / (omega * omega);

        // Calculate expected value
        let expected = 2.0 * 0.5 * 1500.0_f64.powi(3) / (2.0 * PI * 1e6).powi(2);

        assert!(
            (diffusivity - expected).abs() < 1e-10,
            "Formula calculation mismatch: got {}, expected {}",
            diffusivity,
            expected
        );

        // Test case 3: Verify frequency scaling
        let freq2: f64 = 2e6;
        let omega2 = 2.0 * PI * freq2;
        let diffusivity2 = 2.0 * alpha * c.powi(3) / (omega2 * omega2);

        // Diffusivity should scale as 1/f² for constant α
        assert!(
            (diffusivity2 - diffusivity / 4.0).abs() < 1e-10,
            "Frequency scaling incorrect: {} vs {}",
            diffusivity2,
            diffusivity / 4.0
        );

        // Test case 4: Verify the actual value is reasonable
        // For α = 0.5 Np/m, c = 1500 m/s, f = 1 MHz
        // δ = 2 * 0.5 * 1500³ / (2π * 10⁶)² ≈ 8.5e-5 m²/s
        assert!(
            diffusivity > 1e-6 && diffusivity < 1e-3,
            "Diffusivity value seems unreasonable: {}",
            diffusivity
        );
    }
}
