//! Test support mocks for acoustic wave testing
//!
//! This module contains mock implementations used for testing acoustic wave physics.

#[cfg(test)]
pub(crate) mod mocks {
    use crate::error::KwaversResult;
    use crate::grid::Grid;
    use crate::medium::absorption::TissueType;
    use ndarray::{Array3, ArrayViewMut3};
    use std::f64::consts::PI;

    /// Mock heterogeneous medium for testing
    ///
    /// This mock properly implements position-dependent properties for testing
    /// heterogeneous acoustic propagation scenarios.
    #[derive(Clone, Debug)]
    pub struct HeterogeneousMediumMock {
        pub position_dependent: bool,
        pub density: Array3<f64>,
        pub sound_speed: Array3<f64>,
        pub bubble_radius: Array3<f64>,
        pub absorption: Array3<f64>,
        pub nonlinearity: Array3<f64>,
    }

    impl HeterogeneousMediumMock {
        pub fn new(position_dependent: bool) -> Self {
            Self {
                position_dependent,
                density: Array3::from_elem((10, 10, 10), 1000.0),
                sound_speed: Array3::from_elem((10, 10, 10), 1500.0),
                bubble_radius: Array3::from_elem((10, 10, 10), 1e-6),
                absorption: Array3::from_elem((10, 10, 10), 0.0022),
                nonlinearity: Array3::from_elem((10, 10, 10), 5.0),
            }
        }
    }

    impl crate::medium::core::CoreMedium for HeterogeneousMediumMock {
        fn density(&self, i: usize, j: usize, k: usize) -> f64 {
            if self.position_dependent {
                // Spatially varying density simulating tissue heterogeneity
                let base_density = 1000.0;
                let variation = 50.0
                    * ((i as f64 * 0.1).sin() + (j as f64 * 0.1).cos() + (k as f64 * 0.1).sin());
                base_density + variation
            } else {
                1000.0
            }
        }

        fn sound_speed(&self, i: usize, j: usize, k: usize) -> f64 {
            if self.position_dependent {
                // Spatially varying sound speed
                let base_speed = 1500.0;
                let variation = 40.0
                    * ((i as f64 * 0.05).cos() + (j as f64 * 0.05).sin() + (k as f64 * 0.05).cos());
                base_speed + variation
            } else {
                1500.0
            }
        }

        fn reference_frequency(&self) -> f64 {
            1e6
        }

        fn absorption(&self, i: usize, j: usize, k: usize) -> f64 {
            if self.position_dependent {
                0.1 + 0.05 * ((i as f64 * 0.1).sin() + (j as f64 * 0.1).cos())
            } else {
                0.1
            }
        }

        fn nonlinearity(&self, _i: usize, _j: usize, _k: usize) -> f64 {
            5.0 // B/A parameter for water
        }

        fn max_sound_speed(&self) -> f64 {
            if self.position_dependent {
                1540.0 // max with variations
            } else {
                1500.0
            }
        }

        fn is_homogeneous(&self) -> bool {
            !self.position_dependent
        }

        fn validate(&self, _grid: &Grid) -> KwaversResult<()> {
            Ok(())
        }
    }

    impl crate::medium::core::ArrayAccess for HeterogeneousMediumMock {
        fn density_array(&self) -> ndarray::ArrayView3<f64> {
            self.density.view()
        }

        fn sound_speed_array(&self) -> ndarray::ArrayView3<f64> {
            self.sound_speed.view()
        }

        fn density_array_mut(&mut self) -> Option<ArrayViewMut3<f64>> {
            Some(self.density.view_mut())
        }

        fn sound_speed_array_mut(&mut self) -> Option<ArrayViewMut3<f64>> {
            Some(self.sound_speed.view_mut())
        }

        fn absorption_array(&self) -> ndarray::ArrayView3<f64> {
            self.absorption.view()
        }

        fn nonlinearity_array(&self) -> ndarray::ArrayView3<f64> {
            self.nonlinearity.view()
        }
    }

    impl crate::medium::acoustic::AcousticProperties for HeterogeneousMediumMock {
        fn absorption_coefficient(
            &self,
            x: f64,
            y: f64,
            z: f64,
            grid: &Grid,
            frequency: f64,
        ) -> f64 {
            // Power law absorption: α = α₀ * f^y
            const ALPHA_0: f64 = 0.5; // dB/cm/MHz
            const POWER_LAW_EXPONENT: f64 = 1.1;
            const DB_TO_NP_PER_M: f64 = 100.0 / 8.686;

            if self.position_dependent {
                let spatial_factor = 1.0 + 0.2 * ((x / grid.dx).sin() + (y / grid.dy).cos());
                ALPHA_0
                    * (frequency / 1e6).powf(POWER_LAW_EXPONENT)
                    * DB_TO_NP_PER_M
                    * spatial_factor
            } else {
                ALPHA_0 * (frequency / 1e6).powf(POWER_LAW_EXPONENT) * DB_TO_NP_PER_M
            }
        }

        fn attenuation(&self, x: f64, y: f64, z: f64, frequency: f64, grid: &Grid) -> f64 {
            self.absorption_coefficient(x, y, z, grid, frequency)
        }

        fn nonlinearity_parameter(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // B/A parameter for nonlinear acoustics
            // Water: 5.0, Tissue: 6-9, Fat: 10
            if self.position_dependent {
                let (ix, iy, _iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                // Vary B/A based on position to simulate tissue heterogeneity
                let normalized_x = ix as f64 / grid.nx as f64;
                let normalized_y = iy as f64 / grid.ny as f64;
                5.0 + 2.0 * normalized_x + 1.5 * normalized_y.sin()
            } else {
                5.0 // Water value
            }
        }

        fn nonlinearity_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Beta = 1 + B/(2A)
            let b_over_a = self.nonlinearity_parameter(x, y, z, grid);
            1.0 + b_over_a / 2.0
        }

        fn acoustic_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            // Acoustic diffusivity δ = (4μ/3 + μ_B)/(ρc²)
            const BASE_DIFFUSIVITY: f64 = 1.4e-7;

            if self.position_dependent {
                let (ix, iy, _iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                let variation = 0.3 * ((ix as f64 * 0.1).sin() + (iy as f64 * 0.1).cos());
                BASE_DIFFUSIVITY * (1.0 + variation)
            } else {
                BASE_DIFFUSIVITY
            }
        }

        fn tissue_type(&self, x: f64, y: f64, z: f64, grid: &Grid) -> Option<TissueType> {
            if self.position_dependent {
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
                None
            }
        }
    }

    // Additional trait implementations for completeness
    impl crate::medium::elastic::ElasticProperties for HeterogeneousMediumMock {
        fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            const BASE_LAMBDA: f64 = 2.2e9;

            if self.position_dependent {
                let (ix, iy, _iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                let variation = 0.2 * ((ix as f64 * 0.05).sin() + (iy as f64 * 0.05).cos());
                BASE_LAMBDA * (1.0 + variation)
            } else {
                BASE_LAMBDA
            }
        }

        fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            if self.position_dependent {
                let (ix, iy, _iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                let base_mu = 10e3; // 10 kPa typical for soft tissue
                let variation = (ix as f64 / grid.nx as f64) + (iy as f64 / grid.ny as f64) * 0.5;
                base_mu * (1.0 + variation)
            } else {
                0.0 // Water has no shear resistance
            }
        }

        fn shear_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            if self.position_dependent {
                let mu = self.lame_mu(x, y, z, grid);
                let rho = crate::medium::density_at(self, x, y, z, grid);
                if mu > 0.0 {
                    (mu / rho).sqrt()
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }

        fn compressional_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            crate::medium::sound_speed_at(self, x, y, z, grid)
        }
    }

    impl crate::medium::elastic::ElasticArrayAccess for HeterogeneousMediumMock {
        fn lame_lambda_array(&self) -> Array3<f64> {
            self.density.clone()
        }

        fn lame_mu_array(&self) -> Array3<f64> {
            self.bubble_radius.clone()
        }
    }

    // Thermal properties
    impl crate::medium::thermal::ThermalProperties for HeterogeneousMediumMock {
        fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            const WATER_SPECIFIC_HEAT: f64 = 4180.0;
            const TISSUE_SPECIFIC_HEAT: f64 = 3600.0;

            if self.position_dependent {
                let (ix, iy, _iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                let tissue_fraction =
                    (ix as f64 / grid.nx as f64) * 0.3 + (iy as f64 / grid.ny as f64) * 0.2;
                WATER_SPECIFIC_HEAT * (1.0 - tissue_fraction)
                    + TISSUE_SPECIFIC_HEAT * tissue_fraction
            } else {
                WATER_SPECIFIC_HEAT
            }
        }

        fn specific_heat_capacity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            self.specific_heat(x, y, z, grid)
        }

        fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            const BASE_CONDUCTIVITY: f64 = 0.6;

            if self.position_dependent {
                let (ix, iy, iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                let variation = 0.1
                    * ((ix as f64 * 0.1).sin() + (iy as f64 * 0.1).cos() + (iz as f64 * 0.1).sin());
                BASE_CONDUCTIVITY * (1.0 + variation)
            } else {
                BASE_CONDUCTIVITY
            }
        }

        fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            let k = self.thermal_conductivity(x, y, z, grid);
            let rho = crate::medium::density_at(self, x, y, z, grid);
            let cp = self.specific_heat(x, y, z, grid);
            k / (rho * cp)
        }

        fn thermal_expansion(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
            const WATER_EXPANSION: f64 = 2.1e-4;
            const TISSUE_EXPANSION: f64 = 3.5e-4;

            if self.position_dependent {
                let (ix, iy, _iz) = grid.position_to_indices(x, y, z).unwrap_or((0, 0, 0));
                let tissue_fraction =
                    (ix as f64 / grid.nx as f64 + iy as f64 / grid.ny as f64) * 0.5;
                WATER_EXPANSION * (1.0 - tissue_fraction) + TISSUE_EXPANSION * tissue_fraction
            } else {
                WATER_EXPANSION
            }
        }
    }

    impl crate::medium::viscous::ViscousProperties for HeterogeneousMediumMock {
        fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            0.001 // Pa·s (water at 20°C)
        }
    }

    impl crate::medium::optical::OpticalProperties for HeterogeneousMediumMock {
        fn optical_absorption_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            0.01 // 1/m
        }

        fn optical_scattering_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            10.0 // 1/m
        }
    }

    impl crate::medium::thermal::ThermalField for HeterogeneousMediumMock {
        fn thermal_field(&self) -> &Array3<f64> {
            &self.density // Return reference for test
        }

        fn update_thermal_field(&mut self, _new_temperature: &Array3<f64>) {
            // No-op for test
        }
    }

    impl crate::medium::bubble::BubbleProperties for HeterogeneousMediumMock {
        fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            0.0728 // N/m for water
        }

        fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            101325.0 // Pa (1 atm)
        }

        fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            2338.0 // Pa for water at 20°C
        }

        fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            1.4 // Air
        }

        fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            2e-9 // m²/s for air in water
        }
    }

    impl crate::medium::bubble::BubbleState for HeterogeneousMediumMock {
        fn bubble_radius(&self) -> &Array3<f64> {
            &self.bubble_radius
        }

        fn bubble_velocity(&self) -> &Array3<f64> {
            &self.bubble_radius // Reuse for simplicity
        }

        fn update_bubble_state(&mut self, _radius: &Array3<f64>, _velocity: &Array3<f64>) {
            // No-op for test
        }
    }
}
