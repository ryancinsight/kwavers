// src/medium/mod.rs

use std::fmt::Debug;

// Module declarations
pub mod absorption;
pub mod acoustic;
pub mod anisotropic;
pub mod bubble;
pub mod core;
pub mod elastic;
pub mod frequency_dependent;
pub mod heterogeneous;
pub mod homogeneous;
pub mod interface;
pub mod optical;
pub mod thermal;
pub mod traits;
pub mod viscous;
pub mod wrapper;

// Re-export types from submodules
pub use absorption::{PowerLawAbsorption, TissueType};
pub use anisotropic::{AnisotropyType, ChristoffelEquation, MuscleFiberModel, StiffnessTensor};
pub use frequency_dependent::{FrequencyDependentProperties, TissueFrequencyModels};
pub use homogeneous::HomogeneousMedium;

// Re-export new modular traits
pub use acoustic::AcousticProperties;
pub use bubble::{BubbleProperties, BubbleState};
pub use core::{ArrayAccess, CoreMedium};
pub use elastic::{ElasticArrayAccess, ElasticProperties};
pub use optical::OpticalProperties;
pub use thermal::{ThermalField, ThermalProperties};
pub use traits::Medium;
pub use viscous::ViscousProperties;

// Re-export utility functions and types
pub use core::{continuous_to_discrete, max_sound_speed, max_sound_speed_pointwise};
pub use interface::{find_interfaces, InterfacePoint};
pub use wrapper::{
    absorption_at, absorption_at_core, density_at, density_at_core, nonlinearity_at,
    nonlinearity_at_core, sound_speed_at, sound_speed_at_core,
};

// The max_sound_speed function is now provided by the core module
// and re-exported above for backward compatibility

/// Custom iterators for medium property traversal
pub mod iterators {
    use super::{AcousticProperties, CoreMedium, Debug, Medium};
    use crate::grid::Grid;

    use rayon::prelude::*;

    /// Iterator over medium properties at each grid point
    #[derive(Debug)]
    pub struct MediumPropertyIterator<'a> {
        medium: &'a dyn Medium,
        grid: &'a Grid,
        current: usize,
    }

    impl<'a> MediumPropertyIterator<'a> {
        pub fn new(medium: &'a dyn Medium, grid: &'a Grid) -> Self {
            Self {
                medium,
                grid,
                current: 0,
            }
        }
    }

    impl<'a> Iterator for MediumPropertyIterator<'a> {
        type Item = MediumProperties;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current >= self.grid.size() {
                return None;
            }

            let k = self.current % self.grid.nz;
            let j = (self.current / self.grid.nz) % self.grid.ny;
            let i = self.current / (self.grid.ny * self.grid.nz);

            let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);

            let properties = MediumProperties {
                density: crate::medium::density_at(self.medium, x, y, z, self.grid),
                sound_speed: crate::medium::sound_speed_at(self.medium, x, y, z, self.grid),
                absorption: AcousticProperties::absorption_coefficient(
                    self.medium,
                    x,
                    y,
                    z,
                    self.grid,
                    CoreMedium::reference_frequency(self.medium),
                ),
                nonlinearity: AcousticProperties::nonlinearity_coefficient(
                    self.medium,
                    x,
                    y,
                    z,
                    self.grid,
                ),
                position: (x, y, z),
                indices: (i, j, k),
            };

            self.current += 1;
            Some(properties)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let remaining = self.grid.size() - self.current;
            (remaining, Some(remaining))
        }
    }

    impl<'a> ExactSizeIterator for MediumPropertyIterator<'a> {}

    #[derive(Debug, Clone)]
    pub struct MediumProperties {
        pub density: f64,
        pub sound_speed: f64,
        pub absorption: f64,
        pub nonlinearity: f64,
        pub position: (f64, f64, f64),
        pub indices: (usize, usize, usize),
    }

    /// Iterator that finds interfaces in the medium
    #[derive(Debug)]
    pub struct InterfaceIterator<'a> {
        medium: &'a dyn Medium,
        grid: &'a Grid,
        threshold: f64,
        current: usize,
    }

    impl<'a> InterfaceIterator<'a> {
        pub fn new(medium: &'a dyn Medium, grid: &'a Grid, threshold: f64) -> Self {
            Self {
                medium,
                grid,
                threshold,
                current: 0,
            }
        }

        /// Check if a point is at an interface
        fn is_interface(&self, i: usize, j: usize, k: usize) -> bool {
            if i == 0
                || j == 0
                || k == 0
                || i >= self.grid.nx - 1
                || j >= self.grid.ny - 1
                || k >= self.grid.nz - 1
            {
                return false;
            }

            let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
            let center_density = crate::medium::density_at(self.medium, x, y, z, self.grid);

            // Check all 6 neighbors
            let neighbors = [
                (i + 1, j, k),
                (i - 1, j, k),
                (i, j + 1, k),
                (i, j - 1, k),
                (i, j, k + 1),
                (i, j, k - 1),
            ];

            neighbors.iter().any(|&(ni, nj, nk)| {
                let (nx, ny, nz) = self.grid.indices_to_coordinates(ni, nj, nk);
                let neighbor_density =
                    crate::medium::density_at(self.medium, nx, ny, nz, self.grid);
                ((neighbor_density - center_density).abs() / center_density) > self.threshold
            })
        }
    }

    impl<'a> Iterator for InterfaceIterator<'a> {
        type Item = InterfacePoint;

        fn next(&mut self) -> Option<Self::Item> {
            while self.current < self.grid.size() {
                let k = self.current % self.grid.nz;
                let j = (self.current / self.grid.nz) % self.grid.ny;
                let i = self.current / (self.grid.ny * self.grid.nz);

                self.current += 1;

                if self.is_interface(i, j, k) {
                    let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
                    return Some(InterfacePoint {
                        indices: (i, j, k),
                        position: (x, y, z),
                        density_jump: self.calculate_density_jump(i, j, k),
                        normal: self.calculate_normal(i, j, k),
                    });
                }
            }

            None
        }
    }

    impl<'a> InterfaceIterator<'a> {
        fn calculate_density_jump(&self, i: usize, j: usize, k: usize) -> f64 {
            let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
            let center = crate::medium::density_at(self.medium, x, y, z, self.grid);

            let mut max_jump = 0.0;
            for di in -1..=1 {
                for dj in -1..=1 {
                    for dk in -1..=1 {
                        if di == 0 && dj == 0 && dk == 0 {
                            continue;
                        }

                        let ni = (i as i32 + di) as usize;
                        let nj = (j as i32 + dj) as usize;
                        let nk = (k as i32 + dk) as usize;

                        if ni < self.grid.nx && nj < self.grid.ny && nk < self.grid.nz {
                            let (nx, ny, nz) = self.grid.indices_to_coordinates(ni, nj, nk);
                            let neighbor =
                                crate::medium::density_at(self.medium, nx, ny, nz, self.grid);
                            max_jump = f64::max(max_jump, (neighbor - center).abs());
                        }
                    }
                }
            }

            max_jump
        }

        fn calculate_normal(&self, i: usize, j: usize, k: usize) -> (f64, f64, f64) {
            let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);

            // Calculate gradient using central differences
            let dx = if i > 0 && i < self.grid.nx - 1 {
                let (x1, _, _) = self.grid.indices_to_coordinates(i + 1, j, k);
                let (x0, _, _) = self.grid.indices_to_coordinates(i - 1, j, k);
                let d1 = crate::medium::density_at(self.medium, x1, y, z, self.grid);
                let d0 = crate::medium::density_at(self.medium, x0, y, z, self.grid);
                (d1 - d0) / (2.0 * self.grid.dx)
            } else {
                0.0
            };

            let dy = if j > 0 && j < self.grid.ny - 1 {
                let (_, y1, _) = self.grid.indices_to_coordinates(i, j + 1, k);
                let (_, y0, _) = self.grid.indices_to_coordinates(i, j - 1, k);
                let d1 = crate::medium::density_at(self.medium, x, y1, z, self.grid);
                let d0 = crate::medium::density_at(self.medium, x, y0, z, self.grid);
                (d1 - d0) / (2.0 * self.grid.dy)
            } else {
                0.0
            };

            let dz = if k > 0 && k < self.grid.nz - 1 {
                let (_, _, z1) = self.grid.indices_to_coordinates(i, j, k + 1);
                let (_, _, z0) = self.grid.indices_to_coordinates(i, j, k - 1);
                let d1 = crate::medium::density_at(self.medium, x, y, z1, self.grid);
                let d0 = crate::medium::density_at(self.medium, x, y, z0, self.grid);
                (d1 - d0) / (2.0 * self.grid.dz)
            } else {
                0.0
            };

            // Normalize
            let mag = (dx * dx + dy * dy + dz * dz).sqrt();
            if mag > 0.0 {
                (dx / mag, dy / mag, dz / mag)
            } else {
                (0.0, 0.0, 0.0)
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct InterfacePoint {
        pub indices: (usize, usize, usize),
        pub position: (f64, f64, f64),
        pub density_jump: f64,
        pub normal: (f64, f64, f64),
    }

    /// Parallel iterator adapter for medium properties
    #[derive(Debug)]
    pub struct ParallelMediumIterator<'a> {
        medium: &'a dyn Medium,
        grid: &'a Grid,
    }

    impl<'a> ParallelMediumIterator<'a> {
        pub fn new(medium: &'a dyn Medium, grid: &'a Grid) -> Self {
            Self { medium, grid }
        }

        /// Map properties in parallel
        pub fn par_map<F, T>(&self, f: F) -> Vec<T>
        where
            F: Fn(MediumProperties) -> T + Sync + Send,
            T: Send,
        {
            (0..self.grid.size())
                .into_par_iter()
                .map(|idx| {
                    let k = idx % self.grid.nz;
                    let j = (idx / self.grid.nz) % self.grid.ny;
                    let i = idx / (self.grid.ny * self.grid.nz);

                    let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);

                    let properties = MediumProperties {
                        density: crate::medium::density_at(self.medium, x, y, z, self.grid),
                        sound_speed: crate::medium::sound_speed_at(self.medium, x, y, z, self.grid),
                        absorption: AcousticProperties::absorption_coefficient(
                            self.medium,
                            x,
                            y,
                            z,
                            self.grid,
                            CoreMedium::reference_frequency(self.medium),
                        ),
                        nonlinearity: AcousticProperties::nonlinearity_coefficient(
                            self.medium,
                            x,
                            y,
                            z,
                            self.grid,
                        ),
                        position: (x, y, z),
                        indices: (i, j, k),
                    };

                    f(properties)
                })
                .collect()
        }

        /// Filter and collect in parallel
        pub fn par_filter<F>(&self, predicate: F) -> Vec<MediumProperties>
        where
            F: Fn(&MediumProperties) -> bool + Sync + Send,
        {
            (0..self.grid.size())
                .into_par_iter()
                .filter_map(|idx| {
                    let k = idx % self.grid.nz;
                    let j = (idx / self.grid.nz) % self.grid.ny;
                    let i = idx / (self.grid.ny * self.grid.nz);

                    let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);

                    let properties = MediumProperties {
                        density: crate::medium::density_at(self.medium, x, y, z, self.grid),
                        sound_speed: crate::medium::sound_speed_at(self.medium, x, y, z, self.grid),
                        absorption: AcousticProperties::absorption_coefficient(
                            self.medium,
                            x,
                            y,
                            z,
                            self.grid,
                            CoreMedium::reference_frequency(self.medium),
                        ),
                        nonlinearity: AcousticProperties::nonlinearity_coefficient(
                            self.medium,
                            x,
                            y,
                            z,
                            self.grid,
                        ),
                        position: (x, y, z),
                        indices: (i, j, k),
                    };

                    if predicate(&properties) {
                        Some(properties)
                    } else {
                        None
                    }
                })
                .collect()
        }
    }

    /// Extension trait to add iterator methods to Medium
    pub trait MediumIteratorExt {
        /// Create an iterator over medium properties
        fn iter_properties<'a>(&'a self, grid: &'a Grid) -> MediumPropertyIterator<'a>;

        /// Create an interface iterator
        fn iter_interfaces<'a>(&'a self, grid: &'a Grid, threshold: f64) -> InterfaceIterator<'a>;

        /// Create a parallel iterator
        fn par_iter<'a>(&'a self, grid: &'a Grid) -> ParallelMediumIterator<'a>;
    }

    impl MediumIteratorExt for dyn Medium {
        fn iter_properties<'a>(&'a self, grid: &'a Grid) -> MediumPropertyIterator<'a> {
            MediumPropertyIterator::new(self, grid)
        }

        fn iter_interfaces<'a>(&'a self, grid: &'a Grid, threshold: f64) -> InterfaceIterator<'a> {
            InterfaceIterator::new(self, grid, threshold)
        }

        fn par_iter<'a>(&'a self, grid: &'a Grid) -> ParallelMediumIterator<'a> {
            ParallelMediumIterator::new(self, grid)
        }
    }
}
