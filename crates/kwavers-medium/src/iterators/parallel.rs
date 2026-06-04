//! Rayon-parallel medium property iterator.

use rayon::prelude::*;

use kwavers_grid::Grid;
use crate::{AcousticProperties, CoreMedium, Medium};

use super::property::MediumProperties;

/// Parallel iterator adapter for medium properties
#[derive(Debug)]
pub struct ParallelMediumIterator<'a> {
    pub(super) medium: &'a dyn Medium,
    pub(super) grid: &'a Grid,
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
                    density: crate::density_at(self.medium, x, y, z, self.grid),
                    sound_speed: crate::sound_speed_at(
                        self.medium,
                        x,
                        y,
                        z,
                        self.grid,
                    ),
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
                    density: crate::density_at(self.medium, x, y, z, self.grid),
                    sound_speed: crate::sound_speed_at(
                        self.medium,
                        x,
                        y,
                        z,
                        self.grid,
                    ),
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
