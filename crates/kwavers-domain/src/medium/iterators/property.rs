//! Sequential medium property iterator.

use kwavers_grid::Grid;
use crate::medium::{AcousticProperties, CoreMedium, Medium};

/// Iterator over medium properties at each grid point
#[derive(Debug)]
pub struct MediumPropertyIterator<'a> {
    pub(super) medium: &'a dyn Medium,
    pub(super) grid: &'a Grid,
    pub(super) current: usize,
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
