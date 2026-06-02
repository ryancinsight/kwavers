use pyo3::prelude::*;

use super::Simulation;

#[pymethods]
impl Simulation {
    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "Simulation(grid=Grid({nx}x{ny}x{nz}), sources={sources}, transducers={transducers}, solver={solver:?}, sensor={sensor})",
            nx = self.grid.inner.nx,
            ny = self.grid.inner.ny,
            nz = self.grid.inner.nz,
            sources = self.sources.len(),
            transducers = self.transducers.len(),
            solver = self.solver_type,
            sensor = if self.sensor.is_some() { "Sensor" } else { "Transducer" },
        )
    }
}
