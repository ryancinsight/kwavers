//! PyO3 binding: `MultiRowRingArray` Python class.

use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use numpy::{ToPyArray, PyArray2};
use pyo3::prelude::*;

use super::helpers::{kwavers_to_py, points_to_array};

#[pyclass(name = "MultiRowRingArray")]
#[derive(Clone)]
pub struct PyMultiRowRingArray {
    pub(super) inner: MultiRowRingArray,
}

#[pymethods]
impl PyMultiRowRingArray {
    #[new]
    pub fn new(
        circumferential_elements: usize,
        rows: usize,
        diameter_m: f64,
        row_spacing_m: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MultiRowRingArray::new(
                circumferential_elements,
                rows,
                diameter_m,
                row_spacing_m,
            )
            .map_err(kwavers_to_py)?,
        })
    }

    #[staticmethod]
    pub fn ali_2025() -> PyResult<Self> {
        Ok(Self {
            inner: MultiRowRingArray::ali_2025().map_err(kwavers_to_py)?,
        })
    }

    #[getter]
    pub fn circumferential_elements(&self) -> usize {
        self.inner.circumferential_elements()
    }

    #[getter]
    pub fn rows(&self) -> usize {
        self.inner.rows()
    }

    #[getter]
    pub fn diameter_m(&self) -> f64 {
        self.inner.diameter_m()
    }

    #[getter]
    pub fn row_spacing_m(&self) -> f64 {
        self.inner.row_spacing_m()
    }

    #[getter]
    pub fn element_count(&self) -> usize {
        self.inner.element_count()
    }

    pub fn elements<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        points_to_array(self.inner.elements())
            .to_pyarray(py)
            .into()
    }

    pub fn cylindrical_source<'py>(
        &self,
        py: Python<'py>,
        transmit_index: usize,
    ) -> Py<PyArray2<f64>> {
        points_to_array(&self.inner.cylindrical_source(transmit_index))
            .to_pyarray(py)
            .into()
    }
}

