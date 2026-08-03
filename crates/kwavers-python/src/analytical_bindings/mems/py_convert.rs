//! Binding-layer parsing and geometry validation helpers for MEMS wrappers.

use kwavers_transducer::mems::{
    cmut::CmutCell,
    pmut::{PiezoFilm, PmutCell},
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(super) fn cmut(radius: f64, thickness: f64, gap: f64) -> PyResult<CmutCell> {
    CmutCell::silicon(radius, thickness, gap)
        .ok_or_else(|| PyValueError::new_err("invalid CMUT geometry (all dimensions must be > 0)"))
}

pub(super) fn pmut(film: &str, radius: f64, t_p: f64, t_s: f64) -> PyResult<PmutCell> {
    PmutCell::new(radius, t_p, t_s, parse_film(film)?)
        .ok_or_else(|| PyValueError::new_err("invalid PMUT geometry (all dimensions must be > 0)"))
}

fn parse_film(name: &str) -> PyResult<PiezoFilm> {
    match name.to_ascii_lowercase().as_str() {
        "aln" => Ok(PiezoFilm::Aln),
        "pzt" => Ok(PiezoFilm::Pzt),
        other => Err(PyValueError::new_err(format!(
            "unknown piezo film '{other}' (use 'aln' or 'pzt')"
        ))),
    }
}
