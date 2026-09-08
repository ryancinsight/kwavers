//! Binding-layer parsing and geometry validation helpers for MEMS wrappers.

use kwavers_transducer::mems::{
    cmut::CmutCell,
    pmut::{PiezoFilm, PmutCell},
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::quantity_args::PyLength;

pub(super) fn cmut(radius: PyLength, thickness: PyLength, gap: PyLength) -> PyResult<CmutCell> {
    CmutCell::silicon(
        radius.quantity(),
        thickness.quantity(),
        gap.quantity(),
    )
    .ok_or_else(|| PyValueError::new_err("invalid CMUT geometry (all dimensions must be > 0)"))
}

pub(super) fn pmut(film: &str, radius: PyLength, t_p: PyLength, t_s: PyLength) -> PyResult<PmutCell> {
    PmutCell::new(
        radius.quantity(),
        t_p.quantity(),
        t_s.quantity(),
        parse_film(film)?,
    )
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
