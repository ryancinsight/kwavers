//! Frequency-domain radiation from oriented, rigidly baffled planar apertures.
//!
//! For prescribed normal surface velocity, the Rayleigh first integral is
//!
//! `p(x,ω) = -i ωρ/(2π) ∫_S v_n(s) exp(i k R)/R dS`.
//!
//! This module accepts the equivalent surface-pressure phasor
//! `p_s = ρ c v_n`, so its prefactor is `-i k/(2π)`. Each finite aperture is
//! integrated directly; applying a separate piston-directivity factor would
//! count the same aperture diffraction twice. The convention follows Qin et
//! al., *Ultrasonics* 51 (2011), Eq. 1,
//! <https://doi.org/10.1016/j.ultras.2010.12.011>.

mod aperture;
mod input;
mod integral;
mod path;
mod quadrature;
#[cfg(test)]
mod tests;
mod vector;

pub use aperture::{PlanarAperture, PlanarApertureGeometry, PlanarApertureShape};
pub use integral::{rayleigh_pressure, RayleighIntegralSpec};
pub use path::{RayleighLayer, RayleighPropagationPath};
