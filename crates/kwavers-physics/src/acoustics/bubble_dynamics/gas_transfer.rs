//! Dissolved gas transport around an isolated spherical interface.
//!
//! The incompressible liquid velocity is `u = R² Ṙ / r²`. The material
//! coordinate `x = (r³ − R³)/(3 R_ref³)` obeys `dx/dt = 0` along that
//! velocity. For the concentration excess `c = C − C∞` this gives
//!
//! ```text
//! ∂c/∂t = (D/R_ref²) ∂ₓ(a ∂ₓc),  a = ((R/R_ref)³ + 3x)^(4/3)
//! d n_bubble/dt = 4π D R_ref a(0) ∂ₓc(0).
//! ```
//!
//! This is the spherical convection–diffusion equation of Smith & Wang,
//! [JFM 939 A28, Eq. (2.4)](https://doi.org/10.1017/jfm.2022.218), using
//! the full spherical metric in the material coordinate. The interface
//! motion and interfacial concentration are supplied by the caller.

use aequitas::systems::si::quantities::{AreaPerTime, Length, MolarConcentration};
use eunomia::FloatElement;

/// Physical inputs for single-species gas diffusion in an incompressible liquid.
#[derive(Debug, Clone, Copy)]
pub struct SphericalGasDiffusion<T = f64> {
    /// Initial bubble radius and length scale.
    pub reference_radius: Length<T>,
    /// Constant molecular gas diffusivity in the liquid.
    pub diffusivity: AreaPerTime<T>,
    /// Constant far-field dissolved-gas concentration.
    pub bulk_concentration: MolarConcentration<T>,
}

/// State prescribed at the liquid side of the bubble interface.
#[derive(Debug, Clone, Copy)]
pub struct GasDiffusionWall<T = f64> {
    /// Instantaneous bubble radius.
    pub radius: Length<T>,
    /// Interfacial dissolved-gas concentration, for example from Henry's law.
    pub concentration: MolarConcentration<T>,
}

/// Dimensionless material coordinate for `r/R_ref = 1 + gap` at `R = R_ref`.
#[inline]
pub fn spherical_material_coordinate<T: FloatElement>(gap: T) -> T {
    gap * (T::ONE + gap + gap * gap / T::from_count(3))
}

/// Spherical diffusion coefficient for normalized cubed wall radius and
/// dimensionless material coordinate, excluding the scalar `D/R_ref²`.
#[inline]
pub fn spherical_diffusion_metric<T: FloatElement>(wall_radius_cubed: T, coordinate: T) -> T {
    let radius = (wall_radius_cubed + T::from_count(3) * coordinate).cbrt();
    let radius_squared = radius * radius;
    radius_squared * radius_squared
}
