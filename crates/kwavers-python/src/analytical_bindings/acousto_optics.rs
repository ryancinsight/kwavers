//! PyO3 bindings for `kwavers_physics::analytical::acousto_optics`
//! (acousto-optic diffraction: Raman–Nath, Bragg, Klein–Cook solver).

mod geometry;
mod orders;
mod regime;

pub use geometry::{bragg_angle_rad, diffraction_angle_rad, diffraction_frequency_shift_hz};
pub use orders::{raman_nath_order_intensities, solve_coupled_orders};
pub use regime::{bragg_diffraction_efficiency, klein_cook_parameter, raman_nath_parameter};
