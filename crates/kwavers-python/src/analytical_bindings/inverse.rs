//! PyO3 bindings for `kwavers_physics::analytical::inverse` and
//! `kwavers_math::inverse_problems::parameter_selection`.

mod arrays;
mod convergence;
mod operators;
mod reconstruction;
mod seismic;
mod selection;

pub use convergence::{adjoint_gradient_convergence, exponential_convergence_curve};
pub use operators::{helmholtz_1d_fd_matrix, matrix_singular_values, tikhonov_lcurve};
pub use reconstruction::{born_inversion_regularized, gaussian_deconvolution_fixture};
pub use seismic::{eikonal_traveltime_2d, kirchhoff_point_scatterer_image_2d};
pub use selection::{l_curve_corner, morozov_lambda};
