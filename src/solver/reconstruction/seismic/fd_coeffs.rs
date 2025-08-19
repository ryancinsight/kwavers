//! Finite difference coefficients for seismic reconstruction

/// Fourth-order finite difference coefficients for Laplacian
pub const FD_COEFF_0: f64 = -5.0 / 2.0;  // Central coefficient
pub const FD_COEFF_1: f64 = 4.0 / 3.0;   // First neighbor
pub const FD_COEFF_2: f64 = -1.0 / 12.0; // Second neighbor