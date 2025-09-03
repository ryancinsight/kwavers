//! Numerical flux implementations for DG methods
//!
//! This module provides various numerical flux functions for discontinuous
//! Galerkin methods, including upwind, Lax-Friedrichs, and Riemann solvers.

use crate::KwaversResult;

/// Numerical flux type
#[derive(Debug, Clone, Copy)]
pub enum FluxType {
    /// Local Lax-Friedrichs (Rusanov) flux
    LaxFriedrichs,
    /// Roe flux
    Roe,
    /// HLL flux
    HLL,
    /// HLLC flux
    HLLC,
}

/// Limiter type for shock capturing
#[derive(Debug, Clone, Copy)]
pub enum LimiterType {
    /// No limiting
    None,
    /// Minmod limiter
    Minmod,
    /// Van Albada limiter
    VanAlbada,
    /// MC limiter
    MC,
    /// Superbee limiter
    Superbee,
}

/// Compute numerical flux at interface
pub fn compute_numerical_flux(
    flux_type: FluxType,
    left_state: f64,
    right_state: f64,
    left_flux: f64,
    right_flux: f64,
    wave_speed: f64,
    normal: f64,
) -> KwaversResult<f64> {
    let flux = match flux_type {
        FluxType::LaxFriedrichs => {
            // Local Lax-Friedrichs (Rusanov) flux
            let max_speed = wave_speed.abs();
            0.5 * (left_flux + right_flux - max_speed * (right_state - left_state) * normal)
        }
        FluxType::Roe => {
            // Roe flux - simplified for scalar case
            let a_roe = if (right_state - left_state).abs() > 1e-10 {
                (right_flux - left_flux) / (right_state - left_state)
            } else {
                wave_speed
            };

            if a_roe > 0.0 {
                left_flux
            } else {
                right_flux
            }
        }
        FluxType::HLL => {
            // HLL flux
            let s_l = -wave_speed; // Left wave speed
            let s_r = wave_speed; // Right wave speed

            if s_l >= 0.0 {
                left_flux
            } else if s_r <= 0.0 {
                right_flux
            } else {
                (s_r * left_flux - s_l * right_flux + s_l * s_r * (right_state - left_state))
                    / (s_r - s_l)
            }
        }
        FluxType::HLLC => {
            // HLLC flux - for now, defaults to HLL for scalar case
            // Full HLLC requires additional wave structure
            compute_numerical_flux(
                FluxType::HLL,
                left_state,
                right_state,
                left_flux,
                right_flux,
                wave_speed,
                normal,
            )?
        }
    };

    Ok(flux)
}

/// Apply limiter function
#[must_use]
pub fn apply_limiter(limiter_type: LimiterType, a: f64, b: f64) -> f64 {
    match limiter_type {
        LimiterType::None => a,
        LimiterType::Minmod => minmod(a, b),
        LimiterType::VanAlbada => van_albada(a, b),
        LimiterType::MC => mc_limiter(a, b),
        LimiterType::Superbee => superbee(a, b),
    }
}

/// Minmod limiter
#[must_use]
pub fn minmod(a: f64, b: f64) -> f64 {
    if a * b > 0.0 {
        if a.abs() < b.abs() {
            a
        } else {
            b
        }
    } else {
        0.0
    }
}

/// Van Albada limiter
#[must_use]
pub fn van_albada(a: f64, b: f64) -> f64 {
    if a * b > 0.0 {
        (a * b * (a + b)) / (a * a + b * b + 1e-10)
    } else {
        0.0
    }
}

/// MC (Monotonized Central) limiter
#[must_use]
pub fn mc_limiter(a: f64, b: f64) -> f64 {
    let c = 0.5 * (a + b);
    let d = 2.0 * a;
    let e = 2.0 * b;

    if a * b > 0.0 {
        minmod(c, minmod(d, e))
    } else {
        0.0
    }
}

/// Superbee limiter
#[must_use]
pub fn superbee(a: f64, b: f64) -> f64 {
    if a * b > 0.0 {
        let s1 = minmod(b, 2.0 * a);
        let s2 = minmod(a, 2.0 * b);
        s1.abs().max(s2.abs()) * a.signum()
    } else {
        0.0
    }
}

/// Compute upwind flux for simple advection
#[must_use]
pub fn upwind_flux(velocity: f64, left_value: f64, right_value: f64) -> f64 {
    if velocity > 0.0 {
        velocity * left_value
    } else {
        velocity * right_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_minmod_limiter() {
        assert_eq!(minmod(1.0, 2.0), 1.0);
        assert_eq!(minmod(-1.0, -2.0), -1.0);
        assert_eq!(minmod(1.0, -1.0), 0.0);
        assert_eq!(minmod(-1.0, 1.0), 0.0);
    }

    #[test]
    fn test_upwind_flux() {
        let velocity = 1.0;
        let left = 2.0;
        let right = 3.0;
        assert_eq!(upwind_flux(velocity, left, right), 2.0);
        assert_eq!(upwind_flux(-velocity, left, right), -3.0);
    }

    #[test]
    fn test_lax_friedrichs_flux() {
        let flux = compute_numerical_flux(
            FluxType::LaxFriedrichs,
            1.0, // left state
            2.0, // right state
            1.0, // left flux
            4.0, // right flux
            2.0, // wave speed
            1.0, // normal
        )
        .unwrap();

        // LF flux = 0.5 * (F_L + F_R - |a| * (U_R - U_L))
        let expected = 0.5 * (1.0 + 4.0 - 2.0 * (2.0 - 1.0));
        assert_relative_eq!(flux, expected, epsilon = 1e-10);
    }
}
