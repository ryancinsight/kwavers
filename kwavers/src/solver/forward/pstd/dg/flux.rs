#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxType {
    LaxFriedrichs,
    Upwind,
    Central,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimiterType {
    Minmod,
    Superbee,
    VanLeer,
}

/// Compute numerical flux at interface
pub fn compute_numerical_flux(
    left: f64,
    right: f64,
    flux_type: FluxType,
    wave_speed: f64,
    normal: f64,
) -> f64 {
    // Flux f(u) = c * u
    let f_left = wave_speed * left;
    let f_right = wave_speed * right;

    match flux_type {
        FluxType::LaxFriedrichs => {
            // F = 0.5 * (f_l + f_r) - 0.5 * alpha * (u_r - u_l)
            // alpha = max|f'(u)| = wave_speed
            0.5 * (f_left + f_right) * normal - 0.5 * wave_speed.abs() * (right - left)
        }
        FluxType::Upwind => {
            // If normal * c > 0, flow is left to right -> take left
            if normal * wave_speed > 0.0 {
                f_left * normal
            } else {
                f_right * normal
            }
        }
        FluxType::Central => 0.5 * (f_left + f_right) * normal,
    }
}

/// Apply slope limiter
/// Returns the limited slope or value based on two slopes/differences a and b.
pub fn apply_limiter(limiter_type: LimiterType, a: f64, b: f64) -> f64 {
    // Minmod: zero if signs opposite, else min magnitude
    let minmod = |x: f64, y: f64| -> f64 {
        if x * y <= 0.0 {
            0.0
        } else {
            x.signum() * x.abs().min(y.abs())
        }
    };

    match limiter_type {
        LimiterType::Minmod => minmod(a, b),
        LimiterType::Superbee => {
            // max(minmod(a, 2b), minmod(2a, b))
            let s1 = minmod(a, 2.0 * b);
            let s2 = minmod(2.0 * a, b);
            if s1.abs() > s2.abs() {
                s1
            } else {
                s2
            }
        }
        LimiterType::VanLeer => {
            if a * b <= 0.0 {
                0.0
            } else {
                2.0 * a * b / (a + b)
            }
        }
    }
}
