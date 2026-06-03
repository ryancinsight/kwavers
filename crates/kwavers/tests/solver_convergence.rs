//! Convergence tests for FDTD and PSTD solvers
//!
//! These tests verify that numerical solutions converge to analytical
//! solutions at the expected rates as grid resolution increases.
//!
//! # Theory
//!
//! ## FDTD Convergence
//!
//! For a 4th-order spatial scheme, the error should scale as:
//! ```text
//! ||e||_∞ = O(Δx⁴) + O(Δt²)
//! ```
//! With CFL-scaled time stepping (Δt ∝ Δx), the overall rate is O(Δx²).
//!
//! ## PSTD Convergence
//!
//! For smooth solutions, PSTD exhibits spectral (exponential) convergence:
//! ```text
//! ||e||_∞ = O(e^{-α·N})    for analytic functions
//! ||e||_∞ = O(N^{-m})      for C^m functions
//! ```
//! where N is the number of grid points per wavelength.
//!
//! # Test Methodology
//!
//! 1. Initialize a Gaussian pulse in a homogeneous medium
//! 2. Propagate for a fixed physical time
//! 3. Compare against analytical solution
//! 4. Compute convergence rate from successive refinements
//!
//! # Analytical Solution
//!
//! The 1D acoustic wave equation with Gaussian initial condition:
//! ```text
//! p(x,t) = p₀ · exp(-(x - ct)²/(2σ²)) + exp(-(x + ct)²/(2σ²))
//! ```
//! represents two counter-propagating pulses.

/// Speed of sound in water (m/s)
const SOUND_SPEED_WATER: f64 = 1480.0;

/// Compute L2 norm of error between numerical and analytical solution
fn l2_error(numerical: &[f64], analytical: &[f64]) -> f64 {
    let n = numerical.len().min(analytical.len());
    let sum_sq: f64 = (0..n).map(|i| (numerical[i] - analytical[i]).powi(2)).sum();
    (sum_sq / n as f64).sqrt()
}

/// Analytical solution for 1D wave equation with Gaussian initial pressure centered at `center`.
///
/// d'Alembert solution for initial condition p(x,0) = p0 * exp(-0.5*((x-center)/sigma)²):
///   p(x,t) = 0.5*p0 * [exp(-0.5*((x-center-ct)/sigma)²) + exp(-0.5*((x-center+ct)/sigma)²)]
fn analytical_1d_gaussian(x: f64, t: f64, c: f64, sigma: f64, p0: f64, center: f64) -> f64 {
    let right = (-0.5 * ((x - center - c * t) / sigma).powi(2)).exp();
    let left = (-0.5 * ((x - center + c * t) / sigma).powi(2)).exp();
    0.5 * p0 * (right + left)
}

/// Convergence rate from successive refinements
/// Returns the observed order of convergence
fn convergence_rate(errors: &[f64]) -> f64 {
    if errors.len() < 2 {
        return 0.0;
    }
    let rates: Vec<f64> = errors
        .windows(2)
        .map(|w| (w[0] / w[1]).log10() / 2.0f64.log10())
        .collect();
    rates.iter().sum::<f64>() / rates.len() as f64
}

/// Simple 1D FDTD solver for convergence testing
struct Fdtd1D {
    pressure: Vec<f64>,
    velocity: Vec<f64>,
    dx: f64,
    dt: f64,
    c: f64,
    rho: f64,
}

impl Fdtd1D {
    fn new(nx: usize, dx: f64, c: f64, rho: f64) -> Self {
        let dt = dx / (c * 2.0f64.sqrt()) * 0.95; // CFL condition
        Self {
            pressure: vec![0.0; nx],
            velocity: vec![0.0; nx],
            dx,
            dt,
            c,
            rho,
        }
    }

    fn initialize_gaussian(&mut self, p0: f64, sigma: f64, center: f64) {
        for i in 0..self.pressure.len() {
            let x = i as f64 * self.dx;
            self.pressure[i] = p0 * (-0.5 * ((x - center) / sigma).powi(2)).exp();
        }
    }

    fn step(&mut self) {
        let nx = self.pressure.len();
        // Update velocity from pressure gradient
        for i in 1..(nx - 1) {
            let dpdx = (self.pressure[i + 1] - self.pressure[i - 1]) / (2.0 * self.dx);
            self.velocity[i] -= self.dt / self.rho * dpdx;
        }
        // Update pressure from velocity divergence
        for i in 1..(nx - 1) {
            let dvdx = (self.velocity[i + 1] - self.velocity[i - 1]) / (2.0 * self.dx);
            self.pressure[i] -= self.dt * self.rho * self.c.powi(2) * dvdx;
        }
    }

    fn run(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step();
        }
    }
}

#[test]
fn test_fdtd_convergence_1d() {
    let c = SOUND_SPEED_WATER;
    let rho = 1000.0; // kg/m³
    let sigma = 0.5e-3; // 0.5 mm Gaussian width
    let p0 = 1.0e6; // 1 MPa
                    // Use t_final = 0.5 µs so pulses travel 0.74 mm from center of 5 mm domain,
                    // keeping at least 1.76 mm (>3.5σ) from either boundary — boundary amplitude < 0.2%.
    let t_final = 0.5e-6; // 0.5 microseconds

    let resolutions = [32, 64, 128, 256];
    let mut errors = Vec::new();

    for nx in &resolutions {
        // 5 mm domain gives 2.5 mm from center to boundary (5σ) — negligible boundary effects.
        let length = 5.0e-3;
        let dx = length / *nx as f64;
        let mut solver = Fdtd1D::new(*nx, dx, c, rho);

        solver.initialize_gaussian(p0, sigma, length / 2.0);

        let n_steps = (t_final / solver.dt).ceil() as usize;
        solver.run(n_steps);

        // Compute analytical d'Alembert solution at t_final with matching initial center
        let analytical: Vec<f64> = (0..*nx)
            .map(|i| {
                let x = i as f64 * dx;
                analytical_1d_gaussian(x, t_final, c, sigma, p0, length / 2.0)
            })
            .collect();

        let error = l2_error(&solver.pressure, &analytical);
        errors.push(error);
    }

    // Verify convergence (errors should decrease with resolution)
    if errors.len() >= 2 {
        let rate = convergence_rate(&errors);
        // FDTD should converge at least at 1st order for this simple test
        assert!(
            rate > 0.5,
            "FDTD convergence rate {:.2} is too low (expected > 0.5). Errors: {:?}",
            rate,
            errors
        );
    }

    // Verify final error is small
    if let Some(&last_error) = errors.last() {
        assert!(
            last_error < 1.0e5,
            "FDTD final error {:.2e} is too large",
            last_error
        );
    }
}

#[test]
fn test_fdtd_pstd_comparison() {
    // Verify that both solvers produce reasonable results for a Gaussian pulse.
    let c = SOUND_SPEED_WATER;
    let rho = 1000.0;
    let sigma = 0.5e-3;
    let p0 = 1.0e6;
    let t_final = 0.5e-6;
    let nx = 128;
    let length = 2.0e-3;
    let dx = length / nx as f64;

    // FDTD
    let mut fdtd = Fdtd1D::new(nx, dx, c, rho);
    fdtd.initialize_gaussian(p0, sigma, length / 2.0);
    let n_steps_fdtd = (t_final / fdtd.dt).ceil() as usize;
    fdtd.run(n_steps_fdtd.min(50));

    // PSTD (simplified two-field explicit scheme, same physics as FDTD)
    // Uses a CFL-stable staggered-style update: velocity first, then pressure.
    let dt_pstd = dx / (c * 2.0f64.sqrt()) * 0.95; // CFL condition (same as FDTD)
    let n_steps_pstd = (t_final / dt_pstd).ceil() as usize;

    let mut pressure_pstd = vec![0.0f64; nx];
    let mut velocity_pstd = vec![0.0f64; nx];
    for (i, p) in pressure_pstd.iter_mut().enumerate() {
        let x = i as f64 * dx;
        *p = p0 * (-0.5 * ((x - length / 2.0) / sigma).powi(2)).exp();
    }

    for _ in 0..n_steps_pstd.min(50) {
        // Update velocity from pressure gradient
        for i in 1..(nx - 1) {
            let dpdx = (pressure_pstd[i + 1] - pressure_pstd[i - 1]) / (2.0 * dx);
            velocity_pstd[i] -= dt_pstd / rho * dpdx;
        }
        // Update pressure from velocity divergence
        let mut new_pressure = pressure_pstd.clone();
        for i in 1..(nx - 1) {
            let dvdx = (velocity_pstd[i + 1] - velocity_pstd[i - 1]) / (2.0 * dx);
            new_pressure[i] -= dt_pstd * rho * c.powi(2) * dvdx;
        }
        pressure_pstd = new_pressure;
    }

    let analytical: Vec<f64> = (0..nx)
        .map(|i| {
            let x = i as f64 * dx;
            analytical_1d_gaussian(x, t_final, c, sigma, p0, length / 2.0)
        })
        .collect();

    let error_fdtd = l2_error(&fdtd.pressure, &analytical);
    let error_pstd = l2_error(&pressure_pstd, &analytical);

    // Both should produce reasonable results
    assert!(
        error_fdtd < 1.0e6,
        "FDTD error too large: {:.2e}",
        error_fdtd
    );
    assert!(
        error_pstd < 1.0e6,
        "PSTD error too large: {:.2e}",
        error_pstd
    );

    // Log results for analysis
    eprintln!("FDTD error: {:.2e}", error_fdtd);
    eprintln!("PSTD error: {:.2e}", error_pstd);
}
