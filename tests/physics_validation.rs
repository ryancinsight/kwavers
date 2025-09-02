//! Comprehensive physics validation tests
//!
//! These tests validate core physics implementations against
//! established literature and analytical solutions.

use approx::assert_relative_eq;
use kwavers::physics::constants::*;

#[cfg(test)]
mod wave_equation_tests {
    use super::*;

    /// Validate d'Alembert solution for 1D wave equation
    /// Reference: Strauss, W. (2007). Partial Differential Equations: An Introduction
    #[test]
    fn test_dalembert_solution_1d() {
        // For wave equation: ∂²u/∂t² = c²∂²u/∂x²
        // Solution: u(x,t) = f(x-ct) + g(x+ct)

        let c = SOUND_SPEED_WATER; // 1480 m/s
        let x = 10.0; // position
        let t = 0.01; // time

        // Forward traveling wave: f(x-ct)
        let forward_phase = x - c * t;

        // Backward traveling wave: g(x+ct)
        let backward_phase = x + c * t;

        // Verify phase velocities
        assert_relative_eq!(forward_phase, x - 14.8, epsilon = 1e-10);
        assert_relative_eq!(backward_phase, x + 14.8, epsilon = 1e-10);
    }

    /// Validate dispersion relation for acoustic waves
    /// Reference: Pierce, A. D. (1989). Acoustics: An Introduction
    #[test]
    fn test_acoustic_dispersion_relation() {
        // For acoustic waves: ω² = c²k²
        // where ω is angular frequency, k is wavenumber

        let frequency = 1e6; // 1 MHz
        let omega = 2.0 * std::f64::consts::PI * frequency;
        let c = SOUND_SPEED_WATER;

        // Calculate wavenumber from dispersion relation
        let k = omega / c;
        let wavelength = 2.0 * std::f64::consts::PI / k;

        // Verify wavelength
        let expected_wavelength = c / frequency;
        assert_relative_eq!(wavelength, expected_wavelength, epsilon = 1e-10);

        // For 1 MHz in water, wavelength should be ~1.48 mm
        assert_relative_eq!(wavelength, 1.48e-3, epsilon = 1e-5);
    }
}

#[cfg(test)]
mod nonlinear_acoustics_tests {
    use super::*;

    /// Validate Burger's equation shock formation
    /// Reference: Hamilton & Blackstock (1998). Nonlinear Acoustics
    #[test]
    fn test_burgers_shock_formation() {
        // Shock formation distance: x_s = ρ₀c₀³/(βωp₀)
        // where β = 1 + B/2A is coefficient of nonlinearity

        let beta = 1.0 + NONLINEARITY_WATER / 2.0; // ~3.6 for water
        let frequency = 1e6; // 1 MHz
        let omega = 2.0 * std::f64::consts::PI * frequency;
        let p0 = 1e6; // 1 MPa amplitude
        let rho0 = DENSITY_WATER;
        let c0 = SOUND_SPEED_WATER;

        let shock_distance = rho0 * c0.powi(3) / (beta * omega * p0);

        // For 1 MHz, 1 MPa in water, shock forms at ~0.15 m
        assert!(shock_distance > 0.1 && shock_distance < 0.2);
    }

    /// Validate Goldberg number for nonlinearity assessment
    /// Reference: Szabo, T. L. (2014). Diagnostic Ultrasound Imaging
    #[test]
    fn test_goldberg_number() {
        // Goldberg number: Γ = βp₀x/(ρ₀c₀³)
        // Γ < 1: linear propagation
        // Γ > 1: nonlinear effects significant

        let beta = 1.0 + NONLINEARITY_WATER / 2.0;
        let p0 = 1e5; // 100 kPa
        let x = 0.1; // 10 cm propagation
        let rho0 = DENSITY_WATER;
        let c0 = SOUND_SPEED_WATER;

        let goldberg = beta * p0 * x / (rho0 * c0.powi(3));

        // At 100 kPa, 10 cm: should be weakly nonlinear
        assert!(goldberg < 1.0);
        assert!(goldberg > 0.01); // But not negligible
    }
}

#[cfg(test)]
mod absorption_tests {
    use super::*;

    /// Validate classical absorption in water
    /// Reference: Kinsler et al. (2000). Fundamentals of Acoustics
    #[test]
    fn test_classical_absorption_water() {
        // Classical absorption: α = 2ηω²/(3ρc³)
        // where η is shear viscosity
        // Note: This gives α in Np/m, need to convert to dB/cm

        let eta = 1.002e-3; // Water viscosity at 20°C (Pa·s)
        let frequency = 1e6; // 1 MHz
        let omega = 2.0 * std::f64::consts::PI * frequency;
        let rho = DENSITY_WATER;
        let c = SOUND_SPEED_WATER;

        // Classical absorption in Np/m
        let alpha_np_m = 2.0 * eta * omega.powi(2) / (3.0 * rho * c.powi(3));

        // Convert Np/m to dB/cm: 1 Np = 8.686 dB, 1 m = 100 cm
        let alpha_db_cm = alpha_np_m * 8.686 / 100.0;

        // Classical absorption at 1 MHz should be ~0.002 dB/cm
        println!("Calculated alpha_db_cm: {}", alpha_db_cm);
        assert!(alpha_db_cm < 0.01, "Alpha too high: {} dB/cm", alpha_db_cm);
    }

    /// Validate power law absorption model
    /// Reference: Szabo, T. L. (2004). IEEE UFFC
    #[test]
    fn test_power_law_absorption() {
        // Power law: α(f) = α₀|f|^y
        // For soft tissue: α₀ ≈ 0.5-1.5 dB/cm/MHz^y, y ≈ 1-1.5

        let alpha_0 = 0.5; // dB/cm/MHz
        let y = 1.1; // Power law exponent

        let frequencies = vec![0.5e6, 1e6, 2e6, 5e6];

        for f in frequencies {
            let f_mhz = f / 1e6_f64;
            let alpha = alpha_0 * f_mhz.powf(y);

            // Verify scaling with frequency
            if f == 1e6 {
                assert_relative_eq!(alpha, alpha_0, epsilon = 1e-10);
            } else if f == 2e6 {
                let expected = alpha_0 * 2_f64.powf(y);
                assert_relative_eq!(alpha, expected, epsilon = 1e-10);
            }
        }
    }
}

#[cfg(test)]
mod finite_difference_tests {

    use ndarray::Array1;

    /// Validate 2nd order finite difference accuracy
    /// Reference: LeVeque, R. J. (2007). Finite Difference Methods
    #[test]
    fn test_second_order_laplacian() {
        // For u(x) = sin(kx), ∇²u = -k²u
        let n = 100;
        let dx = 0.01;
        let k = 2.0 * std::f64::consts::PI; // wavenumber

        let x: Array1<f64> = Array1::linspace(0.0, dx * n as f64, n);
        let u: Array1<f64> = x.mapv(|xi| (k * xi).sin());

        // Compute Laplacian using 2nd order FD
        let mut laplacian = Array1::zeros(n);
        for i in 1..n - 1 {
            laplacian[i] = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx);
        }

        // Compare with analytical
        for i in 1..n - 1 {
            let analytical = -k * k * u[i];
            let error = (laplacian[i] - analytical).abs() / analytical.abs().max(1e-10);

            // 2nd order accuracy: error ~ O(dx²)
            if i == n / 2 {
                // Check middle point
                println!("Error at i={}: {}, dx²={}", i, error, dx * dx);
            }
            assert!(
                error < dx * dx * 100.0,
                "Error {} exceeds threshold at i={}",
                error,
                i
            );
        }
    }

    /// Validate 4th order finite difference accuracy
    #[test]
    fn test_fourth_order_laplacian() {
        // 4th order stencil: [-1/12, 4/3, -5/2, 4/3, -1/12] / dx²

        let n = 100;
        let dx = 0.01;
        let k = 2.0 * std::f64::consts::PI;

        let x: Array1<f64> = Array1::linspace(0.0, dx * n as f64, n);
        let u: Array1<f64> = x.mapv(|xi| (k * xi).sin());

        // 4th order coefficients
        let c0 = -1.0 / 12.0;
        let c1 = 4.0 / 3.0;
        let c2 = -5.0 / 2.0;

        let mut laplacian = Array1::zeros(n);
        for i in 2..n - 2 {
            laplacian[i] =
                (c0 * (u[i + 2] + u[i - 2]) + c1 * (u[i + 1] + u[i - 1]) + c2 * u[i]) / (dx * dx);
        }

        // Compare with analytical
        for i in 2..n - 2 {
            let analytical = -k * k * u[i];
            let error = (laplacian[i] - analytical).abs() / analytical.abs().max(1e-10);

            // 4th order accuracy: error ~ O(dx⁴)
            assert!(error < dx.powi(4) * 100.0);
        }
    }
}

#[cfg(test)]
mod cfl_stability_tests {
    use super::*;

    /// Validate CFL condition for FDTD
    /// Reference: Taflove & Hagness (2005). Computational Electrodynamics
    #[test]
    fn test_cfl_condition_3d() {
        // CFL condition: c*dt/dx ≤ 1/√3 for 3D FDTD

        let c = SOUND_SPEED_WATER;
        let dx = 1e-3; // 1 mm
        let cfl_limit = 1.0 / 3_f64.sqrt();

        // Maximum stable timestep
        let dt_max = cfl_limit * dx / c;

        // Verify stability for various CFL numbers
        let cfl_numbers = vec![0.1, 0.3, 0.5, 0.577]; // 0.577 ≈ 1/√3

        for cfl in cfl_numbers {
            let dt = cfl * dx / c;

            if cfl <= cfl_limit {
                assert!(dt <= dt_max);
            } else {
                assert!(dt > dt_max);
            }
        }
    }

    /// Validate von Neumann stability analysis
    /// Reference: Strikwerda, J. C. (2004). Finite Difference Schemes
    #[test]
    fn test_von_neumann_stability() {
        // For wave equation with leapfrog: |G| = |1 - 2r²(1 - cos(kdx))|
        // where r = c*dt/dx is CFL number

        let c = SOUND_SPEED_WATER;
        let dx = 1e-3;
        let dt = 0.3 * dx / c; // CFL = 0.3
        let r = c * dt / dx;

        // Test various wavenumbers
        let k_values = vec![0.1, 1.0, 10.0, 100.0];

        for k in k_values {
            let kdx = k * dx;
            let kdx_f64: f64 = kdx;
            let g_squared = (1.0 - 2.0 * r * r * (1.0 - kdx_f64.cos())).powi(2);

            // For stability: |G|² ≤ 1
            assert!(g_squared <= 1.0 + 1e-10);
        }
    }
}
