//! Comprehensive physics validation tests
//!
//! These tests validate core physics implementations against
//! established literature and analytical solutions.

use eunomia::assert_relative_eq;
use kwavers_core::constants::{DENSITY_WATER, SOUND_SPEED_WATER, WATER_NONLINEARITY_B_A};
use kwavers_physics::analytical::wave::shock_formation_distance;

#[cfg(test)]
mod wave_equation_tests {
    use super::*;

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

        let beta = 1.0 + WATER_NONLINEARITY_B_A / 2.0; // ~3.6 for water
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
        // sigma = z / z_shock, where the shock-formation distance is
        // z_shock = rho c^3 / (beta omega p0). The frequency is the point: the
        // earlier version wrote `beta * p0 * x / (rho * c^3)` with no omega, an
        // expression carrying units of seconds rather than a dimensionless
        // ratio. It evaluated to 1.08e-8 against a lower bound of 0.01 and
        // could not have passed at any pressure.
        //
        // The distance now comes from `shock_formation_distance`, which is
        // where kwavers owns this relation (Blackstock 1966); the test asserts
        // the operating point rather than restating the formula.
        let beta = 1.0 + WATER_NONLINEARITY_B_A / 2.0;
        let p0 = 1e5; // 100 kPa
        let f0 = 1e6; // 1 MHz
        let z = 0.1; // 10 cm propagation

        let z_shock = shock_formation_distance(p0, f0, SOUND_SPEED_WATER, DENSITY_WATER, beta);
        let sigma = z / z_shock;

        // At 100 kPa and 1 MHz over 10 cm, water is weakly nonlinear: the wave
        // has developed measurable harmonic content but is far from shocked.
        assert!(
            sigma < 1.0,
            "sigma = {sigma:.4} is at or past shock formation; the case is meant              to sit in the pre-shock regime"
        );
        assert!(
            sigma > 0.01,
            "sigma = {sigma:.4} makes nonlinearity negligible; the case is meant              to exercise a regime where it is not"
        );
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
mod spectral_laplacian_tests {
    use kwavers_grid::Grid;
    use kwavers_solver::forward::nonlinear::kuznetsov::numerical::compute_laplacian;
    use leto::Array3;
    use std::f64::consts::PI;

    /// The solver's Laplacian is spectral, so it is exact on a resolved mode.
    ///
    /// This replaces two tests that hand-wrote 3- and 5-point stencils in the
    /// test file and compared them to `-k^2 u`. They exercised no library code,
    /// and both were broken besides: the sample spacing was `dx * n / (n - 1)`
    /// while the stencil divided by `dx`, a 2% disagreement, and the tolerance
    /// `error < dx * dx * 100.0` compared a dimensionless relative error to a
    /// bound carrying units of m^2 -- the truncation error for `u = sin(kx)` is
    /// `k^2 dx^2 / 12` relative, so the `k^2 / 12` factor was missing entirely.
    ///
    /// kwavers is pseudospectral: `compute_laplacian` transforms, multiplies by
    /// `-|k|^2`, and transforms back. For a sinusoid at an exact grid frequency
    /// on a periodic domain that is not an approximation -- the mode is an
    /// eigenfunction of the discrete operator, so the only error is FFT
    /// round-off.
    ///
    /// The bound follows from that. Round-off through a radix-2 transform pair
    /// of length `N` accumulates as O(sqrt(log2 N)) * eps; at `N = 32` and
    /// `eps = 2.2e-16` that predicts a few times `1e-15` relative, and the
    /// measured worst case is `7.97e-15`. `1e-11` therefore sits about 1250x
    /// above round-off.
    ///
    /// It sits far below a wrong operator. At this resolution `k0 * dx = 0.785`,
    /// so a second-order stencil's `(k0 dx)^2 / 12` truncation error would be
    /// about 5% -- nine orders above the bound. The test separates a spectral
    /// operator from a finite-difference one, which is the distinction it
    /// exists to make, and it was confirmed to fail when the bound was dropped
    /// below the measured round-off rather than assumed to be live.
    #[test]
    fn spectral_laplacian_matches_the_analytical_eigenvalue() {
        const RELATIVE_TOLERANCE: f64 = 1.0e-11;
        let (nx, ny, nz) = (32, 8, 8);
        let dx = 1.0e-4;
        let grid = Grid::new(nx, ny, nz, dx, dx, dx).expect("reference grid");

        // Four full periods across the x extent: an exact grid frequency, so
        // the mode is periodic and representable without spectral leakage. A
        // frequency between grid modes would leak and the exactness claim would
        // not hold -- that is a property of the sampling, not of the operator.
        let modes = 4.0;
        let k0 = 2.0 * PI * modes / (nx as f64 * dx);

        let field = Array3::from_shape_fn([nx, ny, nz], |[i, _, _]| (k0 * i as f64 * dx).sin());
        let laplacian = compute_laplacian(&field, &grid);

        let scale = k0 * k0;
        let mut worst = 0.0_f64;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let analytical = -scale * field[[i, j, k]];
                    // Normalised by the field's amplitude rather than by the
                    // local value: near a zero crossing the pointwise relative
                    // error is unbounded for any operator, so dividing by it
                    // measures the grid, not the Laplacian.
                    let error = (laplacian[[i, j, k]] - analytical).abs() / scale;
                    worst = worst.max(error);
                }
            }
        }

        assert!(
            worst < RELATIVE_TOLERANCE,
            "spectral Laplacian departs from -k^2 u by {worst:.3e} relative,              above the {RELATIVE_TOLERANCE:.0e} round-off bound; a departure this              large means the operator is not spectral on a resolved mode"
        );
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
