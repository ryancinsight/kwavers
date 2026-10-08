//! CylindricalKZKConfig, CylindricalKZKSolver, and thomas_solve.

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::constants::{
    ACOUSTIC_ABSORPTION_TISSUE, DENSITY_WATER_NOMINAL, REFERENCE_FREQUENCY_HZ,
};
use kwavers_math::fft::Complex64;
use kwavers_physics::acoustics::wave_propagation::nonlinear::kzk::CylindricalKZKSolverTrait;
use leto::{Array1, Array2};
use moirai_parallel::{enumerate_mut_with, Adaptive};
use tracing::warn;

use super::super::DiffractionScheme;

/// Configuration for the axisymmetric KZK solver.
#[derive(Debug, Clone)]
pub struct CylindricalKZKConfig {
    /// Radial grid points.
    pub nr: usize,
    /// Axial propagation steps.
    pub nz: usize,
    /// Radial spacing (m).
    pub dr: f64,
    /// Axial spacing (m).
    pub dz: f64,
    /// Retarded-time spacing (s).
    pub dt: f64,
    /// Retarded-time samples.
    pub nt: usize,
    /// Small-signal sound speed (m/s).
    pub c0: f64,
    /// Ambient density (kg/m³).
    pub rho0: f64,
    /// Nonlinearity ratio B/A.
    pub b_over_a: f64,
    /// Absorption coefficient (dB/cm/MHz^y).
    pub alpha0: f64,
    /// Power-law exponent.
    pub alpha_power: f64,
    /// Enable diffraction sub-steps.
    pub include_diffraction: bool,
    /// Enable absorption sub-steps.
    pub include_absorption: bool,
    /// Enable nonlinear sub-steps.
    pub include_nonlinearity: bool,
    /// Operating frequency (Hz).
    pub frequency: f64,
    /// Diffraction scheme selector. Padé variants are not supported in the
    /// cylindrical solver because the radial Crank-Nicolson march replaces the
    /// transverse spectral propagator.
    pub diffraction_scheme: DiffractionScheme,
}

impl Default for CylindricalKZKConfig {
    fn default() -> Self {
        Self {
            nr: 256,
            nz: 256,
            dr: 0.5e-3,
            dz: 0.5e-3,
            dt: 10e-9,
            nt: 1000,
            c0: kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE,
            rho0: DENSITY_WATER_NOMINAL,
            b_over_a: 5.0,
            alpha0: ACOUSTIC_ABSORPTION_TISSUE,
            alpha_power: 1.1,
            include_diffraction: true,
            include_absorption: true,
            include_nonlinearity: true,
            frequency: REFERENCE_FREQUENCY_HZ,
            diffraction_scheme: DiffractionScheme::Parabolic,
        }
    }
}

use super::operators::{CylindricalAbsorption, CylindricalNonlinear};

/// Axisymmetric KZK solver with radial Crank-Nicolson diffraction.
#[derive(Debug)]
pub struct CylindricalKZKSolver {
    pub(crate) config: CylindricalKZKConfig,
    pub(crate) pressure: Array2<Complex64>,
    pub(crate) pressure_prev: Array2<Complex64>,
    pub(crate) r: Vec<f64>,
    pub(crate) absorption: CylindricalAbsorption,
    pub(crate) nonlinear: CylindricalNonlinear,
    current_z_step: usize,
    current_time: f64,
}

impl CylindricalKZKSolver {
    /// Create a new cylindrical KZK solver.
    ///
    /// ## Contract
    ///
    /// The solver stores `p(r, τ)` as a complex field so that the implicit
    /// radial diffraction step can conserve phase and energy in the same
    /// complex-field sense as the Cartesian spectral KZK solver.
    ///
    /// ## References
    ///
    /// - Lee Y-S, Hamilton MF (1995). J. Acoust. Soc. Am. 97(2), 906–917.
    /// - Strang G (1968). SIAM J. Numer. Anal. 5(3), 506–517.
    ///
    /// # Errors
    ///
    /// Returns an error when grid sizes, spacings, or supported diffraction
    /// scheme constraints are violated.
    #[must_use]
    pub fn new(config: CylindricalKZKConfig) -> Result<Self, String> {
        if config.nr < 2 || config.nz < 2 || config.nt < 2 {
            return Err("Cylindrical KZK grid dimensions must be at least 2".to_owned());
        }
        if config.dr <= 0.0 || config.dz <= 0.0 || config.dt <= 0.0 {
            return Err("Cylindrical KZK spacings must be positive".to_owned());
        }
        if config.c0 <= 0.0 || config.rho0 <= 0.0 {
            return Err("Cylindrical KZK material parameters must be positive".to_owned());
        }
        match config.diffraction_scheme {
            DiffractionScheme::Parabolic | DiffractionScheme::WideAngle => {}
            DiffractionScheme::Pade11 | DiffractionScheme::Pade22 => {
                return Err(
                    "CylindricalKZKSolver supports only Parabolic or WideAngle diffraction schemes"
                        .to_owned(),
                );
            }
        }
        if config.diffraction_scheme == DiffractionScheme::WideAngle {
            warn!(
                "CylindricalKZKSolver currently uses the axisymmetric Crank-Nicolson radial Laplacian march for cylindrical diffraction"
            );
        }

        let pressure = Array2::<Complex64>::zeros((config.nr, config.nt));
        let pressure_prev = Array2::<Complex64>::zeros((config.nr, config.nt));
        let r = (0..config.nr)
            .map(|i| (i as f64 + 0.5) * config.dr)
            .collect::<Vec<_>>();
        let absorption = CylindricalAbsorption::new(&config);
        let nonlinear = CylindricalNonlinear::new(&config);

        Ok(Self {
            config,
            pressure,
            pressure_prev,
            r,
            absorption,
            nonlinear,
            current_z_step: 0,
            current_time: 0.0,
        })
    }

    /// March one axial step using Strang splitting.
    pub fn step(&mut self) {
        let dz = self.config.dz;

        if self.config.include_diffraction {
            self.apply_diffraction(dz * 0.5);
        }
        if self.config.include_absorption {
            self.apply_absorption(dz * 0.5);
        }
        if self.config.include_nonlinearity {
            self.apply_nonlinearity(dz);
        }
        if self.config.include_absorption {
            self.apply_absorption(dz * 0.5);
        }
        if self.config.include_diffraction {
            self.apply_diffraction(dz * 0.5);
        }

        self.pressure_prev.assign(&self.pressure);
        self.current_z_step += 1;
        self.current_time += dz / self.config.c0;
    }

    /// Propagate the cylindrical field forward by `n_steps` axial planes.
    ///
    /// # Errors
    ///
    /// Returns an error when `n_steps` exceeds the configured axial extent.
    pub fn solve(&mut self, n_steps: usize) -> Result<(), String> {
        if n_steps > self.config.nz {
            return Err(format!(
                "CylindricalKZKSolver::solve: n_steps={n_steps} exceeds config.nz={}",
                self.config.nz
            ));
        }

        for _ in 0..n_steps {
            self.step();
        }

        Ok(())
    }

    /// Set a time-harmonic axisymmetric source profile at `z = 0`.
    ///
    /// # Panics
    ///
    /// Panics if `source.len() != self.config.nr`.
    pub fn set_source(&mut self, source: Array1<f64>, frequency: f64) {
        assert_eq!(
            source.len(),
            self.config.nr,
            "cylindrical source length must equal nr"
        );

        self.config.frequency = frequency;
        self.absorption = CylindricalAbsorption::new(&self.config);
        self.nonlinear = CylindricalNonlinear::new(&self.config);

        let omega = TWO_PI * frequency;
        for t in 0..self.config.nt {
            let temporal = (omega * t as f64 * self.config.dt).sin();
            for i in 0..self.config.nr {
                self.pressure[[i, t]] = Complex64::new(source[i] * temporal, 0.0);
            }
        }

        self.pressure_prev.assign(&self.pressure);
    }

    /// Return the RMS pressure profile versus radius.
    #[must_use]
    pub fn current_field(&self) -> Array1<f64> {
        let nt = self.config.nt;
        let nt_f64 = nt as f64;
        let mut rms = Array1::<f64>::zeros(self.config.nr);
        let pressure = self
            .pressure
            .as_slice()
            .expect("invariant: cylindrical KZK pressure is standard-layout");
        let rms_slice = rms
            .as_slice_mut()
            .expect("invariant: cylindrical KZK RMS output is standard-layout");
        enumerate_mut_with::<Adaptive, _, _>(rms_slice, |idx, out| {
            let base = idx * nt;
            let sum_sq: f64 = (0..nt).map(|t| pressure[base + t].re.powi(2)).sum();
            *out = (sum_sq / nt_f64).sqrt();
        });
        rms
    }

    pub(crate) fn apply_diffraction(&mut self, step_size: f64) {
        let nr = self.config.nr;
        let dr2 = self.config.dr * self.config.dr;
        let k0 = TWO_PI * self.config.frequency / self.config.c0;
        let sigma = Complex64::new(0.0, step_size / (4.0 * k0));

        let mut lower_d = vec![0.0_f64; nr - 1];
        let diag_d = vec![-2.0 / dr2; nr];
        let mut upper_d = vec![0.0_f64; nr];

        upper_d[0] = 2.0 / dr2;
        for i in 1..nr - 1 {
            let ratio = self.config.dr / (2.0 * self.r[i]);
            lower_d[i - 1] = (1.0 - ratio) / dr2;
            upper_d[i] = (1.0 + ratio) / dr2;
        }
        let last_ratio = self.config.dr / (2.0 * self.r[nr - 1]);
        lower_d[nr - 2] = (1.0 - last_ratio) / dr2;

        let identity = Complex64::new(1.0, 0.0);
        let mut lower_a = vec![Complex64::new(0.0, 0.0); nr - 1];
        let mut diag_a = vec![identity; nr];
        let mut upper_a = vec![Complex64::new(0.0, 0.0); nr];
        let mut lower_b = vec![Complex64::new(0.0, 0.0); nr - 1];
        let mut diag_b = vec![identity; nr];
        let mut upper_b = vec![Complex64::new(0.0, 0.0); nr];

        diag_a[0] = identity - sigma * diag_d[0];
        diag_b[0] = identity + sigma * diag_d[0];
        upper_a[0] = -sigma * upper_d[0];
        upper_b[0] = sigma * upper_d[0];

        for i in 1..nr {
            diag_a[i] = identity - sigma * diag_d[i];
            diag_b[i] = identity + sigma * diag_d[i];
            lower_a[i - 1] = -sigma * lower_d[i - 1];
            lower_b[i - 1] = sigma * lower_d[i - 1];
            if i < nr - 1 {
                upper_a[i] = -sigma * upper_d[i];
                upper_b[i] = sigma * upper_d[i];
            }
        }

        let mut rhs = vec![Complex64::new(0.0, 0.0); nr];
        for t in 0..self.config.nt {
            for i in 0..nr {
                let mut value = diag_b[i] * self.pressure[[i, t]];
                if i > 0 {
                    value += lower_b[i - 1] * self.pressure[[i - 1, t]];
                }
                if i + 1 < nr {
                    value += upper_b[i] * self.pressure[[i + 1, t]];
                }
                rhs[i] = value;
            }

            thomas_solve(&lower_a, &diag_a, &upper_a, &mut rhs);

            for i in 0..nr {
                self.pressure[[i, t]] = rhs[i];
            }
        }
    }

    fn apply_absorption(&mut self, step_size: f64) {
        self.absorption.apply(&mut self.pressure, step_size);
    }

    fn apply_nonlinearity(&mut self, step_size: f64) {
        self.nonlinear
            .apply(&mut self.pressure, &self.pressure_prev, step_size);
    }
}

impl CylindricalKZKSolverTrait for CylindricalKZKSolver {}

fn thomas_solve(
    lower: &[Complex64],
    diag: &[Complex64],
    upper: &[Complex64],
    rhs: &mut [Complex64],
) {
    let n = diag.len();
    let mut c_prime = vec![Complex64::new(0.0, 0.0); n];
    let mut d_prime = vec![Complex64::new(0.0, 0.0); n];

    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let m = diag[i] - lower[i - 1] * c_prime[i - 1];
        c_prime[i] = if i < n - 1 {
            upper[i] / m
        } else {
            Complex64::new(0.0, 0.0)
        };
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / m;
    }

    rhs[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        rhs[i] = d_prime[i] - c_prime[i] * rhs[i + 1];
    }
}
