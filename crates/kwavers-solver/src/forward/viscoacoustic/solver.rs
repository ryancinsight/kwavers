//! N-dimensional pseudospectral memory-variable viscoacoustic solver.
//!
//! One canonical implementation covers 1-D, 2-D, and 3-D: a `(n,1,1)` grid is
//! 1-D, `(nx,ny,1)` is 2-D, and `(nx,ny,nz)` is full 3-D — the spectral
//! derivative along a singleton axis is identically zero, so the lower-D cases
//! reduce exactly with no special-casing.

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::{get_fft_for_grid, Complex64, Fft3d};
use kwavers_medium::viscoelastic::GeneralizedMaxwellModel;
use ndarray::{Array3, Axis, Zip};
use std::sync::Arc;

/// One relaxation arm with its precomputed per-voxel exponential-integrator
/// coefficients (uniform fields for a homogeneous medium).
#[derive(Debug, Clone)]
struct Arm {
    /// `e^{-Δt/τₗ(x)}` (decay over one step).
    decay: Array3<f64>,
    /// `−ΔMₗ(x)·τₗ(x)·(1 − e^{-Δt/τₗ})` — coefficient of `∇·v` in the σ update.
    gain: Array3<f64>,
    /// `1/τₗ(x)` \[s⁻¹] — for the trapezoidal pressure contribution.
    inv_tau: Array3<f64>,
}

/// Build an arm's exponential-integrator coefficient fields from per-voxel
/// relaxation strength `ΔMₗ(x)` and time `τₗ(x)`.
fn build_arm(delta_m: &Array3<f64>, tau: &Array3<f64>, dt: f64) -> Arm {
    let decay = tau.mapv(|t| (-dt / t).exp());
    let inv_tau = tau.mapv(|t| 1.0 / t);
    let gain = Zip::from(delta_m)
        .and(tau)
        .and(&decay)
        .map_collect(|&dm, &t, &dc| -dm * t * (1.0 - dc));
    Arm {
        decay,
        gain,
        inv_tau,
    }
}

/// Time-domain memory-variable viscoacoustic solver (1-D/2-D/3-D, pseudospectral).
///
/// Construct from a [`GeneralizedMaxwellModel`] (the relaxation spectrum) or from
/// raw moduli, then call [`Self::step`] to advance the velocity–pressure state.
#[derive(Clone)]
pub struct ViscoacousticMemorySolver {
    nx: usize,
    ny: usize,
    nz: usize,
    cell_volume: f64, // dx·dy·dz
    dt: f64,
    /// Per-voxel `1/ρ(x)` \[m³·kg⁻¹] for the velocity update.
    inv_rho: Array3<f64>,
    /// Per-voxel unrelaxed (instantaneous) modulus `M_U(x) = M_∞(x) + Σ ΔMₗ(x)` \[Pa].
    m_u: Array3<f64>,
    /// Per-voxel equilibrium (relaxed) modulus `M_∞(x)` \[Pa] — potential-energy norm.
    m_inf: Array3<f64>,
    /// Maximum unrelaxed sound speed over the grid \[m·s⁻¹] — the CFL reference.
    max_unrelaxed_speed: f64,
    arms: Vec<Arm>,

    // Spectral derivative: apollo's batched, cache-tiled, parallel per-axis 3-D
    // FFT (forward_axis → ·ik → inverse_axis) reusing one complex scratch.
    fft: Arc<Fft3d>,
    kx: Vec<f64>,
    ky: Vec<f64>,
    kz: Vec<f64>,
    cbuf: Array3<Complex64>,

    // State.
    p: Array3<f64>,
    vx: Array3<f64>,
    vy: Array3<f64>,
    vz: Array3<f64>,
    sigma: Vec<Array3<f64>>, // one memory field per arm

    // Preallocated derivative buffers (gx is reused as the divergence ∇·v and gy
    // as the relaxation accumulator after the velocity-divergence pass).
    gx: Array3<f64>,
    gy: Array3<f64>,
    gz: Array3<f64>,

    // Optional absorbing boundary: per-cell multiplicative decay `exp(-γ Δt)`
    // applied to `p` and `v` each step, with `γ` ramping up inside the boundary
    // layer and zero in the interior (`None` ⇒ periodic, non-absorbing).
    damping_decay: Option<Array3<f64>>,

    // Driven-simulation I/O.
    step_count: usize,
    /// Additive (soft) pressure sources: `(grid index, time signal)`.
    pressure_sources: Vec<((usize, usize, usize), Vec<f64>)>,
    /// Pressure-sensor grid indices and their recorded time traces.
    pressure_sensors: Vec<(usize, usize, usize)>,
    sensor_record: Vec<Vec<f64>>,
}

impl std::fmt::Debug for ViscoacousticMemorySolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViscoacousticMemorySolver")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .field("dt", &self.dt)
            .field("max_unrelaxed_speed", &self.max_unrelaxed_speed)
            .field("arms", &self.arms.len())
            .finish()
    }
}

impl ViscoacousticMemorySolver {
    /// Build from raw parameters: grid `(nx,ny,nz)` with spacings `(dx,dy,dz)`,
    /// time step `dt`, density `ρ`, equilibrium modulus `M_∞`, and relaxation
    /// arms `(ΔMₗ, τₗ)`. An empty arm list yields the lossless wave equation.
    /// # Errors
    /// - Any zero dimension, non-positive `dx`/`dy`/`dz`/`dt`/`ρ`/`M_∞`, or a
    ///   non-positive arm parameter.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        rho: f64,
        m_inf: f64,
        arms: &[(f64, f64)],
    ) -> KwaversResult<Self> {
        if nx == 0
            || ny == 0
            || nz == 0
            || dx <= 0.0
            || dy <= 0.0
            || dz <= 0.0
            || dt <= 0.0
            || rho <= 0.0
            || m_inf <= 0.0
        {
            return Err(KwaversError::InvalidInput(
                "viscoacoustic solver requires positive grid, spacings, dt, ρ, M_∞".to_owned(),
            ));
        }
        if arms.iter().any(|&(dm, tau)| dm <= 0.0 || tau <= 0.0) {
            return Err(KwaversError::InvalidInput(
                "relaxation arms require ΔM>0 and τ>0".to_owned(),
            ));
        }

        // Homogeneous medium: broadcast the scalar parameters to uniform fields
        // and delegate to the per-voxel assembler.
        let shape = (nx, ny, nz);
        let inv_rho = Array3::from_elem(shape, 1.0 / rho);
        let m_inf_field = Array3::from_elem(shape, m_inf);
        let arm_fields: Vec<Arm> = arms
            .iter()
            .map(|&(dm, tau)| {
                build_arm(
                    &Array3::from_elem(shape, dm),
                    &Array3::from_elem(shape, tau),
                    dt,
                )
            })
            .collect();
        Ok(Self::assemble(
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
            inv_rho,
            m_inf_field,
            arm_fields,
        ))
    }

    /// Build a **heterogeneous** medium from per-voxel fields: density `ρ(x)`,
    /// equilibrium modulus `M_∞(x)`, and relaxation arms `(ΔMₗ(x), τₗ(x))` (all of
    /// grid shape). This lets a CT-derived tissue model (§4.5) drive the broadband
    /// solver with spatially-varying viscoacoustic properties.
    /// # Errors
    /// - Any field shape ≠ `(nx,ny,nz)`, a non-positive `ρ`/`M_∞`/`ΔM`/`τ`, or
    ///   non-positive grid/spacing/`dt`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_heterogeneous(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        rho: &Array3<f64>,
        m_inf: &Array3<f64>,
        arms: &[(Array3<f64>, Array3<f64>)],
    ) -> KwaversResult<Self> {
        let shape = [nx, ny, nz];
        let ok_shape = |a: &Array3<f64>| a.shape() == shape;
        if nx == 0 || ny == 0 || nz == 0 || dx <= 0.0 || dy <= 0.0 || dz <= 0.0 || dt <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "viscoacoustic solver requires positive grid, spacings, dt".to_owned(),
            ));
        }
        if !ok_shape(rho)
            || !ok_shape(m_inf)
            || rho.iter().any(|&r| r <= 0.0)
            || m_inf.iter().any(|&m| m <= 0.0)
        {
            return Err(KwaversError::InvalidInput(
                "ρ and M_∞ fields must be grid-shaped and positive".to_owned(),
            ));
        }
        if arms.iter().any(|(dm, tau)| {
            !ok_shape(dm)
                || !ok_shape(tau)
                || dm.iter().any(|&v| v <= 0.0)
                || tau.iter().any(|&v| v <= 0.0)
        }) {
            return Err(KwaversError::InvalidInput(
                "relaxation arm fields must be grid-shaped with ΔM>0 and τ>0".to_owned(),
            ));
        }

        let inv_rho = rho.mapv(|r| 1.0 / r);
        let arm_fields: Vec<Arm> = arms
            .iter()
            .map(|(dm, tau)| build_arm(dm, tau, dt))
            .collect();
        Ok(Self::assemble(
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
            inv_rho,
            m_inf.clone(),
            arm_fields,
        ))
    }

    /// Allocate state/scratch and assemble the solver from prepared per-voxel
    /// `inv_rho`, `m_inf`, and arm coefficient fields. `M_U = M_∞ + Σ ΔMₗ` is
    /// recovered from the arm gains; the CFL speed is the grid max of `√(M_U/ρ)`.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        inv_rho: Array3<f64>,
        m_inf: Array3<f64>,
        arms: Vec<Arm>,
    ) -> Self {
        let shape = (nx, ny, nz);
        // M_U(x) = M_∞(x) + Σ ΔMₗ(x); recover ΔMₗ = −gain / (τ(1−decay)) = −gain·inv_tau/(1−decay).
        let mut m_u = m_inf.clone();
        for arm in &arms {
            Zip::from(&mut m_u)
                .and(&arm.gain)
                .and(&arm.decay)
                .and(&arm.inv_tau)
                .for_each(|mu, &gain, &decay, &inv_tau| {
                    *mu += -gain * inv_tau / (1.0 - decay);
                });
        }
        let max_unrelaxed_speed = Zip::from(&m_u)
            .and(&inv_rho)
            .fold(0.0_f64, |acc, &mu, &ir| acc.max((mu * ir).sqrt()));

        Self {
            nx,
            ny,
            nz,
            cell_volume: dx * dy * dz,
            dt,
            inv_rho,
            m_u,
            m_inf,
            max_unrelaxed_speed,
            sigma: vec![Array3::zeros(shape); arms.len()],
            arms,
            fft: get_fft_for_grid(nx, ny, nz),
            kx: fft_wavenumbers(nx, dx),
            ky: fft_wavenumbers(ny, dy),
            kz: fft_wavenumbers(nz, dz),
            cbuf: Array3::from_elem(shape, Complex64::new(0.0, 0.0)),
            p: Array3::zeros(shape),
            vx: Array3::zeros(shape),
            vy: Array3::zeros(shape),
            vz: Array3::zeros(shape),
            gx: Array3::zeros(shape),
            gy: Array3::zeros(shape),
            gz: Array3::zeros(shape),
            damping_decay: None,
            step_count: 0,
            pressure_sources: Vec::new(),
            pressure_sensors: Vec::new(),
            sensor_record: Vec::new(),
        }
    }

    /// Enable an **absorbing boundary layer** (sponge) of `thickness` cells on
    /// every face whose axis is long enough to hold it. Outgoing waves entering
    /// the layer are damped before they reach (and wrap around) the periodic
    /// boundary, suppressing artificial reflections.
    ///
    /// `gamma_max` \[s⁻¹] is the peak damping rate at the outermost cell; the
    /// rate ramps in as a quadratic profile `γ(d) = γ_max ((L-d)/L)²` over the
    /// layer depth `d ∈ [0, L)` (zero in the interior), summed across axes so
    /// corners damp in every direction. A smooth ramp keeps layer reflection low.
    /// Calling again rebuilds the profile; `thickness = 0` disables it.
    pub fn enable_absorbing_layer(&mut self, thickness: usize, gamma_max: f64) {
        if thickness == 0 || gamma_max <= 0.0 {
            self.damping_decay = None;
            return;
        }
        // Per-axis ramp: γ contribution at index `i` along an axis of extent `n`.
        let ramp = |i: usize, n: usize| -> f64 {
            if n <= 2 * thickness {
                return 0.0; // axis too short to host the layer (e.g. singleton)
            }
            let l = thickness as f64;
            let depth = if i < thickness {
                (thickness - i) as f64 // distance from the low boundary
            } else if i >= n - thickness {
                (i - (n - thickness) + 1) as f64 // distance from the high boundary
            } else {
                0.0
            };
            let frac = depth / l;
            gamma_max * frac * frac
        };

        let dt = self.dt;
        let decay = Array3::from_shape_fn((self.nx, self.ny, self.nz), |(i, j, k)| {
            let gamma = ramp(i, self.nx) + ramp(j, self.ny) + ramp(k, self.nz);
            (-gamma * dt).exp()
        });
        self.damping_decay = Some(decay);
    }

    /// 1-D convenience constructor (`ny = nz = 1`).
    /// # Errors
    /// - Propagates [`Self::new`] validation failures.
    pub fn new_1d(
        n: usize,
        dx: f64,
        dt: f64,
        rho: f64,
        m_inf: f64,
        arms: &[(f64, f64)],
    ) -> KwaversResult<Self> {
        Self::new(n, 1, 1, dx, 1.0, 1.0, dt, rho, m_inf, arms)
    }

    /// Build from a [`GeneralizedMaxwellModel`] (its `M_∞`, arms, and density)
    /// plus the grid and time step.
    /// # Errors
    /// - Propagates [`Self::new`] validation failures.
    #[allow(clippy::too_many_arguments)]
    pub fn from_generalized_maxwell(
        model: &GeneralizedMaxwellModel,
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
    ) -> KwaversResult<Self> {
        Self::new(
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
            model.density(),
            model.equilibrium_modulus(),
            model.arms(),
        )
    }

    /// Build from a **CT-derived power-law medium**: per-voxel density `ρ(x)`,
    /// sound speed `c(x)`, and target amplitude attenuation `α(x)` \[Np·m⁻¹] at
    /// the reference frequency `f_ref`, with a (uniform) power-law exponent `y`.
    ///
    /// A relaxation spectrum reproducing this absorption is fitted with a **shared**
    /// log-spaced relaxation-time grid `τₗ` over `[f_min, f_max]` and Fung (1993)
    /// weights `ΔMₗ ∝ τₗ^{1-y}`, scaled **per voxel** so the spectrum's analytic
    /// `α(ω_ref)` matches the target (the equilibrium modulus is `M_∞ = ρ c²`).
    /// This is the bridge from the `HuAcousticModel`/`CtMediumBuilder` tissue
    /// pipeline (book §4.5) to the broadband solver.
    /// # Errors
    /// - Field shape ≠ grid, non-positive `ρ`/`c`/`α`, `y ∉ (0,2)`, bad band, or
    ///   `n_arms == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_power_law_fields(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        rho: &Array3<f64>,
        c: &Array3<f64>,
        alpha_np_m: &Array3<f64>,
        y: f64,
        f_min: f64,
        f_max: f64,
        n_arms: usize,
        f_ref: f64,
    ) -> KwaversResult<Self> {
        let shape = [nx, ny, nz];
        let ok = |a: &Array3<f64>, positive: bool| {
            a.shape() == shape && (!positive || a.iter().all(|&v| v > 0.0))
        };
        if !ok(rho, true) || !ok(c, true) || !ok(alpha_np_m, false) {
            return Err(KwaversError::InvalidInput(
                "ρ, c, α fields must be grid-shaped (ρ, c positive)".to_owned(),
            ));
        }
        if !(y > 0.0 && y < 2.0 && f_min > 0.0 && f_max > f_min && n_arms >= 1 && f_ref > 0.0) {
            return Err(KwaversError::InvalidInput(
                "require 0<y<2, 0<f_min<f_max, n_arms≥1, f_ref>0".to_owned(),
            ));
        }

        // Shared log-spaced relaxation times and normalised Fung weights (Σ=1).
        let tau_max = 1.0 / (TWO_PI * f_min);
        let tau_min = 1.0 / (TWO_PI * f_max);
        let (ln_lo, ln_hi) = (tau_min.ln(), tau_max.ln());
        let taus: Vec<f64> = (0..n_arms)
            .map(|l| {
                let frac = if n_arms == 1 {
                    0.5
                } else {
                    l as f64 / (n_arms - 1) as f64
                };
                ((ln_hi - ln_lo) * frac + ln_lo).exp()
            })
            .collect();
        let raw: Vec<f64> = taus.iter().map(|&t| t.powf(1.0 - y)).collect();
        let raw_sum: f64 = raw.iter().sum();
        let base: Vec<f64> = raw.iter().map(|&r| r / raw_sum).collect(); // Σ = 1

        // Per-voxel: M_∞ = ρc², calibrate the total relaxation strength so the
        // spectrum's α(ω_ref) equals the target (α scales ~linearly with strength).
        let omega_ref = TWO_PI * f_ref;
        let m_inf = Zip::from(rho).and(c).map_collect(|&r, &cc| r * cc * cc);
        let mut scale = Array3::<f64>::zeros((nx, ny, nz));
        Zip::from(&mut scale)
            .and(rho)
            .and(&m_inf)
            .and(alpha_np_m)
            .for_each(|sc, &r, &mi, &a_target| {
                // α for unit total strength (weights = `base`, summing to 1 Pa).
                let a_unit = relaxation_attenuation(omega_ref, r, mi, &base, &taus);
                *sc = if a_unit > 0.0 && a_target > 0.0 {
                    a_target / a_unit
                } else {
                    0.0
                };
            });

        // Arm fields: ΔMₗ(x) = scale(x)·baseₗ, τₗ uniform.
        let arms: Vec<(Array3<f64>, Array3<f64>)> = (0..n_arms)
            .map(|l| {
                let dm = scale.mapv(|s| (s * base[l]).max(f64::MIN_POSITIVE));
                let tau = Array3::from_elem((nx, ny, nz), taus[l]);
                (dm, tau)
            })
            .collect();

        Self::new_heterogeneous(nx, ny, nz, dx, dy, dz, dt, rho, &m_inf, &arms)
    }

    /// Maximum unrelaxed (high-frequency) sound speed `max √(M_U(x)/ρ(x))`
    /// \[m·s⁻¹] over the grid — the CFL reference speed.
    #[must_use]
    pub fn unrelaxed_speed(&self) -> f64 {
        self.max_unrelaxed_speed
    }

    /// Pressure field \[Pa].
    #[must_use]
    pub fn pressure(&self) -> &Array3<f64> {
        &self.p
    }

    /// Overwrite the pressure field (shape must match the grid); resets velocity
    /// and memory variables to zero. Used to set an initial condition.
    /// # Errors
    /// - Shape mismatch.
    pub fn set_pressure(&mut self, p: &Array3<f64>) -> KwaversResult<()> {
        if p.shape() != [self.nx, self.ny, self.nz] {
            return Err(KwaversError::InvalidInput(
                "pressure shape must equal the grid".to_owned(),
            ));
        }
        self.p.assign(p);
        self.vx.fill(0.0);
        self.vy.fill(0.0);
        self.vz.fill(0.0);
        for s in &mut self.sigma {
            s.fill(0.0);
        }
        // Restart the simulation clock and sensor traces (sources/sensors kept).
        self.step_count = 0;
        for trace in &mut self.sensor_record {
            trace.clear();
        }
        Ok(())
    }

    /// Register a **soft (additive) pressure source** at `index` with a per-step
    /// time `signal`: `p[index] += signal[step]` while `step < signal.len()`.
    /// # Errors
    /// - `index` out of grid bounds.
    pub fn add_pressure_source(
        &mut self,
        index: (usize, usize, usize),
        signal: Vec<f64>,
    ) -> KwaversResult<()> {
        self.check_index(index)?;
        self.pressure_sources.push((index, signal));
        Ok(())
    }

    /// Register a pressure sensor at `index`; [`Self::step`] appends `p[index]`
    /// to its trace each step. Returns the sensor id (its index in the record).
    /// # Errors
    /// - `index` out of grid bounds.
    pub fn add_pressure_sensor(&mut self, index: (usize, usize, usize)) -> KwaversResult<usize> {
        self.check_index(index)?;
        self.pressure_sensors.push(index);
        self.sensor_record.push(Vec::new());
        Ok(self.pressure_sensors.len() - 1)
    }

    /// Recorded pressure time trace for sensor `id`.
    #[must_use]
    pub fn sensor_trace(&self, id: usize) -> &[f64] {
        &self.sensor_record[id]
    }

    fn check_index(&self, (i, j, k): (usize, usize, usize)) -> KwaversResult<()> {
        if i < self.nx && j < self.ny && k < self.nz {
            Ok(())
        } else {
            Err(KwaversError::InvalidInput(
                "source/sensor index out of grid bounds".to_owned(),
            ))
        }
    }

    /// Acoustic energy `Σ [p²/(2M_∞) + ρ|v|²/2] ΔV` \[J]. Conserved (to leapfrog
    /// round-off) for the lossless medium; decays monotonically with relaxation.
    #[must_use]
    pub fn energy(&self) -> f64 {
        // PE = Σ p²/(2 M_∞(x));  KE = Σ ρ(x)|v|²/2 = Σ |v|²/(2/ρ) = Σ |v|²·inv_rho⁻¹/2.
        let pe = Zip::from(&self.p)
            .and(&self.m_inf)
            .fold(0.0_f64, |acc, &p, &mi| acc + p * p / (2.0 * mi));
        let ke = Zip::from(&self.vx)
            .and(&self.vy)
            .and(&self.vz)
            .and(&self.inv_rho)
            .fold(0.0_f64, |acc, &vx, &vy, &vz, &ir| {
                acc + (vx * vx + vy * vy + vz * vz) / (2.0 * ir)
            });
        (pe + ke) * self.cell_volume
    }

    /// Spectral derivative `∂field/∂xₐ → out` via apollo's batched tiled per-axis
    /// 3-D FFT: forward along `axis`, multiply by `i·k`, inverse along `axis`.
    /// Reuses the owned complex scratch `cbuf` (no allocation).
    fn axis_derivative(
        fft: &Fft3d,
        k: &[f64],
        axis: usize,
        field: &Array3<f64>,
        cbuf: &mut Array3<Complex64>,
        out: &mut Array3<f64>,
    ) {
        Zip::from(&mut *cbuf)
            .and(field)
            .for_each(|c, &f| *c = Complex64::new(f, 0.0));
        fft.forward_axis_complex_inplace(cbuf, axis);
        for (m, mut lane) in cbuf.axis_iter_mut(Axis(axis)).enumerate() {
            let factor = Complex64::new(0.0, k[m]);
            lane.mapv_inplace(|v| v * factor);
        }
        fft.inverse_axis_complex_inplace(cbuf, axis);
        Zip::from(&mut *out).and(&*cbuf).for_each(|o, c| *o = c.re);
    }

    /// Advance the state by one time step `Δt`.
    pub fn step(&mut self) {
        // 1. Velocity half-step: v += -(Δt/ρ(x)) ∇p (per component, per voxel).
        let dt = self.dt;
        Self::axis_derivative(
            &self.fft,
            &self.kx,
            0,
            &self.p,
            &mut self.cbuf,
            &mut self.gx,
        );
        Self::axis_derivative(
            &self.fft,
            &self.ky,
            1,
            &self.p,
            &mut self.cbuf,
            &mut self.gy,
        );
        Self::axis_derivative(
            &self.fft,
            &self.kz,
            2,
            &self.p,
            &mut self.cbuf,
            &mut self.gz,
        );
        Zip::from(&mut self.vx)
            .and(&self.gx)
            .and(&self.inv_rho)
            .for_each(|v, &g, &ir| *v += -dt * ir * g);
        Zip::from(&mut self.vy)
            .and(&self.gy)
            .and(&self.inv_rho)
            .for_each(|v, &g, &ir| *v += -dt * ir * g);
        Zip::from(&mut self.vz)
            .and(&self.gz)
            .and(&self.inv_rho)
            .for_each(|v, &g, &ir| *v += -dt * ir * g);

        // 2. Dilatation rate D = ∇·v (accumulated into gx).
        Self::axis_derivative(
            &self.fft,
            &self.kx,
            0,
            &self.vx,
            &mut self.cbuf,
            &mut self.gx,
        );
        Self::axis_derivative(
            &self.fft,
            &self.ky,
            1,
            &self.vy,
            &mut self.cbuf,
            &mut self.gy,
        );
        Self::axis_derivative(
            &self.fft,
            &self.kz,
            2,
            &self.vz,
            &mut self.cbuf,
            &mut self.gz,
        );
        Zip::from(&mut self.gx)
            .and(&self.gy)
            .and(&self.gz)
            .for_each(|d, &y, &z| *d += y + z); // gx = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z

        // 3. Advance each σ_l with the exact exponential integrator (per-voxel
        //    coefficients) and accumulate its trapezoidal pressure contribution
        //    into gy (reused as the relax sum): σ_new = decay·σ + gain·D.
        self.gy.fill(0.0);
        for (arm, sigma) in self.arms.iter().zip(self.sigma.iter_mut()) {
            Zip::from(sigma)
                .and(&self.gx)
                .and(&mut self.gy)
                .and(&arm.decay)
                .and(&arm.gain)
                .and(&arm.inv_tau)
                .for_each(|s, &d, r, &decay, &gain, &inv_tau| {
                    let old = *s;
                    let new = decay.mul_add(old, gain * d);
                    *r += 0.5 * (old + new) * inv_tau;
                    *s = new;
                });
        }

        // 4. Pressure update: p += -Δt (M_U(x) D + Σ_l ½(σ_l+σ_l^new)/τ_l(x)).
        Zip::from(&mut self.p)
            .and(&self.gx)
            .and(&self.gy)
            .and(&self.m_u)
            .for_each(|p, &d, &relax, &mu| *p -= dt * (mu * d + relax));

        // 5. Soft pressure sources: p[index] += signal[step].
        for (index, signal) in &self.pressure_sources {
            if let Some(&s) = signal.get(self.step_count) {
                self.p[[index.0, index.1, index.2]] += s;
            }
        }

        // 6. Absorbing boundary: damp p and v inside the sponge layer.
        if let Some(decay) = &self.damping_decay {
            Zip::from(&mut self.p).and(decay).for_each(|p, &d| *p *= d);
            Zip::from(&mut self.vx).and(decay).for_each(|v, &d| *v *= d);
            Zip::from(&mut self.vy).and(decay).for_each(|v, &d| *v *= d);
            Zip::from(&mut self.vz).and(decay).for_each(|v, &d| *v *= d);
        }

        // 7. Record sensor traces, then advance the simulation clock.
        for (trace, &index) in self
            .sensor_record
            .iter_mut()
            .zip(self.pressure_sensors.iter())
        {
            trace.push(self.p[[index.0, index.1, index.2]]);
        }
        self.step_count += 1;
    }
}

/// Analytic amplitude attenuation `α(ω) = |Im k|`, `k = ω√(ρ/M*(ω))` of a
/// generalized-Maxwell spectrum `M*(ω) = M_∞ + Σ ΔMₗ iωτₗ/(1+iωτₗ)` \[Np·m⁻¹].
/// Used to calibrate the per-voxel relaxation strength against a target absorption.
fn relaxation_attenuation(omega: f64, rho: f64, m_inf: f64, weights: &[f64], taus: &[f64]) -> f64 {
    let mut mstar = Complex64::new(m_inf, 0.0);
    for (&w, &t) in weights.iter().zip(taus) {
        let iwt = Complex64::new(0.0, omega * t);
        mstar += w * iwt / (1.0 + iwt);
    }
    let k = (Complex64::new(rho, 0.0) / mstar).sqrt() * omega;
    k.im.abs()
}

/// FFT-order signed wavenumbers `k[m] = 2π m'/(n·Δx)` with `m' = m` for `m < n/2`
/// and `m' = m − n` otherwise. For `n = 1` this is `[0]` (derivative along a
/// singleton axis is zero).
///
/// The Nyquist bin (`m = n/2`, even `n`) is forced to 0: the first-derivative
/// operator `i·k` is purely imaginary there, so a nonzero Nyquist wavenumber
/// would inject a spurious antisymmetric (non-real) component into `∂p/∂α` of a
/// real field. Zeroing it guarantees a real-valued spectral derivative, matching
/// the KZK τ-derivative convention (`kzk::nonlinearity`).
fn fft_wavenumbers(n: usize, dx: f64) -> Vec<f64> {
    let norm = TWO_PI / (n as f64 * dx);
    (0..n)
        .map(|m| {
            if n.is_multiple_of(2) && m == n / 2 {
                return 0.0;
            }
            let signed = if m < n / 2 {
                m as f64
            } else {
                m as f64 - n as f64
            };
            signed * norm
        })
        .collect()
}
