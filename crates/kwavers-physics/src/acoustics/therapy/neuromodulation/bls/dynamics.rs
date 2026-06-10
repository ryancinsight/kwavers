//! Transient bilayer-sonophore dynamics — the full leaflet Rayleigh–Plesset ODE
//! (Plaksin et al. 2014, Eq. 2; PySONIC `BilayerSonophore.derivatives`).
//!
//! This solves the *exact* inertial/viscous leaflet dynamics, removing the
//! quasi-static approximation of [`super::pressures`]. The state is
//! `[U, Z, n_g]` (apex velocity, deflection, intramembrane gas content) and the
//! equations are reproduced verbatim from the reference implementation:
//!
//! ```text
//! dU/dt = P_tot/(ρ_L·|R|) − 3U²/(2R)              (accP + accNL)
//! dZ/dt = U
//! dn_g/dt = 2·S(Z)·D_gl·(C₀ − P_g/k_H)/ξ          (gas flux)
//! P_tot = P_M + P_g − P₀ − P_ac + P_E + P_V + P_elec
//! P_V   = −12·U·δ₀·μ_S/R² − 4·U·μ_L/|R|
//! ```
//!
//! with `R(Z) = (a²+Z²)/(2Z)`, `S(Z) = π(a²+Z²)`, the intermolecular `P_M`
//! (quadrature), elastic `P_E = −k_A(Z/a)²/R`, electrical `P_elec`, and gas `P_g`
//! pressures of [`super::pressures`] (all exact Plaksin/PySONIC expressions).
//!
//! # Z = 0 handling (exact, no assumption)
//!
//! At the flat membrane `R→∞`, so the acceleration vanishes and `Z=0` is a fixed
//! point. The reference resolves this *not* by regularising the equation but by
//! **seeding** the integration with the quasi-static deflection at the first
//! sub-step, `Z(0) = balanced_deflection(P_ac(dt))` (PySONIC
//! `computeInitialDeflection`/`balancedefQS`). The seed is an exact root of the
//! quasi-steady balance, and the steady-state cycle is independent of it (the
//! transient decays under the viscous terms). `Z` is clamped to
//! `Z_min = −0.49·Δ` (the critical compression), matching the reference.
//!
//! # References
//!
//! - Plaksin, M., Shoham, S. & Kimmel, E. (2014). *Phys. Rev. X* 4, 011004 (Eq. 2).
//! - Lemaire, T. et al. (2019). *J. Neural Eng.* 16, 046007 (PySONIC `bls.py`).

use super::capacitance::bls_capacitance;
use super::pressures::{
    curvature_radius, elastic_pressure, electrical_pressure, gas_pressure, initial_gas_mol,
    molecular_pressure, rest_gap, A_RADIUS_M, C0_MOL_M3, DELTA0_M, DGL_M2_S, KH_PA_M3_MOL,
    MU_L_PA_S, MU_S_PA_S, P0_PA, RHO_L_KG_M3, XI_M,
};
use super::super::intramembrane_cavitation::{CapacitanceSource, PhaseCycle};
use std::f64::consts::PI;

/// Relative lower bound on deflection (critical compression), `Z_min = −0.49·Δ`
/// (PySONIC `rel_Zmin`).
pub const REL_ZMIN: f64 = -0.49;

/// Leaflet surface area `S(Z) = π(a²+Z²)` [m²].
#[inline]
fn surface(z: f64) -> f64 {
    PI * (A_RADIUS_M * A_RADIUS_M + z * z)
}

/// Quasi-steady total pressure used for seeding (PySONIC `PtotQS`):
/// `P_M + P_g − P₀ − P_ac + P_elec` (no elastic/viscous terms).
fn ptot_qs(z: f64, ng: f64, qm: f64, pac: f64, delta: f64) -> f64 {
    molecular_pressure(z, delta) + gas_pressure(ng, z, delta) - P0_PA - pac
        + electrical_pressure(z, qm)
}

/// Quasi-steady seed deflection: root of [`ptot_qs`] over `[Z_min, a]`
/// (PySONIC `balancedefQS`, by bisection). Falls back to 0 if no sign change.
fn balanced_deflection(ng: f64, qm: f64, pac: f64, delta: f64) -> f64 {
    let zmin = REL_ZMIN * delta;
    let zmax = A_RADIUS_M;
    let f = |z: f64| ptot_qs(z, ng, qm, pac, delta);
    if f(zmin) * f(zmax) > 0.0 {
        return 0.0;
    }
    let (mut lo, mut hi) = (zmin, zmax);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if f(lo) * f(mid) <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Leaflet viscous pressure `−12·U·δ₀·μ_S/R² − 4·U·μ_L/|R|` (PySONIC
/// `PVleaflet + PVfluid`).
#[inline]
fn viscous_pressure(u: f64, r: f64) -> f64 {
    -12.0 * u * DELTA0_M * MU_S_PA_S / (r * r) - 4.0 * u * MU_L_PA_S / r.abs()
}

/// Gas molar flux `2·S(Z)·D_gl·(C₀ − P_g/k_H)/ξ` [mol/s] (PySONIC `gasFlux`).
#[inline]
fn gas_flux(z: f64, pg: f64) -> f64 {
    2.0 * surface(z) * DGL_M2_S * (C0_MOL_M3 - pg / KH_PA_M3_MOL) / XI_M
}

/// Mechanical state derivative `[dU, dZ, dn_g]` (SI units: m/s², m/s, mol/s).
///
/// `pm` is the area-averaged intermolecular pressure `P_M(Z)` supplied by the
/// caller (an O(1) table lookup during integration; see [`PmTable`]), keeping the
/// hot path free of the per-step quadrature.
#[inline]
fn derivatives(
    u: f64,
    z: f64,
    ng: f64,
    qm: f64,
    pac: f64,
    delta: f64,
    pm: f64,
) -> (f64, f64, f64) {
    let r = curvature_radius(z);
    let ptot = pm + gas_pressure(ng, z, delta) - P0_PA - pac
        + elastic_pressure(z)
        + viscous_pressure(u, r)
        + electrical_pressure(z, qm);
    // accP + accNL: P_tot/(ρ_L·|R|) − 3U²/(2R).
    let du = if r.is_finite() {
        ptot / (RHO_L_KG_M3 * r.abs()) - 1.5 * u * u / r
    } else {
        0.0
    };
    let pg = gas_pressure(ng, z, delta);
    (du, u, gas_flux(z, pg))
}

/// Precomputed lookup table of the exact area-averaged intermolecular pressure
/// `P_M(Z)` over the *smooth* deflection range, with linear interpolation and an
/// exact-quadrature fallback in the steep steric-wall band and beyond the table.
///
/// `P_M(Z)` (the [`molecular_pressure`] quadrature) depends only on `Z` for a
/// fixed rest gap, so tabulating it turns the per-derivative cost from an
/// `O(N_quad)` integral into an `O(1)` lookup. Linear interpolation is accurate
/// where `P_M` is gentle; the near-wall band (`Z < z_lo`, where `P_M` rises
/// almost vertically) and any excursion past `z_hi` fall back to the exact
/// quadrature — those regions are visited rarely and at tiny step size, so the
/// fallback cost is negligible while preserving stability/accuracy.
struct PmTable {
    z_lo: f64,
    z_hi: f64,
    inv_dz: f64,
    delta: f64,
    values: Vec<f64>,
}

impl PmTable {
    /// Lower edge of the tabulated (smooth) range, set where the inter-leaflet
    /// gap is half the rest gap (`2·z + Δ = Δ/2`); below this the wall is steep.
    fn smooth_lo(delta: f64) -> f64 {
        -0.25 * delta
    }

    /// Build over `[smooth_lo(δ), z_hi]` with `n` samples of the exact quadrature.
    fn new(z_hi: f64, n: usize, delta: f64) -> Self {
        let z_lo = Self::smooth_lo(delta);
        let dz = (z_hi - z_lo) / (n - 1) as f64;
        let values = (0..n)
            .map(|i| molecular_pressure(z_lo + i as f64 * dz, delta))
            .collect();
        Self {
            z_lo,
            z_hi,
            inv_dz: 1.0 / dz,
            delta,
            values,
        }
    }

    /// `P_M(Z)`: interpolated in the smooth range, exact otherwise.
    #[inline]
    fn at(&self, z: f64) -> f64 {
        if !(z > self.z_lo && z < self.z_hi) {
            return molecular_pressure(z, self.delta); // wall band / out of range
        }
        let x = (z - self.z_lo) * self.inv_dz;
        let i = x as usize;
        let frac = x - i as f64;
        self.values[i] * (1.0 - frac) + self.values[i + 1] * frac
    }
}

/// Transient bilayer-sonophore capacitance source: the leaflet deflection `Z(t)`
/// is the steady-state solution of the full Rayleigh–Plesset ODE under the
/// acoustic carrier, mapped to capacitance through the exact curved-dome geometry
/// ([`super::capacitance::bls_capacitance`]). This is the *exact* (inertial) deflection —
/// the acoustic pressure is the input.
#[derive(Debug, Clone)]
pub struct BilayerSonophoreDynamic {
    /// Phase-periodic capacitance cycle (shared SSOT source; holds C_m and
    /// dC_m/dt over one carrier cycle plus the interpolation).
    cycle: PhaseCycle,
    /// Peak leaflet deflection over the steady cycle [m] (the only `Z` datum
    /// retained; the full `Z(t)` array is not stored).
    peak_deflection_m: f64,
}

/// One classic RK4 step of the mechanical state over `dt` (SI), using the
/// precomputed `P_M(Z)` table for the molecular pressure.
#[inline]
fn rk4_step(
    s: (f64, f64, f64),
    t: f64,
    dt: f64,
    qm0: f64,
    delta: f64,
    pac: &impl Fn(f64) -> f64,
    pm: &PmTable,
) -> (f64, f64, f64) {
    let (u, z, ng) = s;
    let add = |w: f64, k: &(f64, f64, f64)| (u + w * k.0, z + w * k.1, ng + w * k.2);
    let k1 = derivatives(u, z, ng, qm0, pac(t), delta, pm.at(z));
    let s2 = add(0.5 * dt, &k1);
    let k2 = derivatives(s2.0, s2.1, s2.2, qm0, pac(t + 0.5 * dt), delta, pm.at(s2.1));
    let s3 = add(0.5 * dt, &k2);
    let k3 = derivatives(s3.0, s3.1, s3.2, qm0, pac(t + 0.5 * dt), delta, pm.at(s3.1));
    let s4 = add(dt, &k3);
    let k4 = derivatives(s4.0, s4.1, s4.2, qm0, pac(t + dt), delta, pm.at(s4.1));
    (
        u + dt / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0),
        z + dt / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1),
        ng + dt / 6.0 * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2),
    )
}

impl BilayerSonophoreDynamic {
    /// Phase samples used to store the steady cycle (fine enough that the
    /// central-difference `dC_m/dt` resolves the sharp capacitance excursion).
    const N_SAMPLES: usize = 1000;
    /// Carrier cycles integrated to reach the steady oscillation (the slow
    /// gas-rectification drift settles within this span at the validated
    /// amplitudes).
    const N_SETTLE: usize = 30;
    /// Deflection error tolerance for adaptive step control [m].
    const Z_TOL_M: f64 = 2.0e-13;
    /// Samples in the precomputed exact `P_M(Z)` lookup table.
    const N_PM_TABLE: usize = 8192;
    /// Hard cap on integration steps (settle + record). Far above the nominal
    /// count (~10⁴); a no-hang backstop for pathological stiff regimes.
    const MAX_STEPS: usize = 200_000;

    /// Construct by integrating the leaflet ODE to its steady cycle under an
    /// acoustic carrier of peak pressure `pressure_amp_pa` [Pa] and frequency
    /// `freq_mhz` [MHz], with resting charge set by `v_rest_mv` and rest
    /// capacitance `cm0_uf_cm2` [µF/cm²].
    ///
    /// Uses adaptive step-doubling RK4 (the leaflet ODE is stiff near the steric
    /// wall, so a fixed step is not stable at therapeutic amplitudes).
    #[must_use]
    pub fn new(cm0_uf_cm2: f64, freq_mhz: f64, pressure_amp_pa: f64, v_rest_mv: f64) -> Self {
        let f_hz = freq_mhz * 1.0e6;
        let omega_si = 2.0 * PI * f_hz;
        let omega_rad_ms = 2.0 * PI * 1.0e3 * freq_mhz;
        let qm0 = (cm0_uf_cm2 * 1.0e-2) * (v_rest_mv * 1.0e-3); // C/m²
        let delta = rest_gap(qm0);
        let ng0 = initial_gas_mol(delta);
        let zmin = REL_ZMIN * delta;

        let period_s = 1.0 / f_hz;
        let pac = |t: f64| pressure_amp_pa * (omega_si * t).sin();
        let dt_max = period_s / 200.0;
        // Lower bound on the adaptive step — fine enough to resolve the stiff
        // steric wall. The per-step cost is dominated by `P_M`, which is an O(1)
        // table lookup here (see `pm_table`), so a fine `dt_min` is affordable.
        let dt_min = period_s / 5.0e6;

        // Exact `P_M(Z)` precomputed over the deflection range (O(1) lookup in the
        // hot path); rebuilt once per construction. Range spans the steric wall to
        // beyond the largest plausible expansion.
        let pm = PmTable::new(A_RADIUS_M, Self::N_PM_TABLE, delta);

        // Adaptive step-doubling RK4 with an inelastic steric wall. The
        // intermolecular pressure rises near-vertically toward `Z_min`, so an
        // explicit step that reaches the wall overshoots; clamping `Z` to `Z_min`
        // **and zeroing the inward velocity** there (the leaflets cannot
        // interpenetrate) removes the runaway that would otherwise force the
        // stepper to collapse to `dt_min` every cycle. The step is accepted once
        // it meets tolerance or hits `dt_min`; `*dt` then holds the accepted step.
        let advance = |s: (f64, f64, f64), t: f64, dt: &mut f64| -> (f64, f64, f64) {
            loop {
                let full = rk4_step(s, t, *dt, qm0, delta, &pac, &pm);
                let half = rk4_step(s, t, 0.5 * *dt, qm0, delta, &pac, &pm);
                let two = rk4_step(half, t + 0.5 * *dt, 0.5 * *dt, qm0, delta, &pac, &pm);
                let err = (two.1 - full.1).abs();
                // Accept once tolerance is met OR the step floor is reached. The
                // `dt_min` branch must accept unconditionally (even on a non-finite
                // step) — otherwise a wall blow-up at the floor cannot shrink
                // further and would loop forever.
                if err <= Self::Z_TOL_M || *dt <= dt_min {
                    let (mut u, mut z, ng) = two;
                    // Hard steric-wall contact (or a blown step at the floor):
                    // stop at the wall, drop the inward velocity, keep prior gas.
                    if !(z.is_finite() && u.is_finite() && ng.is_finite()) {
                        return (0.0, zmin, if ng.is_finite() { ng } else { s.2 });
                    }
                    if z <= zmin {
                        z = zmin;
                        u = u.max(0.0); // inelastic contact: no further inward motion
                    }
                    return (u, z, ng);
                }
                *dt = (0.5 * *dt).max(dt_min);
            }
        };

        // Seed: quasi-steady deflection at the first sub-step (PySONIC).
        let mut s = (
            0.0_f64,
            balanced_deflection(ng0, qm0, pac(dt_max), delta).max(zmin),
            ng0,
        );
        let mut t = 0.0_f64;
        // Each step starts from `dt_max`; `advance` shrinks toward `dt_min` only as
        // the local error requires, so the step size need not be carried.
        let dt = dt_max;

        // Settle to the steady cycle (the slow gas-rectification drift, not just
        // the mechanical peak, must equilibrate). Steps are clamped to land
        // exactly on `settle_end = N·period`, so `t` is at carrier phase 0 for
        // recording. A hard step budget guarantees construction terminates even in
        // a pathological stiff regime (the leaflet should not reach the steric
        // wall at validated amplitudes; if it did, the explicit stepper would
        // collapse to `dt_min` — the budget bounds that instead of hanging).
        let mut steps = 0usize;
        let settle_end = Self::N_SETTLE as f64 * period_s;
        while t < settle_end && steps < Self::MAX_STEPS {
            let mut hh = dt.min(settle_end - t);
            s = advance(s, t, &mut hh);
            t += hh;
            steps += 1;
        }

        // Record one steady period onto a uniform phase grid (O(steps + n): a
        // single forward grid pointer fills samples as `t` passes them).
        let n = Self::N_SAMPLES;
        let t0 = t;
        let record_end = t + period_s;
        let mut z_cycle = vec![s.1; n];
        let mut prev = (t, s.1);
        let mut gi = 0usize;
        while t < record_end && steps < Self::MAX_STEPS {
            let mut hh = dt.min(record_end - t);
            let ns = advance(s, t, &mut hh);
            let cur = (t + hh, ns.1);
            while gi < n {
                let tg = t0 + (gi as f64 / n as f64) * period_s;
                if tg > cur.0 {
                    break;
                }
                let frac = if cur.0 > prev.0 {
                    (tg - prev.0) / (cur.0 - prev.0)
                } else {
                    0.0
                };
                z_cycle[gi] = prev.1 * (1.0 - frac) + cur.1 * frac;
                gi += 1;
            }
            s = ns;
            t += hh;
            prev = cur;
            steps += 1;
        }
        let peak_deflection_m = z_cycle.iter().cloned().fold(f64::MIN, f64::max);
        let cm_cycle: Vec<f64> = z_cycle
            .iter()
            .map(|&zz| bls_capacitance(zz, cm0_uf_cm2, A_RADIUS_M, delta))
            .collect();

        Self {
            cycle: PhaseCycle::new(cm0_uf_cm2, omega_rad_ms, cm_cycle),
            peak_deflection_m,
        }
    }

    /// Peak leaflet deflection over the steady cycle [m] (validation accessor).
    #[inline]
    #[must_use]
    pub fn peak_deflection_m(&self) -> f64 {
        self.peak_deflection_m
    }
}

impl CapacitanceSource for BilayerSonophoreDynamic {
    #[inline]
    fn capacitance(&self, t_ms: f64) -> f64 {
        self.cycle.capacitance(t_ms)
    }
    #[inline]
    fn capacitance_rate(&self, t_ms: f64) -> f64 {
        self.cycle.capacitance_rate(t_ms)
    }
    #[inline]
    fn baseline_capacitance(&self) -> f64 {
        self.cycle.baseline_capacitance()
    }
    #[inline]
    fn carrier_omega_rad_ms(&self) -> f64 {
        self.cycle.carrier_omega_rad_ms()
    }
    #[inline]
    fn is_source_valid(&self) -> bool {
        self.cycle.is_source_valid()
    }
}
