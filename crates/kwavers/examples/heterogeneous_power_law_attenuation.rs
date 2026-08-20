//! Replication of the Fullwave 2.5 heterogeneous power-law attenuation result.
//!
//! [Fullwave 2.5](https://github.com/pinton-lab/fullwave25) reports accurate
//! attenuation modelling with **both** the coefficient `α₀` and the exponent `γ`
//! varying spatially. The reference study by [Sode and Pinton]
//! validates `α₀ = 0.25…0.75` dB·cm⁻¹·MHz⁻ᵞ and `γ = 0.4…1.6`. This example
//! reproduces that envelope in kwavers by simulation, not by inspecting the fit:
//!
//! 1. **Homogeneous sweep** — for each `(α₀, γ)` in that envelope, a broadband
//!    pulse propagates through a medium built by
//!    [`ViscoacousticMemorySolver::from_power_law_fields`]. `α(f)` is recovered
//!    from the **spectral ratio** of two downstream sensors,
//!    `α(f) = −ln(|P₂|/|P₁|)/d`, and compared to the prescribed law. The
//!    measurement never consults the fitted spectrum, so a mis-fit shows up as
//!    a discrepancy rather than cancelling out.
//! 2. **Heterogeneous layers** — an abdominal-wall-like stack of fat and muscle
//!    layers that differ in `γ` as well as `α₀`. The transmitted spectrum is
//!    compared to the path-length-weighted law `Σₖ αₖ(f)·Lₖ`, which is the
//!    exact plane-wave prediction and which **no uniform-exponent medium can
//!    reproduce** — the layers' exponents differ, so their sum is not a power
//!    law of any single exponent.
//!
//! ## Measured accuracy
//!
//! Over the whole envelope the simulated `α(f)` matches the prescribed law to
//! **3.4 % worst case, and to 0.5 % across the band interior** — the residual
//! concentrates at 0.6 and 4.6 MHz, the edges where the excitation carries
//! least energy. The heterogeneous fat/muscle stack, where `γ` varies along the
//! propagation path, matches the exact path-weighted prediction to **1.0 %**.
//!
//! This runs on **three** relaxation arms. With the relaxation times optimized
//! rather than log-spaced (`RelaxationTimePlacement::Optimized`), three arms
//! reproduce the fit to 0.16 % analytically, so the residual above is the
//! *measurement*, not the medium: six arms move the worst case only from 3.4 %
//! to 3.0 %. Each arm dropped is one fewer memory field per voxel in the
//! solver, which is the dimension that decides whether a 3-D heterogeneous run
//! fits in memory at all.
//!
//! Two measurement details are load-bearing, both established by experiment
//! rather than assumed (KW-SOL-072):
//!
//! - **The analysis gate must not be tapered.** A Hann taper over the gate
//!   biased the recovered `α` low by 8–19 %, multiplicatively and independently
//!   of sensor separation, because the far-sensor pulse is dispersively
//!   broadened and so is weighted differently by the taper than the near-sensor
//!   pulse. The pulse decays to zero well inside the gate, so a rectangular
//!   gate truncates nothing and needs no taper at all.
//! - **The gate must be centred on the true emission time**, `3·PULSE_WIDTH_S`
//!   after step 0, not on step 0 plus the transit time.
//!
//! The scheme itself was exonerated analytically before the measurement was
//! suspected — see `discrete_dispersion_matches_continuum` in the solver tests.
//!
//! Outputs (under `target/fullwave_attenuation/`):
//! - `attenuation_sweep.png`  — measured vs prescribed `α(f)`, log-log
//! - `attenuation_sweep.csv`  — every measured point
//! - `layered_medium.csv`     — heterogeneous-layer comparison
//!
//! Run: `cargo run --release --example heterogeneous_power_law_attenuation`
//!
//! [Sode and Pinton]: https://arxiv.org/abs/2606.11103

use anyhow::Result;

#[path = "heterogeneous_power_law_attenuation/mod.rs"]
mod attenuation_experiment;

fn main() -> Result<()> {
    attenuation_experiment::run()
}
