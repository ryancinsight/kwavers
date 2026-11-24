<a id="kw-phys-006"></a>

## KW-PHYS-006 — Single-bubble-extinction cloud attenuation (Xia 2020 route) and coherent shielding factor [patch] — done (Nov 21, 2026)

priority: feature; needs: none; scope: new module `cloud_attenuation.rs` in `kwavers-physics` (`acoustics/bubble_dynamics/`); milestone: none

- **Outcome:** the library can derive a cloud attenuation coefficient from a single bubble's per-cycle dissipation — the route the formation work needed — alongside the existing linear Commander & Prosperetti dispersion, and can turn an attenuation into the coherent shielding factor for influx-type (power-law-in-pressure) quantities.
- **Driver:** Build 9 of the triplet-pulse formation work (Nov 21, 2026) derived α from first principles via the Xia et al. (2020) energy/extinction argument — per-cycle work `W = |∮ p dV|`, extinction cross-section `σ_e = W·f/I`, `α = N·σ_e/2` — obtaining α = 3150 Np/m (Kuji) / 2037 Np/m (Iwasaki) at N = 1.678×10¹¹ m⁻³, values no linear dispersion gives at these void fractions and amplitudes. The linear SSOT (`bubbly_medium::commander_prosperetti_attenuation`) already existed and is unchanged; the analytical shielding ODE that consumes it also already existed. What was missing is this nonlinear route and the shielding factor.
- **Acceptance:** the work integral is exact on an analytic p–V ellipse (`W = π·P̂·V̂·sin φ`); the α round-trip at the Build 9 Kuji point returns 3150 Np/m; the shielding factor has the correct thin/thick limits (→ 1 and → 1/(mαL)).

### Resolution (Nov 21, 2026)

- **Implementation.** `cycle_work(pressure, radius)` (trapezoidal `|∮ p dV|`, loop closed last→first); `extinction_cross_section(work, frequency, pressure_amplitude, density, sound_speed)` with `I = P_A²/(2ρc)`; `cloud_attenuation_coefficient(number_density, cross_section)` = `N·σ_e/2`; `cloud_length(cycles, formation_cycles, frequency, sound_speed)` = `max(n − n_crit, 0)·λ/3` (cloud growth behind the shock front, Bader & Holland §3.4); `shielding_factor_coherent(alpha, length, influx_exponent)` = `(1 − e^{−mαL})/(mαL)` with a small-argument guard. Module docs cross-reference `bubbly_medium` as the linear-dispersion SSOT and state the division of labour.
- **Negative result carried in the docs.** Build 9 used exactly this machinery to test whether cloud shielding can produce the influx cutoff the two protocols jointly require (turnover between 25 and 56 cycles): it cannot — attenuation gives S(n) ∝ 1/n while the pair needs a near-step (γ ≈ 1.8–3), and the differential goes the wrong way (shielding hits Kuji's larger, denser cloud harder, where Iwasaki needs the stronger suppression). The module documents this so the law is reused without re-litigating the conclusion: the law is general physics; the conclusion is regime-specific.
