# Transient transport at an isolated spherical interface

`TransientGasDiffusion` advances one dissolved species from prescribed wall
radius and liquid-side concentration samples. It returns signed exchanges
with the bubble and outer reservoir. `DiffusiveBubble` supplies those
samples from opt-in Rayleigh–Plesset mechanics and a fixed gas-pressure
closure. Both solvers retain the concentration field during acoustic
bursts and quiet intervals.

## Governing equation and conservation

Let `a = ((R/R_ref)^3 + 3x)^(4/3)` in the material coordinate
`x = (r^3 - R^3)/(3 R_ref^3)`, and use `c = C - C_bulk`.
The equation is `c_t = (D/R_ref^2) (a c_x)_x`. The inner boundary
condition uses the supplied wall concentration. The outer material surface
is held at bulk concentration and the liquid starts uniformly at bulk.

Cell averages carry inventory `4π R_ref^3 Δx c`. Fluxes shared by
adjacent cells cancel in the sum of the update equations. The liquid
inventory change plus the two reported outward exchanges therefore sums to
zero in exact arithmetic. The inner flux uses the wall concentration and
the first two cell averages to reproduce the wall gradient of any quadratic
in `x`. Internal face gradients use neighboring averages on a
smoothly graded grid. The first backward Euler step damps high modes after
an initial wall discontinuity. Implicit midpoint thereafter approximates
the time integral of flux at second order for smooth histories.

## Interfaces and scope

The physics crate owns the spherical material coordinate and diffusion
metric. The solver crate owns the mesh, the implicit update, and the
retained liquid state. The same code uses the caller's floating-point
scalar throughout. For `N` radial cells, each step takes O(N) time and
the solver stores O(N) values. The step commits the new wall and liquid
state after the solve and flux checks succeed.

The equilibrium-radius conversion in the fixtures linearizes
`n = (4π/3) (P_0 R_eq^3 + 2σ R_eq^2)/(R_g T)` at the initial radius.
This yields a signed rate for checking flux. It does not evolve the
prescribed radius from the gas inventory. The coupled solver below uses
a different gas closure and its corresponding equilibrium-radius derivative.
The finite outer distance and time and space resolution belong to the
problem setup.

A leading thin-layer history kernel preserves part of the transport at
lower cost. This increment instead retains the full spherical metric so
the solution also passes the fixed-sphere curvature and transient limits.
An independent wall-following, nodal difference scheme with BDF2 tests the
material-volume implementation using different spatial and time operators.

## Coupling to Rayleigh–Plesset

`DiffusiveBubble` evolves a clean spherical interface in an incompressible
Newtonian liquid. Set the initial gas pressure and amount to
`p_g0 = P0 - p_v + 2σ/R0` and `n0 = 4π R0³ p_g0/(3 R_g T0)`.
At each state compute `p_g = p_g0 [(n/n0)(R0/R)^3]^κ`. The
Rayleigh–Plesset force uses `p_g + p_v` and the liquid boundary uses
`C_s = H p_g`, with `H = C_sat/(P0 - p_v)`. Here `C_sat` is the
flat-interface saturation concentration at ambient pressure. Use the
instantaneous absolute far-field pressure at the interval midpoint and the
ambient value during acoustic off time. The model holds κ and vapor
partial pressure fixed. The bulk concentration can drive dissolution or
uptake independently of the acoustic drive.

The mechanics uses implicit midpoint with the midpoint gas inventory.
At a trial endpoint inventory, iterate the wall equations to convergence
and then solve the finite-volume diffusion step using the trial endpoint
radius and Henry concentration. Repeat with the inventory implied by
the signed flux until the residuals fall within 16 scalar machine epsilons,
scaled by the initial radius, `sqrt(P0/ρ)`, and the initial gas amount.
Updating mechanics before each diffusion solve keeps the radial work to one
linear solve per gas iterate. The fixed-point method is local in step size.
An inadmissible trial or unconverged solve returns an error before either
history is committed. A caller can retry with a shorter interval and a
pressure sample at its new midpoint. Resolve the bubble's natural period
and the drive. Keep pulse edges on step boundaries and refine the near-wall
mesh and time interval for each exposure. Midpoint is not L-stable.
Abrupt interfacial changes can excite stiff diffusion modes even during
quiet intervals. The first diffusion step uses backward Euler. The
driven sub-ppm verification below uses `f64`. Static equilibrium is also
checked in `f32`, whose 16-epsilon coupling tolerance can exceed the
transferred amount for these driven cases.

In the accepted update, `Δn = Δn_into_bubble`. Summing the liquid
finite-volume equations cancels all interior face exchanges and yields
`Δn + Δn_liquid + Δn_reservoir = 0`. The solver tracks the change in bubble
gas separately from `n0` to retain sub-ULP increments. The reservoir term
is positive for gas leaving the liquid. This budget covers the
noncondensable species. It models constant vapor pressure, with no vapor
transport equation. Spherical symmetry and incompressibility restrict the
coupled model to an isolated radial bubble. Interfacial contamination,
streaming, and energy transport require additional closures.

### Mechanical equilibrium radius

`DiffusiveBubble::equilibrium_radius` solves the ambient force balance for
the accepted noncondensable inventory. For κ = 1 it solves
`(P0 - p_v) R_eq^3 + 2σ R_eq^2 = 3 n R_g T0/(4π)`. For κ ≥ 1 it uses
the pressure closure of the mechanical solver. This diagnostic leaves
the instantaneous radius, velocity, and liquid state unchanged.

### Verification

The independent C reference in `tests/fixtures/diffusive_bubble/reference.c`
uses physical distance from the wall, `s = r - R(t)`. Its nodal,
geometrically graded mesh resolves the convection and diffusion terms in
`C_t = D C_ss + [2D/(R+s) + Ṙ(1-(R/(R+s))²)] C_s`. The reference uses
fourth-order backward differences after startup, high-order gas extrapolation
into the endpoint pressure, and one-sided wall differentiation followed by
a separate gas balance. The two spatial coordinates, time schemes, and gas
flux discretizations are independent. Compare every cycle in both bursts and
off intervals. The CSV radius and velocity are instantaneous cycle-end
values. The rate is the cycle gas change divided by cycle length and by the
initial static derivative `dn/dR_eq = (n0/R0) [3 - 2σ/(κ R0 p_g0)]`.
Differentiate `p_g(R_eq,n)=P0-p_v+2σ/R_eq` to obtain this derivative.

```sh
fixture=crates/kwavers-solver/tests/fixtures/diffusive_bubble
cc -O3 -std=c11 -Wall -Wextra -Werror "$fixture/reference.c" -lm -o /tmp/kwavers-coupled-reference
/tmp/kwavers-coupled-reference 30 6144 3072 60 20000 1.05
/tmp/kwavers-coupled-reference 42.5 6144 3072 60 20000 1.12
cargo nextest run --locked -p kwavers-solver --test diffusive_bubble
```

Both fixtures start at rest with bulk equal to the initial Henry wall
concentration. The 26.6 kHz drive has a 20 kPa peak acoustic offset and
two 16-cycle bursts with two-cycle cosine ramps at either end. The first
off interval is 16 cycles and the second is 24. The time history spans
2.707 ms. The test uses fresh uniform liquid at each scenario's start and
advances the same liquid and gas fields through the full exposure. The
60 μm far boundary is more than 25 diffusion lengths from the initial
wall over this duration. In the C reference, independent refinements of
time and space change any cycle rate by at most 0.00000170 and
0.000112 μm/min, respectively. Doubling the reference extent and the
number of radial intervals changes the rate by at most 0.000109 μm/min.
The Kwavers comparison uses 1536 radial cells and 8192 steps per period. The gas balance is
checked at every step, including between the bursts.

### Verified operating range

The isothermal refinement matrix uses one isolated 30 µm sphere in a
Newtonian liquid at 293.15 K with density 998 kg/m³, viscosity 1 mPa·s,
surface tension 0.072 N/m, diffusivity 2×10⁻⁹ m²/s, and ambient pressure
101.325 kPa. Vapor pressure is zero in this synthetic model. Solubility
at the flat-interface ambient pressure is 0.26 mol/m³. Every exposure
uses a 26.6 kHz carrier at 3.03975 kPa peak offset, one-cycle cosine
ramps, and 96 carrier cycles of elapsed time (3.61 ms). The nine cases
vary on and off lengths independently over 4, 8, and 16 cycles, with
tested duty cycles of 1/3, 1/2, and 2/3. Bulk concentration is 0.98,
1.00, or 1.02 times the initial curved-wall equilibrium concentration.
The fixture enumerates the tested combinations. The sweep samples
wall radius ratios from 0.98968 to 1.01093 and peak speed below
0.05411 m/s, equivalent to wall Mach below 3.66×10⁻⁵ if the sound
speed is 1480 m/s. Only the enumerated combinations have reference
comparisons.

The baseline uses 768 cells, 1024 steps per carrier cycle (36.7 ns),
and a 120 µm wall-to-reservoir distance. At 512 steps per carrier cycle,
two cases pass the gas-transfer budget and exceed the independent
time-refinement budget for wall radius. All cases pass both at 1024.
Each refinement changes one
control. The radial grid uses physical face gaps
`L(j/N)^2`. Extending the reservoir to 480 µm with 1536 cells
preserves the entire original near-wall grid as a prefix and adds
outer material shells. The separate radial-grid check doubles the
cell count at fixed extent. The acceptance budget for each phase and
cumulative gas transfer is `max(0.003 ppm of n0, 2% of |reference|)`.
The budgets for wall radius and speed are 0.1 nm and 50 µm/s. Each
refinement must move every checked quantity by at most one-third of
its budget. The equilibrium-radius budget follows from
`dR_eq/dn = R_eq/[n(3−2σ/(R_eq p_g))]` in the isothermal case.

| Comparison | Maximum phase gas budget used | Maximum wall radius budget used | Maximum wall speed budget used |
| --- | ---: | ---: | ---: |
| Baseline vs independent BDF | 2.80% | 14.62% | 13.07% |
| 2048 steps per cycle | 2.05% | 10.90% | 9.60% |
| 1536 cells at 120 µm | 0.26% | 0.010% | 0.004% |
| 1536 cells at 480 µm with matched inner faces | 0% | 0% | 0% |

All checked phase-end gas ledgers close within 3.10×10⁻¹³ of gross
exchange, below the 10⁻⁸ gate. The retained second-burst uptake in the
8-on/8-off case agrees with the uninterrupted reference and differs
from the reference that resets only the liquid profile before that
burst. The acceptance test requires at least a 0.003 ppm separation.
The undersaturated and supersaturated cases have opposite net exchange
over the same total duration.

The 20 kPa polytropic fixtures above verify the stated single trajectories
at 30 and 42.5 µm, with κ = 1.05 and 1.12, respectively. The maximum
cycle rate discrepancy is 0.00641 µm/min. Across those trajectories the
sampled instantaneous radii are 28.05–32.56 µm and 39.77–46.55 µm,
respectively. Peak wall speed is 0.9122 m/s, equivalent to Mach below
6.17×10⁻⁴ at a sound speed of 1480 m/s. Their pulse schedule and fluid
inputs differ from the isothermal matrix. The isolated
spherical model has no validated operating envelope for collapse,
shape instability, coated bubbles, boundary confinement, or tissue.
Comparison with experiments requires measured pressure histories,
initial bubble populations, gas species and concentrations, and
thermodynamic and interfacial properties for the exposure.

The independent isothermal reference uses wall-following nodal differences
and adaptive BDF. Reproduce the table inputs and check reference sensitivity
with the commands below. Tightening the time tolerance fourfold, doubling
the comparison grid from 640 to 1280 radial intervals, and extending the reference
domain each change the signed phase and total comparisons by less than
4% of their respective budgets.

```sh
fixture=crates/kwavers-solver/tests/fixtures/diffusive_bubble
uv run --no-project --with scipy==1.18.1 --with numpy==2.5.5 python "$fixture/sensitivity_reference.py"
uv run --no-project --with scipy==1.18.1 --with numpy==2.5.5 python "$fixture/sensitivity_reference.py" --intervals 640
uv run --no-project --with scipy==1.18.1 --with numpy==2.5.5 python "$fixture/sensitivity_reference.py" --rtol 3.125e-10
uv run --no-project --with scipy==1.18.1 --with numpy==2.5.5 python "$fixture/sensitivity_reference.py" --extent 8 --intervals 2560
cargo nextest run --locked -p kwavers-solver --test coupled_acceptance
```

## Opt-in compressible and thermal coupling

Use `DiffusiveBubble::new_with_models` to choose `RadialMechanics` and an
optional `GasHeatTransfer` grid. The existing constructor selects
Rayleigh–Plesset and the given polytrope. Both wall models share the
conservative liquid transport, Henry pressure, gas ledger and equilibrium
diagnostic. Keller–Miksis requires `advance_with_forcing` with absolute
far-field pressure and its time derivative at the same midpoint. For
quiet intervals use `BubbleForcing::steady` at ambient pressure.

Keller–Miksis retains first-order liquid compressibility for radial motion.
For signed Mach number M=U/c and the liquid-wall to far-field jump Δp,

```text
(1−M) R A + (3/2)(1−M/3) U² = [(1+M) Δp + (R/c) dΔp/dt]/ρ.
```

The surface tension and viscous terms come from the same wall traction as
Rayleigh–Plesset. Differentiating viscosity moves `4μ/(ρc)` into the
inertial coefficient. The derivative of the fixed polytrope includes
the signed gas change. The temperature-field branch uses the difference
between accepted and trial gas pressures divided by the step. The slope
is a second-order approximation at the time midpoint. The incompressible
limit recovers Rayleigh–Plesset. The existing uncoupled Keller–Miksis
consumer retains its approximate viscous derivative and shares the
pressure-driven radial kernel.

### Heat equation and transactional coupling

Use `y=r/R` and `q=T/T₀−1` with an even temperature grid. At the center
temperature is even. The clean interface has fixed temperature T₀. For
uniform gas pressure, the ideal-gas law integrated over the sphere gives
`(p/p_g0) 3∫₀¹ y²/(1+q)dy = (n/n₀)(R₀/R)³`. Combining continuous
energy and mass balances yields

```text
ρ_g c_p D_t T = dp/dt + k_g ∇²T
u = −r (dp/dt)/(3γp) + (γ−1) k_g (∂T/∂r)/(γp).
```

At frozen midpoint coefficients, the thermal solve is tridiagonal for
each of two right-hand sides and affine in the gas pressure increment.
Use Simpson integration and a scalar Newton solve to enforce the
endpoint inventory equation. Iterate the midpoint coefficients and wall
motion, then use the shared liquid solver to obtain the gas amount.
Each rejected step retains the last accepted wall, liquid and heat fields.
Work per trial is linear in the two radial grid sizes, and the solver
reuses both accepted and scratch buffers across steps.
The ambient equilibrium radius under heat transfer is the relaxed
isothermal radius for the accepted inventory.

A fixed real polytrope cannot represent the thermal phase lag. A lumped
heat-transfer coefficient would add a different thermal closure from the
standalone benchmark. This option retains the gas temperature gradient
to reproduce that benchmark's ideal-gas energy balance. The forcing
contract requires the caller to supply the rate because a midpoint
pressure sample alone cannot determine the derivative. The heat solve
contracts locally with time refinement. The numerical wall Mach check
escapes the acoustic singularity and does not validate large excursions.

### Reference acceptance

`tests/compressible_thermal_bubble.rs` uses the standalone C reference's
physical-distance solute grid and BDF4 histories. The fixture contains
all four radial and thermal model combinations at each of 220 and 230 kHz.
The isolated 30 µm bubble receives 3.03975 kPa peak pressure in two
10 ms bursts. Each quiet interval lasts 5.625 ms. The heat parameters are
γ=1.4 and a reference gas diffusivity of 2.2×10⁻⁵ m²/s. For Keller–Miksis
the sound speed is 1480 m/s. All other synthetic inputs match the
isothermal case above.

The Rust baseline uses 64 steps per carrier period, 384 liquid cells, and
48 heat intervals when gas heat transfer is enabled. The fixture compares
each signed phase and cumulative exchange using
`max(0.003 ppm of n₀, 2% of |reference|)`, phase-end radius within 0.1 nm,
and phase-end velocity within 50 µm/s. It verifies gas conservation
against gross exchange and tests rollback after a coupled step failure.
The combined model halves the step and doubles each grid independently,
using one-third of the corresponding reference budget per refinement.
See the [fixture provenance](../../../../tests/fixtures/diffusive_bubble/compressible_thermal.md).

The compressible option modifies the wall equation. The liquid
concentration still evolves on incompressible material shells. The heat
option assumes uniform pressure, constant gas conductivity and fixed
wall temperature, with a single noncondensable ideal gas and constant
vapor pressure. The comparison verifies these declared synthetic cases
and makes no experimental, tissue, resonance or collapse claim.

## References

- Smith & Wang (2022), JFM 951 A12, equations (2.1)–(2.8).
  https://doi.org/10.1017/jfm.2022.849

- Smith & Wang (2022), JFM 939 A28, equations (2.4)–(2.7).
  https://doi.org/10.1017/jfm.2022.218
- Peñas-López et al. (2016), JFM 800, 180–212.
  https://doi.org/10.1017/jfm.2016.401
- Keller & Miksis (1980), JASA 68, 628–633.
  https://doi.org/10.1121/1.384720
- Prosperetti, Crum & Commander (1988), JASA 83, 502–514.
  https://doi.org/10.1121/1.396198
