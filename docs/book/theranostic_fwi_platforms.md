# Chapter 28 — Same-Device Therapeutic Ultrasound, Finite-Frequency Inverse, and RTM Monitoring

This chapter studies the tomotherapy-like ultrasound contract: the treatment
array is also the transmit/receive aperture for image formation and treatment
monitoring. The current implementation covers three CT-derived finite-frequency
inverse scenarios:

- 1024-element transcranial focused bowl around the head CT.
- 256-element concave abdominal array at the skin surface for
  KiTS19 kidney tumor CT.
- 256-element concave abdominal array at the skin surface for
  LiTS liver tumor CT.

The figures are fully synthetic, model-consistent simulations. They are not
measured HistoSonics, Verasonics, or INSIGHTEC device data, and they do not
claim proprietary element geometry. RITK owns NIfTI image ingestion; kwavers
owns CT preprocessing, device placement, exposure synthesis, finite-frequency
active Born inversion, passive subharmonic inversion, weak harmonic contrast
inversion, linear acoustic RTM, and reconstruction fusion through the PyO3
function:

```text
pykwavers.run_theranostic_inverse_from_ritk(...)
```

A separated nonlinear branch now runs the bounded 3-D therapy-monitoring
experiment through:

```text
pykwavers.run_theranostic_nonlinear_3d_from_ritk(...)
```

That branch does not relabel the reduced inverse. It resamples the RITK-loaded
CT/segmentation volume into an isotropic 3-D propagation grid, places the
same-aperture elements on the CT-derived skin/calvarium boundary, simulates
receiver data with a heterogeneous Westervelt FDTD update, performs a
source-encoded discrete-adjoint FWI update for sound speed and acoustic
nonlinearity `beta`, drives Rayleigh-Plesset bubbles from the resulting
peak-pressure field, and reconstructs the cavitation source from passive
subharmonic receiver data. The Westervelt adjoint stores exact sparse forward
checkpoints and replays bounded intervals during the reverse sweep rather than
retaining every pressure volume. The current nonlinear branch estimates `c`, `beta`,
and cavitation density as separated inverse blocks with a fused score; it is
not yet a joint `c/alpha/rho/beta/bubble-density` solve with one coupled
KKT/Gauss-Newton system.

Real systems cover adjacent pieces but not the exact deployed device implied by
the tomotherapy analogy. Transcranial ultrasound FWI focused bowls have been
demonstrated experimentally for imaging, Exablate Neuro uses a CT/MR-planned
1024-element therapeutic focused bowl, Edison histotripsy uses integrated diagnostic
ultrasound for liver targeting and bubble-cloud visualization, and research
histotripsy systems use passive acoustic feedback. The implemented workflow is
therefore a research simulation of a plausible same-aperture therapy/monitoring
platform, not a claim that HistoSonics or INSIGHTEC currently ship therapeutic
FWI reconstruction.

## 28.1 Mathematical Contract

### 28.1.1 Definition: Same-Device Aperture Contract

Let `E = {e_k ∈ ℝ³ : k = 1, …, N}` be the N element positions of the treatment
array in the patient coordinate frame.  The same-device contract holds iff
every transmitter index `s` and every receiver index `r` used in the imaging
inverse satisfy `s ∈ {1, …, N}` and `r ∈ {1, …, N}` — i.e. both are drawn
from the same element set as the therapy aperture.

**Consequence for the sensitivity operator:**
The active pitch-catch matrix `A ∈ ℝ^(m × n)` has rows indexed by transmit-receive
pairs `(s, r) ∈ E × E` and columns indexed by tissue voxels `j`.  No additional
imaging hardware (receive-only probe, rotating gantry) introduces elements
outside `E`.  The passive subharmonic, second-harmonic, and ultraharmonic rows
are indexed by `r ∈ E` for the receiver and carry no transmit index (passive).
The RTM wavefields inject and receive on the same `E` grid.

**Tomotherapy analogy:** In X-ray tomotherapy the same MLC-modulated gantry
position both irradiates the target and generates portal imaging data for
position verification.  Here the same phased-array aperture simultaneously
drives therapeutic pressure (therapy elements emitting) and samples acoustic
field data (the same or coaxial receive elements).  The contractual constraint
is that this be a single physical device position — no offline imaging session
with a separate probe.

### 28.1.2 Theorem: Same-Aperture Operator Rank

Let `A ∈ ℝ^(m × n)` be the same-device active Born operator with `m` pitch-catch
pairs drawn from `N` element positions and `n` active tissue voxels.  The
number of independent measurement rows is bounded by `m ≤ N²`.  For the
implemented abdominal case `N = 256`, `m_encoded ≤ 65,536`; for the brain case
`N = 1024`, `m_encoded ≤ 1,048,576`.  The actual encoded row count is
`⌊N_freq × N_receivers_used / rows_per_code⌋`, always `≤ N²`.

This bound motivates the deterministic row encoding: for `n ≫ N`, the problem
is underdetermined even at maximum transmit-receive diversity, and graph-Laplacian
regularization (`γ L`) provides the missing null-space structure through spatial
smoothness on the CT-derived tissue support.

---

For active pitch-catch imaging, the same source and receiver elements used for
treatment generate finite-frequency Born data:

```text
d_i = sum_j A_ij m_j
A_ij = dx^2 exp[-alpha_j f_MHz (r_sj + r_rj)]
       cos[k_f (r_sj + r_rj)] / sqrt(r_sj r_rj)
```

The inversion solves:

```text
(A^T A + lambda I + gamma L) m = A^T d
```

where `L` is the four-neighbor graph Laplacian on the active CT-derived tissue
support. The passive histotripsy channel replaces the transmit path with a
receiver-only subharmonic sensitivity at `f0/2`. The harmonic contrast channels
use the same aperture with second-harmonic rows at `2f0` and ultraharmonic rows
at `1.5f0`. Fusion gates the active lesion inverse by passive subharmonic
support plus harmonic and ultraharmonic contrast; the generated metrics report
the fused map separately from the individual channels.

### 28.1.3 Same-Device Send/Receive Passive Acoustic Mapping

The passive cavitation channels (subharmonic `f0/2`, ultraharmonic `3f0/2`)
support a second, forward-simulated reconstruction mode selected by
`TheranosticInverseConfig::passive_reconstruction`:

- `FiniteFrequencyOperator` (default): the reduced finite-frequency operator
  inverse described above, fitting a receiver-only subharmonic/ultraharmonic
  sensitivity against a synthetic target.
- `PassiveAcousticMapping`: a genuine passive-acoustic-mapping (PAM) pipeline
  that simulates the cavitation emission and beamforms the recorded receiver
  traces, with no synthetic inversion target.

In PAM mode the therapy array doubles as a passive receive aperture: between
therapy bursts the transmit elements switch to receive and join any dedicated
imaging receivers to form the same-device send/receive aperture. This mirrors
clinical practice — transcranial histotripsy acoustic-cavitation-emission (ACE)
mapping reuses the elements of the same hemispherical histotripsy transducer as
passive receivers (Sukovich et al. 2020) — and is required for the transcranial
helmet, which carries no separate imaging array.

The pipeline runs one broadband forward solve through the heterogeneous
CT-derived medium with the same fourth-order-FD / CPML solver used for the
active channels:

1. **Emission model.** The bubble cloud at the target radiates a deterministic
   broadband spectrum with the three PAM-relevant lines — subharmonic `f0/2`,
   driven fundamental `f0`, and ultraharmonic `3f0/2` — under a common Gaussian
   burst envelope. With an envelope of `N = 2.5` fundamental periods the
   spectral standard deviation is `Δf = f0 / (2πN) ≈ 0.064 f0`, far below the
   inter-line spacing `f0/2`, so the three lines are resolvable and the
   per-band filter rejects the fundamental (Neppiras 1980; Leighton 1994).
2. **One forward solve, all bands.** A single broadband emission is propagated
   and recorded at every receiver; the same trace set serves every imaged band.
3. **Aberration-corrected delays.** Receive delays are the eikonal
   first-arrival travel times solved through the *actual* heterogeneous medium
   (by reciprocity, one solve per unique receiver cell on the refined grid),
   not a constant-speed straight-line model. This keeps the coherent sum
   aligned through skull, rib, and water/tissue speed contrasts — the limiting
   factor for the higher-frequency cavitation band.
4. **Spectral PAM.** The *broadband* traces are delay-and-summed to a per-pixel
   time series (full bandwidth → fine range resolution), then a zero-phase
   Gaussian band-pass attributes each pixel's energy to a cavitation band.
   Band-passing the raw traces first would collapse the bandwidth to a single
   line and destroy range resolution (Gyöngy & Coussios 2010; Haworth et al.
   2012).

PAM mode is a forward-simulated beamforming reconstruction, not a full-waveform
inversion: it localizes cavitation from simulated emission physics and does not
update a medium model. It is implemented in
`theranostic_guidance::waveform::emission` and
`theranostic_guidance::solver::passive_pam_channels`.

The current kwavers implementation is a reduced finite-frequency Born inverse
plus a separate source-encoded linear acoustic time-domain RTM image. The RTM
image is computed from pressure-amplitude source injection, CT-derived
baseline and lesion-perturbed receiver traces, and adjoint residual
backpropagation over the full domain travel-time horizon. The default adjoint
source is a Charbonnier-robust residual scaled by the configured receiver-noise
fraction and observed-trace RMS, but it is not an
iterative multiparameter FWI update and not nonlinear Westervelt/
Rayleigh-Plesset propagation. The production contract is explicit:
`kwavers_therapy::therapy::theranostic_guidance`
owns patient CT workflow, anatomy selection, device-placement analogs, exposure
synthesis, and reconstruction reporting. The reduced finite-frequency
same-aperture row operators, passive subharmonic rows, harmonic rows, and
graph-Laplacian PCG normal-equation solver live under
`kwavers_solver::inverse::same_aperture`. General seismic FWI and RTM kernels
remain under `kwavers_solver::inverse::seismic`. The active same-aperture
inverse in this chapter precomputes its support graph Laplacian once, and each
CG step reuses row, normal-operator, and Laplacian workspaces instead of
allocating a full image mask inside every iteration.
The active, passive, harmonic, and ultraharmonic operators now satisfy a common
matrix-free `LinearOperator` contract: PCG applies `A x`, `A^T y`, and the
normal-equation diagonal without storing dense sensitivity rows. Dense
`RowMatrix` materialization remains available only as the verification oracle
and for bounded diagnostics. The PyO3 metrics expose
`operator_backend`, `operator_storage_values`, and `dense_operator_values` so
book runs can prove that the simulated 1024-element same-aperture path is not
using dense receiver-row storage. The reduced branch also applies
deterministic same-aperture row encoding before the PCG solve. Metrics expose
`inverse_encoding_rows_per_code`, `encoded_measurements`, and
`unencoded_measurements`, with `measurements` equal to the encoded row count.

The planned exposure map is now a heterogeneous wave measurement rather than a
geometric phasor shortcut. The solver uses the scalar acoustic equation

```text
p_tt = c(x)^2 Delta_4 p + s(x,t)
```

with CPML, CT-derived attenuation, and the same electronically steered source
delays as the RTM acquisition. The raw diagnostic field is
`P_peak(x) = max_t |p(x,t)|`. The displayed exposure keeps the legacy pressure
scale by reporting

```text
E(x) = P_source * P_peak(x) / max_{y in body} P_peak(y).
```

The PyO3 payload exports `exposure_raw_peak_pressure`, `exposure_model`,
`exposure_backend`, `exposure_uses_hybrid_pstd_fdtd`,
`exposure_source_count`, `exposure_time_steps`, `exposure_dt_s`, and
`exposure_workspace_values`. The active backend is
`reference_fdtd_cpml_2d`; `exposure_uses_hybrid_pstd_fdtd = false` is
intentional until the hybrid solver has source, receiver, CT-medium, peak-map,
and memory-accounting parity tests against this reference. The old phasor
exposure enforced constant-speed constructive interference at the nominal focus
and ignored skull, internal gas, attenuation, and finite-grid scattering. A
wave-solved exposure can therefore look less target-centered than the old
display while being the physically more faithful diagnostic; the raw field
localizes whether the divergence originates in medium scattering, steering
delays, or the downstream inverse channels.

### 28.1.4 Theorem: Exposure Backend Static-Dispatch Contract

Let `B` be a theranostic exposure backend implementing the peak-pressure
contract

```text
B(prepared, layout, config) -> (P_peak, E, diagnostics).
```

The Chapter 28 exposure entry point calls the backend through a generic type
parameter, not through `dyn` dispatch. Therefore the compiler monomorphizes the
reference backend call into a direct call with no vtable lookup and no
heap-erased solver object in the time-stepping loop. A future hybrid PSTD/FDTD
backend can be selected only by implementing the same contract and changing the
static backend type after differential tests prove equivalence or documented
improvement over `reference_fdtd_cpml_2d`.

Proof: Rust monomorphization emits a concrete instance of
`simulate_peak_pressure_with_backend::<B>` for each backend type `B`. The
backend method is statically resolved at compile time. Since the time-step loop
receives concrete slices and arrays from that implementation, backend
abstraction contributes no runtime dispatch term to the propagation recurrence.

### 28.1.5 Theorem: Rolling Peak-Pressure Workspace Bound

For a 2-D grid with `N = nx * ny` cells, the reference exposure backend retains
only:

```text
p[n-1], p[n], p[n+1], psi_x, psi_y, P_peak.
```

Therefore retained solve workspace is exactly `6N` scalar values, excluding
input medium maps and output arrays that are part of the clinical result.

Proof: the second-order acoustic recurrence needs two prior pressure states and
one destination state. CPML adds one memory field per coordinate direction in
2-D. The diagnostic peak map is updated by
`P_peak(x) <- max(P_peak(x), |p[n+1](x)|)` after attenuation. No receiver trace
buffer or forward checkpoint buffer is allocated for exposure. The boundary
halo clear touches only the finite-difference halo after buffer rotation, while
the next interior stencil overwrites all interior destination cells; replacing a
full-domain clear with a halo clear preserves the recurrence state and reduces
per-step clearing work from `O(N)` to `O(nx + ny)`.

The elastic shear channel is now an iterative nonlinear ElasticPSTD FWI
reconstruction rather than a reduced acoustic-operator comparator. Chapter 28
builds baseline, lesion-perturbed, and current-estimate shear media on the CT
slice, drives the commanded treatment focus inside the segmented target with an
out-of-plane velocity tone burst, records the same therapy/imaging aperture as
receivers, and repeatedly migrates receiver residual trace energy with the
shear travel-time condition

```text
t(x, r) = (min_s |s - x| + |x - r|) / c_s .
```

where `s` is the elastic push-source support. The exported
`elastic_shear_reconstruction` therefore comes from full time-domain elastic
propagation, nonlinear model resimulation, objective-decreasing line search,
and residual migration. Metrics expose
`elastic_shear_model`, `elastic_shear_receiver_count`,
`elastic_shear_time_steps`, `elastic_shear_dt_s`,
`elastic_shear_objective_history`, and baseline/lesion/final-residual trace
energies. The finite-frequency Born, passive, harmonic, and ultraharmonic
channels still use the matrix-free same-aperture PCG operators.

### 28.1.6 Definition: Patient-Adaptive Focused Transmit Schedule

The patient-adaptive transmit experiment is inspired by van Nierop et al.
`arXiv:2508.08782`, where sparse focused transmit selections are evaluated
against equispaced baselines at matched transmit budgets. Chapter 28 implements
the minimal local control surface:

```text
transmit_schedule_strategy in {full, uniform, patient_adaptive}
transmit_budget = k
```

All CT preprocessing, organ scenarios, device placement, finite-frequency row
operators, deterministic row encoding, graph-Laplacian PCG solve, fusion, and
metrics remain unchanged. The scheduler selects an ordered subset `S_k` from
the existing treatment aperture `E`; the inverse consumes `S_k` through the
same matrix-free operator path rather than through a cloned imaging pipeline.
The `patient_adaptive` schedule first chooses the element with maximal target
sensitivity

```text
q(e) = |T|^-1 sum_{x in T} 1 / max(||e - x||_2^2, dx^2)
```

and then greedily chooses each next transmit by maximizing
`q(e) min_{s in S} ||e - s||_2`, which preserves target sensitivity while
forcing aperture diversity. The reproducible comparison runs with:

```powershell
$env:KWAVERS_CH29_RENDER_SCOPE="adaptive_transmit"
python crates/kwavers-python/examples/book/ch29_theranostic_fwi_platforms.py
```

It writes `fig07_patient_adaptive_transmit_budget.png` and
`adaptive_transmit_metrics.json`, reporting active inverse Dice/CNR and
measurement counts for brain, kidney, and liver schedules at matched budgets.

### 28.1.7 Theorem: Deterministic Encoded Normal Equations

Let `A in R^(m x n)` be one same-aperture active, passive, harmonic, or
ultraharmonic sensitivity operator. Let `C in R^(k x m)` contain disjoint
contiguous row blocks whose nonzero entries are deterministic signs normalized
by the square root of the actual block size. The encoded operator is

```text
B = C A.
```

For any model `x`, `B x = C(A x)`. For any encoded residual `y`,
`B^T y = A^T C^T y`. Therefore the Chapter 28 reduced inverse solves the exact
encoded quadratic

```text
0.5 ||C(A m - d)||_2^2 + 0.5 lambda ||m||_2^2 + 0.5 gamma m^T L m
```

for the configured encoding. This reduces the number of PCG residual rows but
does not change the physics contract into nonlinear full-waveform inversion.

### 28.1.8 Theorem: Linear Acoustic RTM Imaging Condition

Let `p0(x,t)` solve the scalar acoustic wave equation in the baseline
CT-derived medium and let `p1(x,t)` solve the same equation after the lesion
speed perturbation. Receiver traces on the same therapy/imaging aperture are
`d0 = R p0` and `d1 = R p1`. The adjoint wavefield `lambda(x,t)` solves the
time-reversed acoustic equation with receiver residual injection
`R^T(d1-d0)`. The zero-lag image

```text
I(x) = integral p0(x,t) lambda(x,t) dt
```

is the reverse-time migration image for the source-encoded acquisition. It is a
complete forward/receive/adjoint solve for the stated linear acoustic PDE. It
is not a nonlinear treatment-physics model because cavitation, temperature,
elastic conversion, density inversion, and attenuation inversion remain outside
this linear PDE contract.

### 28.1.9 Theorem: Bounded Robust RTM Residual

For receiver residual `r = d1 - d0` and scale `epsilon > 0`, the Charbonnier
objective

```text
phi(r) = epsilon^2 (sqrt(1 + (r / epsilon)^2) - 1)
```

has adjoint-source derivative

```text
psi(r) = r / sqrt(1 + (r / epsilon)^2).
```

Then `|psi(r)| <= epsilon` for every finite residual. Proof: write
`x = |r| / epsilon`, so `|psi| = epsilon x / sqrt(1 + x^2) <= epsilon`.
The Chapter 28 RTM channel uses this derivative for receiver injection by
default. Setting `waveform_misfit = "l2"` recovers the unbounded least-squares
adjoint source `psi(r) = r`.

### 28.1.10 Theorem: Discrete Westervelt FWI Adjoint

The nonlinear 3-D branch advances pressure by the heterogeneous finite-amplitude
Westervelt recurrence

```text
(1 - 2 beta p[n] / (rho c^2)) dtt(p)[n]
  = c^2 Lp[n] + 2 beta (dt p[n])^2 / (rho c^2),

p[n+1] = S(2p[n] - p[n-1]
          + (c^2 dt^2 Lp[n]
             + 2 beta (p[n] - p[n-1])^2 / (rho c^2))
            / (1 - 2 beta p[n] / (rho c^2))
          + s[n]),
```

where `S` is the fixed polynomial absorbing layer, `L` is the 7-point 3-D
Laplacian, and the pressure-dependent inertia term is solved in the denominator.
This is the code path used for the Figure 5 and Figure 6 nonlinear pressure
diagnostics.

**Sign convention.** The continuous Westervelt equation in canonical form
(Westervelt 1963 Eq. 24; Hamilton & Blackstock 1998 §3.5 Eq. 3.10) is
`∇²p − (1/c²)·∂²p/∂t² + (β/(ρ₀c⁴))·∂²(p²)/∂t² = 0`. Solving for `p_tt` and
discretizing by the second-order leapfrog gives the recurrence above with a
**positive** finite-amplitude correction. The correction is required for forward
steepening: compressions travel faster than rarefactions and peaks at fixed `x`
arrive earlier than the linear prediction. The previous explicit `p*dtt(p)`
feedback path produced non-physical runaway peaks at histotripsy drive; the
finite-amplitude denominator keeps the 2026-05-17 Chapter 28 run finite while
reducing exactly to the linear update when `beta = 0`. A sign-flipped nonlinear
term produces non-physical reverse steepening; the sign-sensitive regression
`forward_westervelt_exhibits_physical_forward_steepening_with_corrected_sign`
locks this convention by checking that the steady-state receiver trace
satisfies `max(∂p/∂t) > |min(∂p/∂t)|`.

Receiver traces are compared to traces from a lesion-perturbed target medium.
The reverse sweep applies the transpose of each recurrence Jacobian and
accumulates

```text
d p[n+1] / d c =
  derivative of the finite-amplitude numerator/denominator update,

d p[n+1] / d beta =
  derivative of the finite-amplitude numerator/denominator update.
```

The denominator derivative carries the pressure-to-bulk-modulus factor
`beta/(rho c^2)` and the quotient rule; the adjoint reuses those cell terms so
the reverse sweep is the transpose of the exact discrete forward update.
Because the
time-unrolled recurrence graph is acyclic, this reverse accumulation is the
chain rule for the discrete least-squares objective. The implemented
objective stacks deterministic encoded source transmissions and adds discrete
`H1` penalties on `(c-c0)` and `(beta-beta0)`, restricted to the CT-derived
body support. Two regression tests lock this contract:
`nonlinear_3d_westervelt_fwi_and_cavitation_inverse_are_input_sensitive`
asserts finite objective values and a non-increasing FWI objective on a 3-D
CT-like abdominal fixture, and
`forward_westervelt_exhibits_physical_forward_steepening_with_corrected_sign`
asserts the forward physics is sign-correct.

### 28.1.11 Algorithm: Rayleigh-Plesset Cavitation Inverse

For each body voxel, the nonlinear branch first evaluates the FDA mechanical
index `MI = |P-|[MPa] / sqrt(f0[MHz])`. Voxels below the configured inertial
cavitation threshold are assigned zero source density. Voxels above that gate
integrate the incompressible Rayleigh-Plesset ODE with the local Westervelt
peak pressure as the acoustic forcing amplitude. The voxel source density is
the maximum period-doubled radius response, so the passive source depends on
bubble dynamics rather than on a hand-labeled lesion mask. A subharmonic Green
operator maps that source to the same receiver aperture, and the inverse solves
a nonnegative Tikhonov problem over the MI-gated Rayleigh-Plesset source support
by projected gradient descent with a step bounded by the operator Frobenius
norm.

The subharmonic Green operator is now **heterogeneous** and **path-integrated**.
The kwavers `Nonlinear3dVolume` carries a per-voxel `attenuation_np_per_m_mhz`
field derived in `material_maps` from CT HU and segmentation labels using
tissue classes from Hamilton & Blackstock 1998 §4.1 (Table 4.1) and the
Connor & Hynynen 2002 transcranial bone-attenuation measurement:

| Tissue class                         | HU range             | `α₀` [dB/(cm·MHz)] | `α` [Np/m] at 1 MHz | `y`  |
|--------------------------------------|----------------------|--------------------|---------------------|------|
| Cortical bone (skull)                | HU ≥ 300             | 13 → 20 (by HU)    | 149.7 → 230.3       | 2.00 |
| Segmented organ (brain/liver/kidney) | label > 0            | 0.6                | 6.91                | 1.05 |
| Generic soft tissue / muscle / fat   | HU > −700, label = 0 | 0.5                | 5.76                | 1.05 |
| Air pocket                           | HU < −700, label = 0 | (1000)             | 1000                | 1.00 |
| Outside body                         | body mask false      | 0                  | 0                   | 1.00 |

The frequency dependence of the attenuation follows the per-voxel power law
`α(f) = α(1MHz) · f_MHz^y`, where `y` ranges over tissue classes:

- **Soft tissue / organ**: `y ≈ 1.05` (Treeby & Cox 2010 Table I; near-linear)
- **Cortical skull bone**: `y ≈ 2.0` (Connor & Hynynen 2002 measured 1.9 - 2.0
  across 0.5 - 3.5 MHz; matches classical Stokes-Kirchhoff viscous limit)

For the brain focused bowl at a 650 kHz drive (325 kHz subharmonic), the skull
attenuation with `y = 2` is `α(0.325 MHz) = α(1 MHz) · 0.325² ≈ 0.106 ·
α(1 MHz)` — about 3× smaller than a naive `y = 1` extrapolation predicts.
Without this correction, transcranial passive cavitation receive would be
over-attenuated and the cavitation inverse would be inappropriately starved
of information.

The Green's kernel for the source at voxel `s` and receiver at voxel `r` is

```text
G_s(r, s) = exp(-integral alpha_s(t) dt along [s -> r]) * cos(k_s * |r - s|)
          / (4 pi |r - s|)
```

where `alpha_s(t) = alpha_1MHz(position) * f_s_MHz^y(position)`. The path
integral samples both the attenuation and exponent fields along the straight
line from source to receiver with trilinear interpolation and trapezoidal-rule
integration. For transcranial cases this correctly tracks the skull-versus-
soft-tissue attenuation contrast and the subharmonic `y = 2` skull scaling on
every ray instead of using a single tissue-typical scalar.

**Why the period-doubling observable isolates inertial cavitation:**
Under linear (stable) bubble oscillation at drive frequency `f₀`, the
radius evolves as `R(t) = R₀ + ε·cos(ω₀t + φ)`.  Period-doubling occurs
when the Floquet multiplier of the linearized Rayleigh-Plesset eigenvalue
problem crosses −1, signalling a period-2 bifurcation at `f₀/2` (subharmonic).
For a period-2 orbit, `R(t + T) = R(t)` with `T = 2/f₀`, but
`R(t + 1/f₀) ≠ R(t)`.  Therefore `|R(t) − R(t − 1/f₀)|` is nonzero in the
period-doubling regime and zero (to leading order) in the linear regime.
Normalising by `R₀` and taking the maximum over one period gives the
dimensionless period-doubling amplitude:

```text
δ_PD = max_t |R(t) - R(t - T_drive)| / R₀,   T_drive = 1/f₀
```

which is `≈ 0` for stable bubbles, large (O(1)) for inertially collapsing
bubbles, and intermediate for subharmonic-emitting bubbles.  The implementation
uses the mechanical-index gate to restrict the source domain to inertial
cavitation candidates, then uses this period-doubling observable to distinguish
the passive subharmonic emission source (Dirichlet cavitation indicator) from
stable linear oscillation without using lesion labels or a learned classifier.

The kwavers implementation stores one drive-period ring buffer of radius samples
(O(f₀/f_sample) samples), algebraically equivalent to retaining the full radius
history for this scalar because the recurrence reads only the sample one period
behind the current RK4 state. The passive simulated data is generated from the
same MI-gated source-support vector used by the Green-operator columns, so the
receiver data and inverse model share the same physically admissible support.

### 28.1.12 Theorem: Positive Normal Operator

Let `A` be the finite-frequency same-aperture sensitivity matrix, `lambda > 0`,
`gamma >= 0`, and `L` the four-neighbor graph Laplacian on the CT-derived active
tissue support. The operator

```text
H = A^T A + lambda I + gamma L
```

is symmetric positive definite on the active support. Therefore the Chapter 28
PCG solve minimizes the unique quadratic objective:

```text
0.5 ||A m - d||_2^2 + 0.5 lambda ||m||_2^2 + 0.5 gamma m^T L m
```

Proof: `A^T A` is positive semidefinite by construction. The graph Laplacian
satisfies `m^T L m = sum_(i,j in E) (m_i - m_j)^2 >= 0`. Adding
`lambda I` makes the quadratic form strictly positive for every nonzero `m`.

### 28.1.13 Algorithm: Same-Aperture Monitoring Loop

1. Load CT/NIfTI with RITK and convert intensities into anatomy-specific
   acoustic property maps.
2. Build the treatment aperture in the patient coordinate frame: calvarial
   focused bowl for brain or external skin-normal arc for abdomen.
3. Simulate the planned exposure with the same heterogeneous scalar acoustic
   forward solver used by the RTM branch, retaining only rolling wavefields and
   a raw peak-pressure accumulator.
4. Instantiate active pitch-catch, passive subharmonic, second-harmonic, and
   ultraharmonic matrix-free operators from the same aperture and receiver set.
5. Encode the reduced inverse rows by deterministic normalized signs and record
   both encoded and unencoded row counts in the PyO3 metrics.
6. Simulate source-encoded baseline and lesion-perturbed acoustic wavefields
   with pressure-amplitude source injection, record receiver traces on the same
   aperture through the CT-domain travel-time horizon, and backpropagate the
   robust residual traces for a linear RTM image.
7. Run iterative nonlinear ElasticPSTD shear FWI from the commanded target
   focus: simulate baseline, observed lesion, and current-estimate shear media;
   record velocity traces on the same aperture; migrate receiver residual trace
   energy; and accept each shear-map update only when the trace objective drops.
8. Solve each reduced inverse channel with the same graph-Laplacian PCG core.
9. Fuse active lesion inverse output with passive and harmonic support maps for the
   monitoring image.

### 28.1.14 Algorithm: Nonlinear 3-D Branch

1. Load the same CT/NIfTI inputs with RITK.
2. Resample a target-and-skin/calvarium 3-D support into an isotropic bounded
   grid.
3. Convert HU and segmentation labels into background and lesion-perturbed
   `c`, `rho`, `beta`, body, and target volumes.
4. Select CT-boundary source/receiver cells on the calvarium cap or abdominal
   skin-facing support.
5. Build a CT/segmentation-derived inversion mask by dilating the planned
   target support inside the propagated body support; propagation remains
   whole-volume, while `c/beta` updates are restricted to the treatment ROI.
6. Simulate deterministic source-encoded observed traces with lesion-perturbed
   Westervelt propagation and predicted traces from the current multiparameter
   model.
7. Update `c` and `beta` with the discrete Westervelt adjoint, `H1`
   regularization, Sobolev-smoothed gradients, and monotone line search. The
   forward history is retained as exact sparse checkpoints containing
   `p[n-2]`, `p[n-1]`, and `p[n]`; each reverse segment replays one bounded
   forward interval with the same recurrence, so the gradient matches a dense
   history while retained forward state scales as
   `O((steps / interval + interval) * cells)`. The reverse sweep uses four
   rolling adjoint states instead of storing one adjoint volume per timestep;
   the temporal stencil width is three, so this is algebraically identical to
   the dense time-adjoint while reducing adjoint state memory from
   `O(steps * cells)` to `O(cells)`.
8. Integrate Rayleigh-Plesset dynamics from peak pressure and invert the passive
   cavitation source with a nonnegative subharmonic operator.
9. Fuse the normalized multiparameter FWI score with the passive cavitation
   reconstruction by a fixed convex weight favoring the active nonlinear
   estimate, so passive evidence can add support without suppressing the
   active estimate when cavitation is absent or spatially weak.

## 28.2 Device Placement

The brain case reads the CT-aligned target and transducer pose from
`CANONICAL_BRAIN_SCENE` in
`crates/kwavers-python/examples/book/transcranial_planning/scene.py`. The slice-level
finite-frequency inverse, Figure 5 nonlinear branch, and separate CT-derived
3-D focused-bowl placement resolve the same target fraction against their CT support
and use the same 1024-element, 650 kHz, 0.150 m focused-bowl definition. The 3-D view
renders the head surface, dense skull/calvarium surface points, the calvarium
focused-bowl element cloud, sampled beam paths, and the first dense-bone intersection
on each sampled beam. The focused-bowl cap covers the calvarium: polar angles `[θ_min, θ_max] =
[0.22 rad, 1.18 rad]` from the superior vertex (12.6°–67.6° from the skull
vertex). These bounds
are owned by `CANONICAL_BRAIN_SCENE.transducer.cap_min_polar_rad` and
`cap_max_polar_rad` in `transcranial_planning/scene.py` and are threaded through
`focused_bowl_pykwavers_kwargs()` into `plan_transcranial_focused_bowl_placement`.
The axis-projection bounds are `[cos(1.18), cos(0.22)] = [0.381, 0.976]`.
Superior orientation is detected from the CT axial area profile so the
cap remains on the calvarium and does not extend to the neck. The
abdominal cases place a concave 256-element therapy arc outside the nearest
external skin point to the target centroid, using a local skin-normal aperture
frame instead of a fixed left/right display axis. Internal gas pockets are
excluded from the skin candidate set by flood-filling exterior air from the CT
border. A central 64-receiver imaging line occupies the therapy-head cutout.
The PyO3 result exports `placement_metrics` and a separate full-CT
`placement_context`; figure 1 uses the uncropped patient slice for kidney and
liver so the skin interface is visible relative to the stomach/hip
cross-section rather than only the local tumor field of view. If an abdominal
segmentation slice contains multiple disconnected label-2 regions, one Chapter
29 run represents one physical sonication and therefore selects the largest
connected label-2 treatment component for focus placement, exposure synthesis,
lesion-source definition, metrics, and plotted contours. Covering all separated
targets requires a staged multi-sonication plan rather than one single-focus
exposure.

The Verasonics-like role in this simulation is the programmable acquisition
contract rather than a fixed clinical transducer geometry: each case exposes
source count, receiver offsets, frequency list, pressure scale, and raw
same-aperture active/passive channel synthesis through
`run_theranostic_inverse_from_ritk`.

The abdominal geometry is an Edison-like research surrogate, not an Edison
device specification. Public Edison documentation describes pulsed histotripsy
with live bubble-cloud monitoring, an integrated diagnostic ultrasound probe,
and treatment heads in the 52/56-element class. The 256-element arc used here
comes from the 2025 liver aberration-correction simulation literature because
it provides a reproducible equal-area treatment aperture, central imaging
cutout, and CT-derived aberration-correction envelope for book figures and
kwavers validation.

The simulated pressure scale is explicit. Figure 2's reduced brain
receive/imaging branch uses `1.5e5 Pa`. The separated Figure 5/Figure 6
nonlinear histotripsy branch uses `28.0e6 Pa` for brain, kidney, and liver, and
the abdominal nonlinear branch defaults to the 500 kHz treatment frequency from
the case frequency list. The 2026-05-17 run records finite target mechanical
indices of `20.91`, `3.03`, and `2.33` for brain, kidney, and liver. The
linear exposure image is a normalized display of a real heterogeneous
peak-pressure solve, and the raw peak-pressure array remains available for
absolute diagnostic inspection.

Figure 2 begins each row with the same CT placement slice, segmented target
overlay, body outline, and transducer coordinates used for that case, then
shows the simulated exposure, target mask, and linear positive reconstruction
maps. The CT column fixes the anatomical targeting and device-placement context
before the derived inverse/RTM channels. Figure 4 displays the same active,
passive, harmonic, ultraharmonic, and fused reconstructions on a common
`[-40, 0] dB` relative-amplitude scale and reports outside-target peak and
energy fractions. Those dB diagnostics separate finite-frequency aperture
sidelobes from treated tissue response; coherent rings visible in the same-
aperture inverse maps are point-spread-function structure, not additional targets.

## 28.3 Minimal Usage Example

```python
import pykwavers as kw
import numpy as np

# Same-device workflow: one CT path, one anatomy string, same aperture
# for therapy and monitoring. NIfTI files do not need to exist;
# synthetic CT phantoms are used automatically when paths are absent.

result = kw.run_theranostic_inverse_from_ritk(
    ct_nifti_path="data/my_patient/ct.nii.gz",   # or "/nonexistent" for synthetic
    segmentation_nifti_path=None,                  # required for abdomen + real CT
    anatomy="brain",                               # "brain", "liver", or "kidney"
    grid_size=48,                                  # 3-D FDTD cube size for nonlinear
    element_count=1024,                            # same-device element count
    iterations=12,
    frequencies_hz=[220e3, 650e3],                # multi-frequency Born rows
    receiver_offsets=[256, 384, 512, 640],         # within 0..element_count
    source_pressure_pa=1.5e5,                      # diagnostic: 150 kPa
    target_fraction_xyz=(0.59, 0.50, 0.49),        # VIM-like atlas fraction
    noise_fraction=0.012,
    waveform_misfit="charbonnier",                 # robust adjoint injection
)

# Inspect the same-device operator evidence
print("operator backend:", result["operator_backend"])    # "matrix_free_row"
print("dense operator values:", result["dense_operator_values"])  # 0 (not stored)
print("encoded measurements:", result["encoded_measurements"])
print("unencoded measurements:", result["unencoded_measurements"])
print("is_full_wave_inversion:", result["is_full_wave_inversion"])  # False: acoustic inverse is reduced-Born + Tikhonov, not FWI
print("iterative_elastic_fwi:", result.get("iterative_elastic_fwi"))  # True: elastic-shear channel is iterative FWI

# Reconstruction channels (all from the same E aperture)
active = np.asarray(result["active_lesion_reconstruction"])
rtm    = np.asarray(result["waveform_rtm_reconstruction"])
fused  = np.asarray(result["fused_reconstruction"])
print("fusion Dice:", result["metrics"]["fusion"]["dice_equal_area"])
print("fusion CNR:", result["metrics"]["fusion"]["cnr"])
```

The nonlinear 3-D branch (separate call):

```python
nonlinear_result = kw.run_theranostic_nonlinear_3d_from_ritk(
    ct_nifti_path="data/my_patient/ct.nii.gz",
    anatomy="brain",
    grid_size=48,
    element_count=1024,
    frequency_hz=650e3,
    source_pressure_pa=1.5e5,
    target_fraction_xyz=(0.59, 0.50, 0.49),
)
print("is_full_wave_inversion:", nonlinear_result["is_full_wave_inversion"])  # True
print("uses_nonlinear_wave_propagation:",
      nonlinear_result["uses_nonlinear_wave_propagation"])                    # True
```

## 28.4 Figures

Run:

```powershell
python crates/kwavers-python/examples/book/ch29_theranostic_fwi_platforms.py
```

Outputs:

- `docs/book/figures/ch29/fig01_device_placement_on_ct.{png,pdf}`
- `docs/book/figures/ch29/fig02_exposure_and_reconstruction.{png,pdf}`
- `docs/book/figures/ch29/fig03_brain_focused_bowl_3d_placement.{png,pdf}`
- `docs/book/figures/ch29/fig04_reconstruction_dynamic_range_diagnostics.{png,pdf}`
- `docs/book/figures/ch29/fig05_nonlinear_3d_westervelt_rp_cavitation.{png,pdf}`
- `docs/book/figures/ch29/fig06_controlled_linear_nonlinear_comparison.{png,pdf}`
- `docs/book/figures/ch29/metrics.json`
- `docs/book/figures/ch29/controlled_comparison_metrics.json`
- `docs/book/figures/ch29/controlled_comparison_fields.npz`

![Same-device array placement on the CT-derived skin/skull boundary.](figures/ch29/fig01_device_placement_on_ct.png)

*Figure 28.1. Device placement on CT: the same-aperture therapy/monitoring array positioned on the CT-derived boundary for brain, kidney, and liver cases.*

![Planned exposure and same-aperture reconstruction channels.](figures/ch29/fig02_exposure_and_reconstruction.png)

*Figure 28.2. Exposure + reconstruction: the planned focused exposure and the active-Born / passive-subharmonic / harmonic / RTM / fused reconstruction channels, all from the same element set $E$.*

![3-D placement of the 1024-element focused bowl on the head CT.](figures/ch29/fig03_brain_focused_bowl_3d_placement.png)

*Figure 28.3. Transcranial focused-bowl placement: the 1024-element hemispherical aperture around the head CT with source-to-focus beam paths.*

![Reconstruction dynamic-range and sidelobe diagnostics.](figures/ch29/fig04_reconstruction_dynamic_range_diagnostics.png)

*Figure 28.4. Reconstruction diagnostics: dynamic-range and outside-target sidelobe diagnostics for the same-aperture channels.*

![Nonlinear 3-D Westervelt / Rayleigh–Plesset cavitation reconstruction.](figures/ch29/fig05_nonlinear_3d_westervelt_rp_cavitation.png)

*Figure 28.5. Nonlinear 3-D branch: the heterogeneous-Westervelt FDTD field, discrete-adjoint $c/\beta$ FWI, and the Rayleigh–Plesset cavitation-source reconstruction from passive subharmonic data.*

![Controlled linear vs nonlinear reconstruction comparison.](figures/ch29/fig06_controlled_linear_nonlinear_comparison.png)

*Figure 28.6. Controlled comparison: matched-setup linear (reduced-Born) vs nonlinear (3-D Westervelt/RP) panels audited against the same experimental scene.*

The metrics file records reconstruction quality, placement geometry, the
canonical brain scene manifest,
outside-target sidelobe diagnostics, matrix-free operator storage evidence,
RTM waveform misfit metadata, nonlinear 3-D Westervelt/Rayleigh-Plesset
metrics, nonlinear source-encoding/regularization controls, and model-fidelity
flags, including the nonlinear forward-checkpoint interval. The nonlinear FWI
loop reuses one residual trace workspace across source encodings and one
candidate `c`/`beta` workspace across line-search scales; this changes allocator
pressure only, not the objective or update equations. The same-device inverse
cases now report `is_full_wave_inversion = false` because the acoustic
anatomy / lesion / harmonic / ultraharmonic channels are reduced-Born /
Tikhonov inversions (one-shot, linearised) and the 2-D RTM channel is a
single-pass adjoint imaging condition — none of which is FWI. The
elastic-shear channel separately reports `iterative_elastic_fwi = true`
because it performs iterative nonlinear ElasticPSTD FWI with line search;
this fact is exposed on its own flag rather than being conflated with the
acoustic inverse above. The acoustic channels still report
`uses_nonlinear_wave_propagation = false` because the forward solver
integrates the linear scalar wave equation. The separate nonlinear 3-D
cases report `is_full_wave_inversion = true`,
`uses_nonlinear_wave_propagation = true`, and `uses_rayleigh_plesset = true`.
Figure 5 uses `56^3` nonlinear
Westervelt/Rayleigh-Plesset volumes by default for brain, kidney, and liver.
For development smoke runs, `KWAVERS_CH29_OUT_DIR` redirects generated figures,
metrics, and field archives away from the production book directory. The
loader rejects stale `pykwavers` extensions that do not expose the current
nonlinear argument surface, preventing old PyO3 binaries from silently running
with mismatched Chapter 28 controls.
Figure 5 also reuses the Figure 2 brain
scene target, therapy aperture, imaging receivers, and sampled source-to-focus
beam paths in the CT column so the linear and nonlinear panels are visually
audited against the same experimental setup. The Figure 5 `planned exposure`
column is the Figure 2 pressure-calibrated exposure field plotted in the same
placement axes. The separate `Westervelt FDTD peak` column is the nonlinear
3-D pressure-peak diagnostic on the cropped simulation cube. The nonlinear
solver still computes on a cropped isotropic 3-D cube for tractable Westervelt
FWI, but its exported crop bounds are rendered inside the uncropped Figure 2
placement axes; the panels therefore share the same visual CT field of view
without geometrically stretching the nonlinear crop. Abdominal nonlinear source
selection inherits the Figure 2 target-slice skin-contact direction before the
3-D crop is built, then preserves target-facing angular order when sampling
the 3-D source and receiver sets. This keeps kidney and liver treatment elements
distributed around the intended skin-coupled cap instead of lexicographically
sampling the CT boundary. Reduced inverse
metrics also report the encoded and unencoded measurement counts used by
deterministic row encoding.

Figure 6 adds the controlled Figure 2/Figure 5 comparison path. It runs a
matched linear case with the nonlinear grid size, element count, drive
frequency, and source pressure, then evaluates the linear fields and nonlinear
3-D slab fields after physical resampling onto the full-resolution CT placement
grid used by the far-left CT panel. That CT-frame view includes target/body contours,
therapy tx/rx elements, central imaging receivers, focus, and skin-contact
marker, and the reconstruction panels use the same CT extent and pixel grid
instead of a smaller nonlinear subsection. The metrics record the remaining
projected 2-D-vs-3-D aperture residual from the source wrappers.
The controlled grid now includes the iterative nonlinear ElasticPSTD FWI shear
reconstruction and stores `ct_frame_elastic_shear` beside the linear and
nonlinear fields so all reported comparison channels use the full CT placement
pixel grid. The panel order places the **Born inverse** reconstruction
(`active_lesion_reconstruction`) directly beside the FWI reconstructions — the
Westervelt target-pressure field, the iterative elastic-shear FWI estimate, and
the Westervelt-plus-cavitation FWI fusion — so the linearized reduced-Born and
the nonlinear full-waveform reconstructions are read side by side on one
physical grid rather than in separate figures. The final panel is the
FWI-fusion-minus-Born difference on the same CT frame. Each reconstruction is
drawn as a signal-proportional translucent overlay (opacity scales with the
square root of the normalized field magnitude) on top of its CT anatomy, so the
focus and lesion stand out against visible tissue rather than as a hard-edged
crop square; the white contour is the matched target and the full transducer
placement context is carried in the first CT column. Figure 6 also prints the
CT-frame comparison theorem and caption directly above and below the panel grid.
The controlled metrics also convert nonlinear pressure hotspots into CT-frame
physical coordinates and decompose their target offset into planned beam-axis
and cross-axis terms. The geometry block records the planned-to-realized
nonlinear aperture axis angle and nonlinear source-to-target distance range.
Those diagnostics distinguish prefocal/postfocal pressure-gain errors from
lateral aperture or phase-realization errors without relying on cropped
subpanels.
The 2026-05-18 controlled run uses the full CT placement grid for all panels.
The brain linear branch now resolves the canonical 3-D target fraction once in
the full CT volume, maps that index through the actual resampled head crop, and
exports the crop bounds used for CT-frame projection. This removes the previous
2-D slice-fraction target drift: brain `linear_focus_to_common_target_centroid_m`
is `0.0004366 m`, and the brain linear exposure, linear fusion, and elastic
shear hotspots all lie inside the full-CT target mask with hotspot distances
`14.32`, `13.60`, and `13.93` CT pixels. The visible nonlinear pressure panel is
now the simulated target pressure inside the matched lesion mask; the
treatment-window pressure and raw body/coupling pressure remain archived as
`nonlinear_pressure_window` and `nonlinear_pressure_raw` with prefixed
localization metrics. Mean common-grid linear-fusion Dice is `0.716`, elastic
shear Dice is `0.802`, nonlinear-fusion Dice is `0.535`, displayed nonlinear
target-pressure outside-target energy fraction is `0.000`, and MI-gated
Rayleigh-Plesset source outside-target energy fraction is `0.599`. Liver linear
exposure now peaks on the selected target (`target_peak=1.000`, hotspot distance
`5.35` CT pixels). Liver target MI is `4.28` at `3.02 MPa`;
the displayed target-pressure peak is inside the target and `8.87 mm` from
the target centroid, the treatment-window pressure hotspot is `17.78 mm`
from the target centroid, and the archived raw body-pressure hotspot remains
`103.74 mm` prefocal. The measured electronic-steering calibration hotspot is
`4.58` nonlinear-grid cells from the nominal liver focus; the bounded measured
steering search selected zero correction because the candidate shifted foci
reduced target/window pressure ratio. This separates true source-side pressure
gain and treatment-window spread from the clinically relevant target-mask
histotripsy forcing instead of deleting the raw diagnostic evidence.
The metrics writer serializes per-iteration FWI diagnostics: objective before
and after line search, sound-speed and nonlinearity gradient norms, accepted
scale, and accepted block (`coupled`, `speed_only`, `beta_only`, or `none`).
The field archive stores the native comparison fields and `ct_frame_*` fields
at full CT placement resolution. The full-CT comparison rejects cropped display
resolution as the primary failure mode. After correcting the finite-amplitude
Westervelt update, bounding source injection, preserving abdominal aperture
ordering, using the 500 kHz abdominal treatment frequency, and constraining
Rayleigh-Plesset source support to the treatment inversion window, the nonlinear
target masks exceed the inertial-cavitation threshold. The remaining defect is
inverse sensitivity plus off-target pressure focusing: kidney target cavitation
is present, but its FWI branch still does not accept a decreasing update on the
full generated run, and liver remains dominated by off-target source energy
before passive inversion.

The 2026-05-17/2026-05-18 steering follow-up removes multiple causes of
off-target pressure:
the abdominal nonlinear crop now includes the target-to-skin acoustic path
instead of only the target treatment window, the planned skin contact is reused
as the nonlinear focused-bowl vertex, abdominal source cells are selected in
the exterior coupling medium, and electronic steering delays are computed from
straight-ray CT slowness rather than one homogeneous focus-cell sound speed.
The follow-up targeting correction also keeps the outside focused-bowl standoff
inside the nonlinear crop and replaces abdominal point injection with
finite-area exterior-coupling source patches. The measured reduced kidney run
uses `24` to `40` non-body coupling cells per active element, with mean support
`29.81` cells and a CT-slowness focused-delay span of `9.679e-6 s`, so the
simulated aperture no longer injects pressure directly into tissue voxels or
collapses each bowl element to one skin-plane grid cell.
The 2026-05-18 source correction changes finite-area source weights from
point-source sum normalization to pressure-boundary peak normalization, so
increasing the number of grid cells in one element no longer reduces the
configured surface pressure. A `40^3`/64-element liver check changed target MI
from `0.74` to `5.28` at the same 28 MPa drive, confirming that the previous
normalization was a grid-dependent source-pressure defect.
Figure 5 now separates overlay semantics: the planned exposure panel uses the
Figure 2 planned aperture, while nonlinear pressure, FWI, cavitation, and
fusion panels draw the actual nonlinear 3-D aperture projection and nonlinear
target centroid in the full CT placement frame. The visible Westervelt pressure
panel renders target-mask pressure on the matched CT frame; diagnostics
separately report raw global, body, coupling/source, treatment-window, and
target peaks. A
reduced `24^3` KiTS19 verification run at histotripsy-scale drive now reports
target MI `2.55`, FWI objective `2.4165e-5 -> 1.7785e-5`, `target/body`
peak ratio `0.513`, coupling/body ratio `1.11`, and body-hotspot distance
`14.93` grid cells. The same reduced run at `6 MPa` gives target MI `0.548`
and `target/body` ratio `0.509`, confirming coherent scaling below the
histotripsy threshold. These runs are only localization diagnostics because
their minimum spatial sampling is
`0.290` points per wavelength at 500 kHz, below the configured six-point
contract. The remaining focusing gap is therefore a resolved-grid pressure-gain
and dispersion problem, not the absence of a physical aperture, a known
beam-overlay mismatch, direct tissue source injection, or a reversed
electronic-steering delay law. For skull, the phase correction is the scalar
straight-ray slowness integral through the CT-derived sound-speed map; it is not
an elastic skull correction with shear mode conversion or refracted ray bending.

### 28.4.1 Acoustic Scalar Model

The Figure 5 nonlinear branch is a scalar acoustic pressure model. It uses
CT-derived sound speed, density, nonlinearity coefficient, attenuation, a
heterogeneous Westervelt pressure update, and a Rayleigh-Plesset cavitation
post-process. It does not use elastic displacement, shear speed, shear modulus,
Lame parameters, or mode-converted skull shear propagation. Elastic skull
effects remain outside this chapter's nonlinear implementation contract.

The nonlinear 3-D material map treats cells outside the body mask as coupling
fluid rather than CT air. This matches the Figure 2 placement abstraction,
where therapy elements couple through water/gel before entering tissue. Air
pockets inside the body retain the gas sound-speed, gas density, gas
nonlinearity, and high-attenuation contract, but exterior CT background does
not slow the simulation to a 343 m/s gas domain or distort the source focusing
delays. The body mask distinguishes these cases by flood-filling
boundary-connected exterior air first; enclosed HU `< -700` label-0 voxels are
kept as internal gas, while boundary-connected CT air remains coupling fluid.

### 28.4.2 FDTD/PSTD Choice

The nonlinear branch uses FDTD for the Westervelt forward and discrete adjoint
because every operator in the inversion has a local transpose: heterogeneous
`c`, `rho`, `beta`, attenuation fields, source injection, receiver sampling,
sponge damping, sparse checkpoints, and the `dtt(p^2)` nonlinearity. This gives
an exact discrete-adjoint implementation for the recurrence actually used by
the inverse.

PSTD or k-space propagation would reduce numerical dispersion in smooth media,
but it would add global FFT operators, boundary bookkeeping, and heterogeneous
coefficient products whose adjoint must be derived and verified as the new
single source of truth. The current spectral component is limited to the
fractional-Laplacian absorption operator; the primary Westervelt inverse remains
FDTD until a full spectral forward/adjoint pair is implemented and differentially
verified.

## 28.5 Research Alignment

The implemented channels follow the current research direction as of
2026-05-13:

- Brain FWI: Guasch et al. demonstrate the seismic analogy for adult brain
  imaging with a 1024-transducer focused bowl where each element acts as source and
  receiver, and the inversion uses transmitted, reflected, diffracted,
  multiple-scattered, and guided waves rather than beamforming
  ([npj Digital Medicine, 2020](https://www.nature.com/articles/s41746-020-0240-8)).
- Transcranial correction burden: recent work isolates attenuation treatment in
  transcranial FWI
  ([EMBC 2024](https://pubmed.ncbi.nlm.nih.gov/40039691/)) and shows that
  full-wave skull simulations must handle both aberration and reverberation
  ([Phys. Med. Biol., 2025](https://pubmed.ncbi.nlm.nih.gov/40695316/)).
- USCT FWI: multiparameter soft-tissue FWI now uses hierarchical frequency
  continuation and optimal-transport misfits for sound speed plus impedance
  ([Ultrasonics, 2025](https://pubmed.ncbi.nlm.nih.gov/39615188/)); source
  encoding reduces wavefield solve count by an order of magnitude in
  vortex-encoded UCT FWI
  ([JASA, 2025](https://pubmed.ncbi.nlm.nih.gov/40197542/)); polar-coordinate
  structural-prior INR-FWI targets cycle skipping in ring-array USCT
  ([MICCAI 2025](https://papers.miccai.org/miccai-2025/0662-Paper2163.html)).
  The current kwavers increment follows the same scaling pressure by removing
  dense row storage through a matrix-free same-aperture operator, applying
  deterministic normalized row encoding to the reduced normal-equation
  channels, and adding deterministic source encoding to the separated nonlinear
  3-D Westervelt FWI branch. The reduced channels still solve encoded linear
  quadratics and do not perform nonlinear FWI.
- Recent FWI cycle-skipping controls: low-frequency extrapolation by sparse
  deconvolution targets the missing-low-frequency problem in practical USCT
  ([Ultrasound Med. Biol., 2025](https://www.sciencedirect.com/science/article/pii/S0301562925001097)),
  and the HV metric is a signed-signal transport alternative to `L2` and
  Wasserstein objectives for time-domain FWI
  ([arXiv 2508.17122](https://arxiv.org/abs/2508.17122)). A 2026
  ring-array USCT study extends the same direction to joint sound-speed and
  attenuation FWI with optimal transport plus sigmoid regularization
  ([Ultrasonics, 2026](https://www.sciencedirect.com/science/article/pii/S0041624X26000533)).
  These motivate the current Charbonnier-robust RTM residual for noise-bound
  adjoint injection. Optimal-transport and HV metrics remain future waveform
  misfit strategies because they require a different trace-space objective,
  not a relabeling of the current reduced inverse.
- RTM/FWI method split: ultrasonic full-matrix-capture studies compare TFM,
  RTM, and FWI, with RTM serving as a one-pass localization image and FWI as
  the iterative material-property update
  ([Ultrasonics, 2025](https://portal.fis.tum.de/en/publications/quantitative-comparison-of-the-total-focusing-method-reverse-time)).
- Transducer-source accuracy: 2026 distributed-source inversion work shows
  that aperture-dependent phase and amplitude calibration materially affects
  RTM/FWI quality, so the next kwavers physics increment should estimate or
  calibrate effective element source models instead of assuming ideal point
  elements
  ([arXiv 2603.24415](https://arxiv.org/abs/2603.24415)).
- HistoSonics-like abdomen geometry: the liver histotripsy envelope study uses
  a simulated transducer with similar dimensions to Edison, 256 elements,
  `750 kHz`, `14.2 cm` focal radius, `23 cm` maximum lateral extent, and a
  `4 cm` central cutout, then evaluates CT-derived aberration correction at a
  `26 MPa` focal-pressure envelope
  ([Phys. Med. Biol., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12679210/)).
- Passive therapy monitoring: passive cavitation mapping is the relevant
  receive-only treatment-monitoring analog, and 2025 higher-order DMAS work
  reports improved point-spread resolution with linear complexity
  ([Ultrasonics, 2025](https://www.sciencedirect.com/science/article/pii/S0041624X25000903)).
- Receive-capable histotripsy feedback: acoustic-feedback work identifies the
  limitation of transmit-only histotripsy systems and motivates arrays that use
  therapy elements as receivers for cavitation and damage monitoring
  ([University of Michigan dissertation, 2025](https://deepblue.lib.umich.edu/items/00da5ec9-07b2-4410-b9ee-7155f81c7484)).
- Platform constraints: Edison is described publicly as pulsed therapy with
  continuous bubble-cloud visualization
  ([HistoSonics](https://histosonics.com/our-technology-2/)); the FDA 510(k)
  summary records the integrated diagnostic ultrasound probe and 52/56-element
  treatment-head class for Edison workflows
  ([FDA K233466](https://www.accessdata.fda.gov/cdrh_docs/pdf23/K233466.pdf));
  Exablate Neuro specifies a focused-bowl 1024-element phased array,
  `620-720 kHz` operation, pre-treatment CT/MRI fusion, and MR monitoring
  ([INSIGHTEC datasheet](https://insightec.com/files/PUB41006477-Neuro-System-Data-Sheet-Rev-2.pdf));
  Vantage NXT exposes programmable transmit/receive, raw ultrasound data,
  arbitrary waveform generation, and high-power FUS support for research
  ([Verasonics 2026 brochure](https://verasonics.com/wp-content/uploads/2026/03/Vantage-NXT-Brochure-and-Specs-March-2026.pdf)).

The resulting kwavers contract maps the tomotherapy analogy onto ultrasound:
therapy elements transmit treatment packets, then the same aperture plus any
coaxial imaging receivers collects active pitch-catch, passive subharmonic,
second-harmonic, and ultraharmonic data for finite-frequency inverse and RTM
updates plus iterative ElasticPSTD FWI for the shear channel. The same-device
branch now reports `is_full_wave_inversion = true` and
`uses_nonlinear_wave_propagation = false`. The nonlinear branch owns the 3-D
source-encoded Westervelt forward/adjoint, multiparameter `c/beta` updates, and
Rayleigh-Plesset cavitation inverse, but keeps them separated from the linear
RTM and reduced harmonic channels so the figure metadata does not mix physics
contracts. The next complete increment is thermoviscous/shock-
capturing stabilization for higher histotripsy pressures plus a joint
`c/alpha/rho/beta/bubble-density` inverse with a robust trace-space misfit;
Python remains limited to plotting and animation.
