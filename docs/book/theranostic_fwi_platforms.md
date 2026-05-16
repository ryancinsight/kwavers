# Same-Device Therapeutic Ultrasound, Finite-Frequency Inverse, and RTM Monitoring

This chapter studies the tomotherapy-like ultrasound contract: the treatment
array is also the transmit/receive aperture for image formation and treatment
monitoring. The current implementation covers three CT-derived finite-frequency
inverse scenarios:

- INSIGHTEC-like 1024-element transcranial helmet around the head CT.
- HistoSonics-like 256-element concave abdominal array at the skin surface for
  KiTS19 kidney tumor CT.
- HistoSonics-like 256-element concave abdominal array at the skin surface for
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
the tomotherapy analogy. Transcranial ultrasound FWI helmets have been
demonstrated experimentally for imaging, Exablate Neuro uses a CT/MR-planned
1024-element therapeutic helmet, Edison histotripsy uses integrated diagnostic
ultrasound for liver targeting and bubble-cloud visualization, and research
histotripsy systems use passive acoustic feedback. The implemented workflow is
therefore a research simulation of a plausible same-aperture therapy/monitoring
platform, not a claim that HistoSonics or INSIGHTEC currently ship therapeutic
FWI reconstruction.

## Mathematical Contract

### Definition: Same-Device Aperture Contract

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

### Theorem: Same-Aperture Operator Rank

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

The current kwavers implementation is a reduced finite-frequency Born inverse
plus a separate source-encoded linear acoustic time-domain RTM image. The RTM
image is computed from pressure-amplitude source injection, CT-derived
baseline and lesion-perturbed receiver traces, and adjoint residual
backpropagation over the full domain travel-time horizon. The default adjoint
source is a Charbonnier-robust residual scaled by the configured receiver-noise
fraction and observed-trace RMS, but it is not an
iterative multiparameter FWI update and not nonlinear Westervelt/
Rayleigh-Plesset propagation. The production contract is explicit:
`kwavers::clinical::therapy::theranostic_guidance`
owns patient CT workflow, anatomy selection, device-placement analogs, exposure
synthesis, and reconstruction reporting. The reduced finite-frequency
same-aperture row operators, passive subharmonic rows, harmonic rows, and
graph-Laplacian PCG normal-equation solver live under
`kwavers::solver::inverse::same_aperture`. General seismic FWI and RTM kernels
remain under `kwavers::solver::inverse::seismic`. The active same-aperture
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

### Theorem: Deterministic Encoded Normal Equations

Let `A in R^(m x n)` be one same-aperture active, passive, harmonic, or
ultraharmonic sensitivity operator. Let `C in R^(k x m)` contain disjoint
contiguous row blocks whose nonzero entries are deterministic signs normalized
by the square root of the actual block size. The encoded operator is

```text
B = C A.
```

For any model `x`, `B x = C(A x)`. For any encoded residual `y`,
`B^T y = A^T C^T y`. Therefore the Chapter 29 reduced inverse solves the exact
encoded quadratic

```text
0.5 ||C(A m - d)||_2^2 + 0.5 lambda ||m||_2^2 + 0.5 gamma m^T L m
```

for the configured encoding. This reduces the number of PCG residual rows but
does not change the physics contract into nonlinear full-waveform inversion.

### Theorem: Linear Acoustic RTM Imaging Condition

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

### Theorem: Bounded Robust RTM Residual

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
The Chapter 29 RTM channel uses this derivative for receiver injection by
default. Setting `waveform_misfit = "l2"` recovers the unbounded least-squares
adjoint source `psi(r) = r`.

### Theorem: Discrete Westervelt FWI Adjoint

The nonlinear 3-D branch advances pressure by the explicit heterogeneous
Westervelt recurrence

```text
p[n+1] = S(2p[n] - p[n-1] + c^2 dt^2 Lp[n]
          + beta dt^2/(rho c^2) dtt(p^2)[n] + s[n]),
```

where `S` is the fixed polynomial absorbing layer, `L` is the 7-point 3-D
Laplacian, and `dtt(p^2)` is the product-rule second time derivative using
`p[n]`, `p[n-1]`, and `p[n-2]`.

**Sign convention.** The continuous Westervelt equation in canonical form
(Westervelt 1963 Eq. 24; Hamilton & Blackstock 1998 §3.5 Eq. 3.10) is
`∇²p − (1/c²)·∂²p/∂t² + (β/(ρ₀c⁴))·∂²(p²)/∂t² = 0`. Solving for `p_tt` and
discretizing by the second-order leapfrog gives the recurrence above with a
**positive** nonlinear coefficient on `p[n+1]`. The positive sign is required
for forward steepening — compressions travel faster than rarefactions and
peaks at fixed `x` arrive earlier than the linear prediction. A sign-flipped
nonlinear term produces non-physical reverse steepening; the sign-sensitive
regression
`forward_westervelt_exhibits_physical_forward_steepening_with_corrected_sign`
locks this convention by checking that the steady-state receiver trace
satisfies `max(∂p/∂t) > |min(∂p/∂t)|`.

Receiver traces are compared to traces from a lesion-perturbed target medium.
The reverse sweep applies the transpose of each recurrence Jacobian and
accumulates

```text
d p[n+1] / d c =
  2 c dt^2 Lp[n] - 2 beta dt^2 dtt(p^2)[n] / (rho c^3),

d p[n+1] / d beta =
  dt^2 dtt(p^2)[n] / (rho c^2).
```

The negative sign on the `c`-sensitivity nonlinear branch comes from
`∂q/∂c = −2·β·dt²/(ρ·c³)` for `q = β·dt²/(ρ·c²) > 0`; the positive sign on the
`beta`-sensitivity comes from `∂q/∂β = dt²/(ρ·c²) > 0`. Because the
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

### Algorithm: Rayleigh-Plesset Cavitation Inverse

For each active voxel, the nonlinear branch integrates the incompressible
Rayleigh-Plesset ODE with the local Westervelt peak pressure as the acoustic
forcing amplitude. The voxel source density is the maximum period-doubled
radius response, so the passive source depends on bubble dynamics rather than
on a hand-labeled lesion mask. A subharmonic Green operator maps that source to
the same receiver aperture, and the inverse solves a nonnegative Tikhonov
problem by projected gradient descent with a step bounded by the operator
Frobenius norm.

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

For the brain helmet at a 650 kHz drive (325 kHz subharmonic), the skull
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
bubbles, and intermediate for subharmonic-emitting bubbles.  This distinguishes
the passive subharmonic emission source (Dirichlet cavitation indicator) from
stable liner oscillation without requiring a threshold pressure or a
classification network.

The kwavers implementation stores one drive-period ring buffer of radius samples
(O(f₀/f_sample) samples), algebraically equivalent to retaining the full radius
history for this scalar because the recurrence reads only the sample one period
behind the current RK4 state. The passive simulated data is generated from the
active-voxel source vector used by the Green-operator columns, so the receiver
data and inverse model share the same active-support indexing.

### Theorem: Positive Normal Operator

Let `A` be the finite-frequency same-aperture sensitivity matrix, `lambda > 0`,
`gamma >= 0`, and `L` the four-neighbor graph Laplacian on the CT-derived active
tissue support. The operator

```text
H = A^T A + lambda I + gamma L
```

is symmetric positive definite on the active support. Therefore the Chapter 29
PCG solve minimizes the unique quadratic objective:

```text
0.5 ||A m - d||_2^2 + 0.5 lambda ||m||_2^2 + 0.5 gamma m^T L m
```

Proof: `A^T A` is positive semidefinite by construction. The graph Laplacian
satisfies `m^T L m = sum_(i,j in E) (m_i - m_j)^2 >= 0`. Adding
`lambda I` makes the quadratic form strictly positive for every nonzero `m`.

### Algorithm: Same-Aperture Monitoring Loop

1. Load CT/NIfTI with RITK and convert intensities into anatomy-specific
   acoustic property maps.
2. Build the treatment aperture in the patient coordinate frame: calvarial
   helmet for brain or external skin-normal arc for abdomen.
3. Synthesize pressure-calibrated exposure from the therapy elements.
4. Instantiate active pitch-catch, passive subharmonic, second-harmonic, and
   ultraharmonic matrix-free operators from the same aperture and receiver set.
5. Encode the reduced inverse rows by deterministic normalized signs and record
   both encoded and unencoded row counts in the PyO3 metrics.
6. Simulate source-encoded baseline and lesion-perturbed acoustic wavefields
   with pressure-amplitude source injection, record receiver traces on the same
   aperture through the CT-domain travel-time horizon, and backpropagate the
   robust residual traces for a linear RTM image.
7. Solve each reduced inverse channel with the same graph-Laplacian PCG core.
8. Fuse active lesion inverse output with passive and harmonic support maps for the
   monitoring image.

### Algorithm: Nonlinear 3-D Branch

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

## Device Placement

The brain case reads the CT-aligned target and transducer pose from
`CANONICAL_BRAIN_SCENE` in
`pykwavers/examples/book/transcranial_planning/scene.py`. The slice-level
finite-frequency inverse, Figure 5 nonlinear branch, and separate CT-derived
3-D helmet placement resolve the same target fraction against their CT support
and use the same 1024-element, 650 kHz, 0.150 m helmet definition. The 3-D view
renders the head surface, dense skull/calvarium surface points, the calvarium
helmet element cloud, sampled beam paths, and the first dense-bone intersection
on each sampled beam. The helmet cap is limited to the superior skull support
determined from the CT axial area profile, so the visualized elements cover the
calvarium instead of extending down the neck. The
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

The simulated pressure scale is explicit. The brain case uses a diagnostic
receive/imaging pressure of `1.5e5 Pa`. The kidney and liver histotripsy cases
use `28.0e6 Pa`, above the `26 MPa` liver-envelope threshold reported in the
2025 Edison-like liver aberration-correction study. The exposure field is
therefore a pressure-calibrated synthetic field, not a unitless display map.

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

## Minimal Usage Example

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
print("is_full_wave_inversion:", result["is_full_wave_inversion"])  # False

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

## Figures

Run:

```powershell
python pykwavers/examples/book/ch29_theranostic_fwi_platforms.py
```

Outputs:

- `docs/book/figures/ch29/fig01_device_placement_on_ct.{png,pdf}`
- `docs/book/figures/ch29/fig02_exposure_and_reconstruction.{png,pdf}`
- `docs/book/figures/ch29/fig03_brain_helmet_3d_placement.{png,pdf}`
- `docs/book/figures/ch29/fig04_reconstruction_dynamic_range_diagnostics.{png,pdf}`
- `docs/book/figures/ch29/fig05_nonlinear_3d_westervelt_rp_cavitation.{png,pdf}`
- `docs/book/figures/ch29/fig06_controlled_linear_nonlinear_comparison.{png,pdf}`
- `docs/book/figures/ch29/metrics.json`
- `docs/book/figures/ch29/controlled_comparison_metrics.json`
- `docs/book/figures/ch29/controlled_comparison_fields.npz`

The metrics file records reconstruction quality, placement geometry, the
canonical brain scene manifest,
outside-target sidelobe diagnostics, matrix-free operator storage evidence,
RTM waveform misfit metadata, nonlinear 3-D Westervelt/Rayleigh-Plesset
metrics, nonlinear source-encoding/regularization controls, and model-fidelity
flags, including the nonlinear forward-checkpoint interval. The nonlinear FWI
loop reuses one residual trace workspace across source encodings and one
candidate `c`/`beta` workspace across line-search scales; this changes allocator
pressure only, not the objective or update equations. The reduced inverse cases report
`is_full_wave_inversion = false`; the separate nonlinear 3-D cases report
`is_full_wave_inversion = true`, `uses_nonlinear_wave_propagation = true`, and
`uses_rayleigh_plesset = true`. Figure 5 uses the same per-case grid contract
as Figure 2 by default: `48^3` for brain and `52^3` for kidney/liver nonlinear
Westervelt/Rayleigh-Plesset volumes. Figure 5 also reuses the Figure 2 brain
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
3-D crop is built, so kidney sources approach from the superior displayed arc
and liver sources approach from the inferior-left displayed arc. Reduced inverse
metrics also report the encoded and unencoded measurement counts used by
deterministic row encoding.

Figure 6 adds the controlled Figure 2/Figure 5 comparison path. It runs a
matched linear case with the nonlinear grid size, element count, drive
frequency, and source pressure, then evaluates the linear fields and nonlinear
3-D slab fields on the same nonlinear crop projection. The target overlay and
therapy aperture overlay are fixed to that common frame; the metrics record
the remaining projected 2-D-vs-3-D aperture residual from the source wrappers.
The 2026-05-16 post-alignment run gives mean common-grid linear-fusion Dice
`0.331`, nonlinear-fusion Dice `0.665`, and nonlinear peak-pressure
outside-target energy fraction `0.964`. Per-case nonlinear outside-target
peak-pressure energy fractions are `0.951` for brain, `0.965` for kidney, and
`0.976` for liver. The residual projected aperture distances from the original
Figure 2 linear layout to the Figure 5 3-D aperture are `45.91 mm`, `105.36 mm`,
and `35.57 mm`. The nonlinear FWI objectives decrease for brain
`0.7072 -> 0.0541`, kidney `0.8344 -> 0.3427`, and liver
`8.1186e-6 -> 5.2076e-6`, so the nonlinear inverse is not stalled. The common-
grid comparison rejects display resolution as the primary failure mode. After
correcting the source weighting, using histotripsy-scale brain drive, and
expanding the brain aperture to the requested element count, nonlinear fusion
is stronger than the reduced linear fusion on this common frame. The remaining
defect is cavitation specificity: the passive cavitation reconstruction still
has zero equal-area Dice in all three cases because the 3-D Westervelt pressure
and Rayleigh-Plesset source remain dominated by off-target energy.

### Acoustic Scalar Model

The Figure 5 nonlinear branch is a scalar acoustic pressure model. It uses
CT-derived sound speed, density, nonlinearity coefficient, attenuation, a
heterogeneous Westervelt pressure update, and a Rayleigh-Plesset cavitation
post-process. It does not use elastic displacement, shear speed, shear modulus,
Lame parameters, or mode-converted skull shear propagation. Elastic skull
effects remain outside this chapter's nonlinear implementation contract.

The nonlinear 3-D material map treats cells outside the body mask as coupling
fluid rather than CT air. This matches the Figure 2 placement abstraction,
where therapy elements couple through water/gel before entering tissue. Air
pockets inside the body retain the gas sound-speed and high-attenuation
contract, but exterior CT background does not slow the simulation to a
343 m/s gas domain or distort the source focusing delays.

### FDTD/PSTD Choice

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

## Research Alignment

The implemented channels follow the current research direction as of
2026-05-13:

- Brain FWI: Guasch et al. demonstrate the seismic analogy for adult brain
  imaging with a 1024-transducer helmet where each element acts as source and
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
  Exablate Neuro specifies a helmet-shaped 1024-element phased array,
  `620-720 kHz` operation, pre-treatment CT/MRI fusion, and MR monitoring
  ([INSIGHTEC datasheet](https://insightec.com/files/PUB41006477-Neuro-System-Data-Sheet-Rev-2.pdf));
  Vantage NXT exposes programmable transmit/receive, raw ultrasound data,
  arbitrary waveform generation, and high-power FUS support for research
  ([Verasonics 2026 brochure](https://verasonics.com/wp-content/uploads/2026/03/Vantage-NXT-Brochure-and-Specs-March-2026.pdf)).

The resulting kwavers contract maps the tomotherapy analogy onto ultrasound:
therapy elements transmit treatment packets, then the same aperture plus any
coaxial imaging receivers collects active pitch-catch, passive subharmonic,
second-harmonic, and ultraharmonic data for finite-frequency inverse and RTM
updates. The reduced branch keeps `is_full_wave_inversion = false` and
`uses_nonlinear_wave_propagation = false`. The new nonlinear branch owns the
3-D source-encoded Westervelt forward/adjoint, multiparameter `c/beta`
updates, and Rayleigh-Plesset cavitation inverse, but keeps them separated from
the linear RTM and reduced harmonic channels so the figure metadata does not
mix physics contracts. The next complete increment is thermoviscous/shock-
capturing stabilization for higher histotripsy pressures plus a joint
`c/alpha/rho/beta/bubble-density` inverse with a robust trace-space misfit;
Python remains limited to plotting and animation.
