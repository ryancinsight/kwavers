# Chapter 23 — Passive Acoustic Mapping

> **Prerequisite:** Chapter 9 (Cavitation and Bubble Dynamics), Chapter 16
> (Transcranial Ultrasound), Chapter 22 (Simulation Orchestration).
> Familiarity with array signal processing and the van Cittert–Zernike theorem
> is assumed.

---

## 23.1 Scope

Passive Acoustic Mapping (PAM) reconstructs the spatial distribution of
acoustic emissions from cavitation inside tissue using a passive multi-element
receive aperture — no active transmit pulse is fired during recording.  In
transcranial focused-ultrasound (tFUS) therapy of the brain it is the primary
tool for distinguishing *stable* (sustained, harmonic) cavitation — a marker
of controlled blood-brain barrier (BBB) opening — from *inertial* (broadband,
collapse) cavitation — a marker of haemorrhagic risk.

This chapter formalises the signal model, derives the van Cittert–Zernike
coherence limit on spatial resolution, describes the two main beamformers
(delay-and-sum and eigenspace), and analyses the transcranial SNR budget.  A
simulation workflow using the kwavers `PhysicsCatalog` is provided as the
worked example (§23.6).

---

## 23.2 Cavitation emission model

A cavitation cloud at position **r**₀ emits pressure waves that travel to a
receive aperture of $N$ elements at positions **r**_i.  The received
signal at element $i$ and frequency $f$ is:

$$
p_i(f) = G(f,\,|\mathbf{r}_0 - \mathbf{r}_i|)\; e^{j 2\pi f |\mathbf{r}_0 - \mathbf{r}_i| / c}
\; s(f) \; + \; n_i(f)
$$

where $G$ is the propagation Green's function (includes skull attenuation),
$s(f)$ is the source spectrum, and $n_i$ is additive noise.

> **Definition 23.1 (Cavitation emission spectrum).**
> *Stable cavitation (SC) emits at harmonics of the driving frequency $f_0$
> and its sub-harmonic $f_0/2$.  Inertial cavitation (IC) adds a broadband
> floor elevated by 15–25 dB above the SC inter-harmonic noise level.*

Discrimination relies on the ratio of sub-harmonic/ultra-harmonic energy to
wideband emission energy — the *cavitation dose* metric of Gyöngy & Coussios
(2010).

---

## 23.3 Delay-and-sum passive beamformer

The passive DAS output at candidate focus position **r**_f is:

$$
b(\mathbf{r}_f) = \left|\sum_{i=1}^{N} w_i\; p_i(t - \tau_i)\right|^2
$$

where $\tau_i = |\mathbf{r}_f - \mathbf{r}_i| / c$ is the focusing delay and
$w_i$ is an apodisation weight.

> **Theorem 23.1 (DAS resolution).**
> *For an incoherent source at position **r**₀, the DAS mainlobe half-width
> (−6 dB) in the lateral dimension is approximately*
> $$\delta_\perp \approx \lambda \, z / D$$
> *where $\lambda = c/f_0$, $z$ is the axial depth of the source, and $D$
> is the aperture width.*

**Proof sketch.**  The DAS output is proportional to the array factor
$|A(\mathbf{r}_f)|$ = $|\sum_i \exp(j\phi_i)|$ where $\phi_i$ is the
phase error for a candidate focus at offset $\delta x$ from the true source.
For a linear aperture with uniform spacing $d$, the standard Fraunhofer
diffraction analysis (Goodman 2005, §4.2) yields a sinc mainlobe of width
$\lambda z / D$.  $\square$

The DAS beamformer is signal-subspace blind: it uses no prior knowledge of the
noise rank and suffers from grating-lobe artefacts when element spacing exceeds
$\lambda/2$.

### kwavers implementation boundary

The authoritative production implementation is
`kwavers::analysis::signal_processing::pam::DelayAndSumPAM`.  The PyO3 function
`pykwavers.passive_acoustic_map_das(passive_data, sensor_positions, grid_points,
sound_speed, sampling_frequency, ...)` borrows NumPy arrays as read-only views
and delegates shape, finite-value, and sensor-count validation to that Rust
boundary.  DAS receive delays use fractional linear interpolation, matching the
KWave.jl delay-law comparison and avoiding blocky integer-delay cavitation maps.
This keeps the signal-processing contract in one location while the Python layer
provides only ABI conversion.

The comparison harness below validates the same impulsive-source theorem used
by the Rust unit test and writes a side-by-side plot:

```bash
python pykwavers/examples/passive_acoustic_mapping_compare.py --plot
```

The harness compares pykwavers against an independent KWave.jl delay-law
reference on the same grid, reports peak localization error in metres, and
exports `pykwavers/examples/output/passive_acoustic_mapping_compare.png`.
With `--plot`, it also reconstructs a 3-D cavitation source volume with a
planar receive aperture and exports
`pykwavers/examples/output/passive_acoustic_mapping_volume_compare.png`, a
side-by-side set of `xy`, `xz`, and `yz` maximum-intensity projections for
pykwavers, the independent delay-law reference, and their absolute difference.

---

## 23.4 Eigenspace beamformer

Let $\mathbf{R} \in \mathbb{C}^{N\times N}$ be the cross-spectral density
matrix estimated from $M$ receive snapshots.  The eigenvalue decomposition:

$$
\mathbf{R} = \mathbf{U}_S \mathbf{\Sigma}_S \mathbf{U}_S^H
           + \mathbf{U}_N \mathbf{\Sigma}_N \mathbf{U}_N^H
$$

partitions the signal subspace $\mathbf{U}_S$ (rank $K$ = number of sources)
from the noise subspace $\mathbf{U}_N$.

> **Theorem 23.2 (Eigenspace PAM noise rejection).**
> *For $K$ incoherent point sources with signal power $\sigma_s^2$ and
> spatially-white noise power $\sigma_n^2$, the $K$ largest singular values
> of $\mathbf{R}$ satisfy $\sigma_k = \sigma_s^2 + \sigma_n^2$ (signal + noise),
> while the remaining $N-K$ singular values equal $\sigma_n^2$ (noise only).
> The signal-subspace projector $\mathbf{P}_S = \mathbf{U}_S \mathbf{U}_S^H$
> suppresses noise-only contributions by a factor of $N-K$.*

**Proof.**  Standard rank-$K$ perturbation of the identity-scaled noise
covariance $\sigma_n^2 \mathbf{I}$ by the rank-$K$ signal covariance
$\mathbf{A}\mathbf{A}^H$ (Arnal et al. 2017, Lemma 1).  $\square$

The eigenspace PAM output replaces $\mathbf{I}$ in the standard DAS
by $\mathbf{P}_S$:

$$
b_{ES}(\mathbf{r}_f) = |\mathbf{a}^H(\mathbf{r}_f)\; \mathbf{P}_S\; \mathbf{a}(\mathbf{r}_f)|^2
$$

where $\mathbf{a}(\mathbf{r}_f) = [e^{j\phi_1}, \ldots, e^{j\phi_N}]^T$ is
the steering vector to the candidate focus.

---

## 23.5 Transcranial SNR budget

Two loss mechanisms reduce transcranial PAM SNR:

**1. Skull attenuation** (Fry & Barger 1978):

$$
\text{IL}(f) = \alpha_0 \, f^{1.2} \, d \quad [\text{dB}]
$$

with $\alpha_0 \approx 6\;\text{dB/(cm·MHz}^{1.2})$ for human calvaria of
thickness $d$.  PAM receives a two-way skull loss $2 \times \text{IL}(f)$.

**2. Phase aberration coherence loss** (Maréchal 1947):

$$
\text{CF}_\phi = e^{-\sigma_\phi^2}
$$

where $\sigma_\phi$ is the RMS wavefront phase error across the aperture due
to skull heterogeneity.  For typical adult skulls $\sigma_\phi \sim 1\;\text{rad}$
at 1 MHz, giving $\text{CF}_\phi \approx -4.3\;\text{dB}$.

> **Corollary 23.1 (Optimal PAM frequency).**
> *The PAM SNR is maximised at the frequency where the total loss
> $2\,\text{IL}(f) + |\text{CF}_\phi(f)|$ is minimised relative to the
> cavitation emission energy spectrum $S_{cav}(f)$.
> For typical clinical parameters this optimum lies in the range 0.5–1.0 MHz.*

---

## 23.6 Simulation workflow — cerebral cavitation PAM

The forward simulation propagates broadband cavitation emissions through a
skull-heterogeneous medium and captures them at a hemispherical receive array.

```rust
use kwavers::plugin::*;
use kwavers::{Grid, AcousticSolver, BoundaryType, PhysicsModelType,
              PhysicsModelConfig, PhysicsConfig, PhysicsCatalog};

// 1. Grid: 128 mm × 128 mm × 128 mm at 0.5 mm resolution
let grid = Grid::new(256, 256, 256, 0.5e-3, 0.5e-3, 0.5e-3)?;

// 2. Config: PSTD acoustics through skull-CT medium
let mut config = PhysicsConfig::new();
config.models.push(PhysicsModelConfig {
    model_type: PhysicsModelType::LinearAcoustics {
        solver_type: AcousticSolver::PSTD { spectral_accuracy: true },
        boundary_conditions: BoundaryType::Absorbing { pml_layers: 12 },
    },
    enabled: true,
    parameters: Default::default(),
});

// 3. Medium: CT-derived heterogeneous skull
let skull_medium = HeterogeneousMedium::from_ct_scan(&ct_volume, &hu_to_acoustic_map)?;

// 4. Build and run
let manager = PhysicsCatalog::build(&config, &grid, &skull_medium, dt)?;
```

Post-processing applies eigenspace PAM beamforming on the hemispherical
receive aperture — see `analysis::beamforming::passive_acoustic_mapping`.

---

## 23.7 Figure sources

Run the companion script to generate all figures:

```bash
python pykwavers/examples/book/ch23_passive_acoustic_mapping.py
```

Outputs are written to `docs/book/figures/ch23/` (PDF and PNG).

| Figure | Content |
|--------|---------|
| fig01  | Stable vs inertial cavitation emission spectra |
| fig02  | Passive DAS sensitivity map vs focal depth |
| fig03  | Van Cittert–Zernike spatial coherence |
| fig04  | Eigenspace SVD: signal vs noise singular values |
| fig05  | Transcranial SNR budget (attenuation + phase aberration) |
| fig06  | Cavitation dose accumulation — SC vs IC |

---

## 23.8 References

- Coviello C., Kozick R., Choi J., Gyöngy M., Jensen C., Smith P.P.,
  Coussios C.C. *Passive Acoustic Mapping utilizing optimal beamforming
  in ultrasound therapy monitoring.* J. Acoust. Soc. Am. **137**(5),
  pp. 2573–2585, 2015. doi:10.1121/1.4916694
- Salgaonkar V.A., Datta S., Holland C.K., Mast T.D. *Passive cavitation
  imaging with ultrasound arrays.* J. Acoust. Soc. Am. **126**(6),
  pp. 3071–3083, 2009. doi:10.1121/1.3238260
- O'Reilly M.A., Hynynen K. *A super-resolution ultrasound method for
  brain vascular mapping.* Med. Phys. **40**(11), 110701, 2013.
  doi:10.1118/1.4823762
- Arnal B., Baranger J., Demene C., Tanter M., Pernot M. *In vivo real-time
  cavitation imaging in moving organs.* IEEE Trans. Med. Imaging **36**(7),
  pp. 1543–1553, 2017. doi:10.1109/TMI.2017.2700909
- Gyöngy M., Coussios C.C. *Passive cavitation mapping for localization and
  tracking of bubble dynamics.* J. Acoust. Soc. Am. **128**(4),
  pp. EL175–EL180, 2010. doi:10.1121/1.3467491
- Goodman J.W. *Introduction to Fourier Optics*, 3rd ed., Roberts & Company,
  2005. §4.2 (Fraunhofer diffraction and array factor).
