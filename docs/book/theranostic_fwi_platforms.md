# Same-Device Therapeutic Ultrasound and FWI/RTM Monitoring

This chapter studies the tomotherapy-like ultrasound contract: the treatment
array is also the transmit/receive aperture for image formation and treatment
monitoring. The current implementation covers three CT-derived scenarios:

- INSIGHTEC-like 1024-element transcranial helmet around the head CT.
- HistoSonics-like 256-element concave abdominal array at the skin surface for
  KiTS19 kidney tumor CT.
- HistoSonics-like 256-element concave abdominal array at the skin surface for
  LiTS liver tumor CT.

The figures are fully synthetic, model-consistent simulations. They are not
measured HistoSonics, Verasonics, or INSIGHTEC device data, and they do not
claim proprietary element geometry. RITK owns NIfTI image ingestion; kwavers
owns CT preprocessing, device placement, exposure synthesis, finite-frequency
active FWI, passive subharmonic inversion, weak nonlinear harmonic inversion,
and reconstruction fusion through the PyO3 function:

```text
pykwavers.run_theranostic_fwi_from_ritk(...)
```

## Mathematical Contract

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
receiver-only subharmonic sensitivity at `f0/2`. The nonlinear channel uses the
same aperture with second-harmonic rows at `2f0` and ultraharmonic rows at
`1.5f0`. Fusion gates active lesion FWI by passive subharmonic support plus
harmonic and ultraharmonic contrast; the generated metrics report the fused map
separately from the individual channels.

The current kwavers implementation is a reduced finite-frequency inverse model,
not a full stored-wavefield adjoint Westervelt/Rayleigh-Plesset inversion. The
production contract is explicit: `kwavers::solver::inverse::seismic::theranostic`
owns CT-derived support masks, source/receiver geometry, pressure-calibrated
exposure synthesis, row construction, and a preconditioned CG solve of the
regularized normal equations. The active-support graph Laplacian is precomputed
once, and each CG step reuses row, normal-operator, and Laplacian workspaces
instead of allocating a full image mask inside every iteration.

## Device Placement

The brain case places all 1024 elements on a circular projection for the
slice-level FWI operator and also emits a separate CT-derived 3-D helmet
placement view. The 3-D view renders the head surface, dense skull/calvarium
surface points, the calvarium helmet element cloud, sampled beam paths, and the
first dense-bone intersection on each sampled beam. The helmet cap is limited to
the superior skull support determined from the CT axial area profile, so the
visualized elements cover the calvarium instead of extending down the neck. The
abdominal cases place a concave 256-element therapy arc outside the nearest
external skin point to the target centroid, using a local skin-normal aperture
frame instead of a fixed left/right display axis. Internal gas pockets are
excluded from the skin candidate set by flood-filling exterior air from the CT
border. A central 64-receiver imaging line occupies the therapy-head cutout.
The PyO3 result exports `placement_metrics` and a separate full-CT
`placement_context`; figure 1 uses the uncropped patient slice for kidney and
liver so the skin interface is visible relative to the stomach/hip
cross-section rather than only the local tumor field of view.

The Verasonics-like role in this simulation is the programmable acquisition
contract rather than a fixed clinical transducer geometry: each case exposes
source count, receiver offsets, frequency list, pressure scale, and raw
same-aperture active/passive channel synthesis through
`run_theranostic_fwi_from_ritk`.

The simulated pressure scale is explicit. The brain case uses a diagnostic
receive/imaging pressure of `1.5e5 Pa`. The kidney and liver histotripsy cases
use `28.0e6 Pa`, above the `26 MPa` liver-envelope threshold reported in the
2025 Edison-like liver aberration-correction study. The exposure field is
therefore a pressure-calibrated synthetic field, not a unitless display map.

## Figures

Run:

```powershell
python pykwavers/examples/book/ch29_theranostic_fwi_platforms.py
```

Outputs:

- `docs/book/figures/ch29/fig01_device_placement_on_ct.{png,pdf}`
- `docs/book/figures/ch29/fig02_exposure_and_reconstruction.{png,pdf}`
- `docs/book/figures/ch29/fig03_brain_helmet_3d_placement.{png,pdf}`
- `docs/book/figures/ch29/metrics.json`

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
- Recent FWI cycle-skipping controls: low-frequency extrapolation by sparse
  deconvolution targets the missing-low-frequency problem in practical USCT
  ([Ultrasound Med. Biol., 2025](https://www.sciencedirect.com/science/article/pii/S0301562925001097)),
  and the HV metric is a signed-signal transport alternative to `L2` and
  Wasserstein objectives for time-domain FWI
  ([arXiv 2508.17122](https://arxiv.org/abs/2508.17122)). These motivate the
  next kwavers step of replacing reduced rows with a full adjoint
  Westervelt/Rayleigh-Plesset path and an explicit misfit strategy trait rather
  than expanding Python-side plotting.
- RTM/FWI method split: ultrasonic full-matrix-capture studies compare TFM,
  RTM, and FWI, with RTM serving as a one-pass localization image and FWI as
  the iterative material-property update
  ([Ultrasonics, 2025](https://portal.fis.tum.de/en/publications/quantitative-comparison-of-the-total-focusing-method-reverse-time)).
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
  ([HistoSonics](https://histosonics.com/our-technology-2/)); Exablate Neuro
  specifies a helmet-shaped 1024-element phased array, `620-720 kHz` operation,
  pre-treatment CT/MRI fusion, and MR monitoring
  ([INSIGHTEC datasheet](https://www.insightec.com/wp-content/uploads/2021/08/Exablate-Neuro-Platform-Datasheet-PUB41004616-NA-Rev1.pdf));
  Vantage NXT exposes programmable transmit/receive, raw ultrasound data,
  arbitrary waveform generation, and high-power FUS support for research
  ([Verasonics brochure](https://verasonics.com/wp-content/uploads/2024/01/Vantage-NXT-Brochure-and-Specifications-Jan-2024.pdf)).

The resulting kwavers contract maps the tomotherapy analogy onto ultrasound:
therapy elements transmit treatment packets, then the same aperture plus any
coaxial imaging receivers collects active pitch-catch, passive subharmonic,
second-harmonic, and ultraharmonic data for RTM/FWI updates.

The next complete increment is a 3-D adjoint Westervelt/Rayleigh-Plesset
variant that estimates `c`, `alpha`, and a cavitation source density jointly
instead of using reduced harmonic rows. The design constraint for that increment
is the same one used here: the forward/adjoint solver must be kwavers-owned and
RITK-backed, while Python remains limited to plotting and animation.
