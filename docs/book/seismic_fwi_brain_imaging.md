# Chapter 27 - Seismic Full-Waveform Brain Imaging

> **Prerequisite:** Chapter 16 (Transcranial Ultrasound), Chapter 17
> (Inverse Problems and PINNs), Chapter 25 (Transcranial HIFU and BBB
> Treatment Planning), and Chapter 26 (Neuromodulation).

---

## 27.1 Scope

This chapter adds a reproducible, CT-derived seismic-imaging workflow for
transcranial brain sound-speed reconstruction.  It follows the published
brain-FWI study reported by UCL and npj Digital Medicine:

- UCL news summary:
  <https://www.ucl.ac.uk/news/2020/mar/seismic-imaging-technology-could-deliver-detailed-images-brain>
- Guasch et al. 2020, npj Digital Medicine:
  <https://www.nature.com/articles/s41746-020-0240-8>
- 2024 transcranial-ultrasound FWI attenuation study:
  <https://pubmed.ncbi.nlm.nih.gov/40039691/>
- 2025 polar-coordinate structural-prior INR-FWI study:
  <https://papers.miccai.org/miccai-2025/0662-Paper2163.html>
- Protopapa and Cueto 2021, frequency-adaptive brain FWI:
  <https://arxiv.org/abs/2111.04700>
- Recent ultrasound-CT FWI work using source/frequency encoding and
  regularized multiscale inversion:
  <https://pubmed.ncbi.nlm.nih.gov/40197542/>
  <https://cpb.iphy.ac.cn/EN/article/downloadArticleFile.do?attachType=PDF&id=127579>
- 2024-2025 histotripsy monitoring work showing lesion visibility from
  Nakagami/B-mode features and cavitation-emission features:
  <https://www.sciencedirect.com/science/article/pii/S1350417724002505>
  <https://pubmed.ncbi.nlm.nih.gov/40015999/>
- 2025 comparison of TFM, RTM, and FWI for ultrasonic defect imaging:
  <https://arxiv.org/abs/2412.07347>
- 2025 pressure-modulated shockwave histotripsy lesion-control study:
  <https://www.nature.com/articles/s41598-025-11512-x>
- 2025 passive-cavitation mapping with higher-order correlation beamforming:
  <https://www.sciencedirect.com/science/article/pii/S0041624X25000903>
- 2026 sparse-aperture 3-D passive acoustic mapping:
  <https://www.nature.com/articles/s41598-026-42764-w>

The published study used an ultrasound helmet acquisition with 1024 unfocused
source/receiver positions and adapted seismic full-waveform inversion to recover
brain acoustic properties through the skull.  The in-repository chapter
implements a bounded reproduction of that acquisition contract with the same
1024-element count used elsewhere in kwavers for INSIGHTEC-style transcranial
arrays.

The executable chapter script is:

`pykwavers/examples/book/ch27_seismic_fwi_brain_imaging.py`

The production implementation is not in the plotting script.  The computation is
owned by:

- `kwavers::solver::inverse::seismic::brain_helmet`
- `pykwavers::run_seismic_helmet_fwi_volume_from_ritk_ct`
- `ritk-io` for CT NIfTI ingestion

No SciPy, nibabel, or pydicom dependency is required for this chapter path.

---

## 27.2 Formal contract

Inputs:

- Head CT volume readable by `ritk-io`.
- CT voxel spacing from the RITK image metadata.
- INSIGHTEC-style element count `N = 1024`.
- Encoded finite-frequency transmit/receive channels over the 1024-element
  hemispherical cap.
- Iteration schedule for the reconstruction optimizer.

Outputs:

- Resampled CT volume in HU.
- Skull mask, brain inversion mask, and CT-derived acoustic speed target volume.
- Initial model with frozen skull and homogeneous brain speed.
- Encoded synthetic data generated from the finite-frequency sensitivity model.
- Single-pass adjoint migration volume reconstructed from the same simulated data.
- Optimized reconstructed brain sound-speed volume.
- Structure-enhanced display image derived from the optimized FWI reconstruction.
- Nonlinear second-harmonic encoded channels from a weak-Westervelt model.
- Multi-slice stack of CT HU, CT-derived acoustic target, regularized FWI
  reconstruction, and error images sliced from the reconstructed 3-D array.
- Centroid-cropped reconstruction stack over the central brain region for
  deep midline inspection from pons-level through thalamus-level slices.
- Objective history and visibility metrics.

The inverse problem minimizes:

$$
J(m)=\frac{1}{2}\|A m-d\|_2^2+\frac{\lambda}{2}\|m\|_2^2
+\gamma\sum_{(i,j)\in E}\left(\sqrt{(m_i-m_j)^2+\epsilon^2}-\epsilon\right),
\qquad
m(x)=\frac{c(x)-c_0}{c_0}.
$$

Here `A` is a finite-frequency Born sensitivity matrix assembled from the
source, receiver, and frequency schedule; `d` is generated from the CT-derived
target contrast; `c0 = 1540 m/s`; `E` is the active-mask 6-neighbor voxel edge
set; and the skull model is CT-derived and frozen.  The last term is a
Charbonnier edge-preserving first-difference penalty.  It is a differentiable
TV-style prior and uses no CT target values during inversion.

Acceptance criteria:

$$
J_{\mathrm{final}} < J_{\mathrm{initial}},
\qquad
\frac{J_{\mathrm{initial}}-J_{\mathrm{final}}}{J_{\mathrm{initial}}} \ge 0.50,
$$

$$
\frac{\Delta c_{\mathrm{recon,p95-p5}}}
{\Delta c_{\mathrm{target,p95-p5}}} \ge 0.35.
$$

These criteria define a visible reconstruction for the chapter artifact.  They
do not define diagnostic accuracy or clinical adequacy.

Reject the run when CT loading bypasses RITK, when fewer than 1024 elements are
used for the chapter default, when the output image is independent of CT values,
or when the reconstruction fails the objective and contrast criteria.

---

## 27.3 Acquisition model

The chapter now solves one coupled 3-D inverse problem and slices the returned
volume for display.  The acquisition geometry is not a ring.  The 1024 elements
are placed on a deterministic equal-area hemispherical cap with radius
`0.11 m`. Receiver offsets are interpreted as azimuthal rotations on the cap
and mapped to the nearest physical element.  The default frequency set is:

$$
f \in \{200, 350, 500, 650, 800\}\ \mathrm{kHz}.
$$

The default receiver-offset set is:

$$
\Delta r \in \{512,384,640,256,768,128,448,576\}.
$$

For 3-D source `s`, receiver `r`, frequency `f`, harmonic order `h`, and active
volume voxel `x = (x,y,z)`, the row of the encoded matrix-free sensitivity
operator is:

$$
A_{srfhx}
=
\Delta V
\frac{
\exp(-h f_{\mathrm{MHz}}\alpha(x)(|x-s|+|x-r|))
H_h(x)\cos(h k_f(|x-s|+|x-r|))}
{\sqrt{|x-s||x-r|}},
\qquad
k_f=\frac{2\pi f}{c_0}.
$$

`H1 = 1`.  The optional nonlinear channel uses `h = 2` with weak-Westervelt
second-harmonic scaling:

$$
H_2(x)=\frac{1}{4}\frac{|x-s|+|x-r|}{x_s},
\qquad
x_s=\frac{\rho_0 c_0^3}{\beta\omega p_0}.
$$

Here `xs` is the pre-shock formation distance, `beta = 1 + B/(2A)`, and `p0`
is the configured source pressure.  This is a bounded second-order nonlinear
encoding model: it adds harmonic information without claiming to be a full
time-domain Westervelt solve.

The CT-derived attenuation path integral is:

$$
\Gamma(a,b)=\int_{a}^{b}\alpha_{\mathrm{CT}}(q)\,dq,
$$

where `alpha_CT` is in `Np/m/MHz`.  The bounded model uses
`0.5 dB/cm/MHz` converted to `Np/m/MHz` for soft tissue and blends toward
`70 Np/m/MHz` through the same skull bone-volume fraction used for sound speed.
The 3-D operator uses the active voxel's CT-derived attenuation as a local path
attenuation factor.  This is still model-consistent synthetic data, not a
measured attenuation calibration.

Each row is normalized before inversion.  This keeps the optimizer scale
defined by encoded channel geometry rather than by arbitrary source amplitude.

---

## 27.4 CT-to-acoustic model

The RITK-loaded CT tensor is transposed from `[z,y,x]` to `[x,y,z]` before
resampling.  The non-air head support is cropped and resampled onto an isotropic
cubic FWI grid.  Skull speed follows a bone-volume-fraction mapping:

$$
\phi(\mathrm{HU})=\mathrm{clamp}(\mathrm{HU}/1000,0,1),
$$

$$
c_{\mathrm{skull}}(\mathrm{HU})
=1500(1-\phi)+2900\phi\ \mathrm{m/s}.
$$

Soft-tissue speed is mapped from CT HU into the brain range used by the
transcranial examples.  The same CT grid also produces the attenuation map used
by the acquisition model.  The skull remains fixed during inversion; only voxels
in the CT-derived central brain mask are updated.

---

## 27.5 Optimization

The first reconstruction is a migration image:

$$
m_{\mathrm{mig}}
=
\mathrm{clip}\left(
(\mathrm{diag}(A^T A)+\lambda I)^{-1}A^T d,
m_{\min},m_{\max}
\right).
$$

It is the diagonal-normalized adjoint of the simulated ultrasound data.  This
image is not the final FWI result; it is the explicit simulated-data
reconstruction baseline used to verify that the encoded source/receiver data
contain spatial brain-speed contrast before iterative inversion.

The optimizer solves the regularized normal equations with a projected,
diagonal-preconditioned conjugate-gradient iteration:

$$
H=A^T A+\lambda I,\qquad b=A^T d,\qquad r_k=b-Hm_k,
$$

with preconditioner:

$$
z_k=(\mathrm{diag}(A^T A)+\lambda I)^{-1}r_k.
$$

Each step applies the matrix-free normal operator to the Krylov direction and
accepts the clipped update only when the composite stage objective is
non-increasing.  Row-normalization factors and per-row source/receiver/frequency
constants are computed once per acquisition row and reused across data
generation, migration, objective evaluation, and Krylov updates.  The optimizer
uses low-to-high frequency continuation and a Sobolev-smoothed preconditioned
residual before the full-band pass.  At each continuation-stage boundary, a
mask-local edge-preserving proximal projection is accepted only when the
full-band composite objective decreases.  This follows current ultrasound-FWI
practice: multiscale low-frequency information reduces cycle skipping,
regularized gradients stabilize updates, and edge-preserving priors suppress
checkerboard artifacts without imposing the CT target.  The
chapter script repeats the run over an iteration schedule and keeps the first
run that satisfies the visible-reconstruction contract.

The returned enhanced volume and the Figure 06 regularized FWI row are display
products, not second physical estimates.  The Figure 06 display applies
mask-aware diffusion plus clipped residual detail to the accepted FWI
sound-speed field so checkerboard artifacts do not dominate the sliced book
figure.  It uses only the FWI reconstruction and brain mask; metrics remain tied
to the physical FWI reconstruction.

---

## 27.6 Nonlinear and therapy-monitoring variants

| Variant | Data stream | Inverted quantity | Histotripsy role | Implemented here |
| --- | --- | --- | --- | --- |
| RTM / migration | Active or passive | Reflector or emission image | Fast localization and QC | Active migration baseline |
| Linear acoustic FWI | Active inter-burst | `c`, optionally `rho` | Post-packet lesion-property update | Brain speed FWI |
| Multiparameter attenuation FWI | Active inter-burst | `c`, `rho`, `alpha` | Amplitude/attenuation lesion contrast | Attenuation in forward rows |
| Tissue-harmonic FWI | Active inter-burst harmonic bands | `c`, `alpha`, `beta` | Nonlinear contrast at `2f0` | Weak second-harmonic row model |
| Passive cavitation source inversion | Passive intra-burst | `q_cav(x,t)` | Real-time bubble-cloud tracking | Subharmonic source FWI in custom simulator |
| Bubble-dynamics nonlinear FWI | Passive and active | Bubble state plus acoustic fields | Sub/ultraharmonic histotripsy feedback | Emission-band diagnostics; bubble state pending |
| Elastic/shear FWI | Post-burst mechanical wave data | `mu`, `lambda`, `rho` | Lesion stiffness confirmation | Pending |

Therapy monitoring must separate active inter-burst transmissions from passive
cavitation emissions during therapy bursts.  The current implementation covers
active migration, acoustic FWI, attenuation-weighted rows, weak
second-harmonic rows, passive multiband RTM, and subharmonic cavitation-source
FWI. The histotripsy simulator now adds deterministic receiver noise,
gain/phase jitter, low-to-high frequency continuation, Huber IRLS weighting,
multiparameter speed/attenuation inversion, and a weak nonlinear harmonic
`beta` branch. Bubble-state FWI should be added as a separate
therapy-monitoring orchestrator over existing Chapter 21, 23, 25, and 27
components.

The custom executable monitoring simulation is:

`pykwavers/examples/book/ch27_histotripsy_fwi_rtm.py`

It uses the RITK-backed Chapter 27 CT baseline, all 1024 receiver elements,
active continuation stages (`110 kHz`, `160 kHz`, `220 kHz`) ending at the
`220 kHz` therapy-monitoring carrier, passive cavitation diagnostics
(`110 kHz`, `220 kHz`, `440 kHz`), and three synthetic lesion states: compact
intrinsic-threshold, shock-elongated, and multi-packet. Reconstruction now
reports normal FWI, multiparameter speed/attenuation FWI, weak nonlinear
harmonic FWI, passive multiband RTM, subharmonic cavitation-source FWI from the
`110 kHz` row, and frequency-gated fusion. The fusion treats the subharmonic
image as cavitation support rather than as a boundary-resolution measurement.

---

## 27.7 Reproducible figures

Run:

```powershell
python pykwavers/examples/book/ch27_seismic_fwi_brain_imaging.py
```

The script writes:

- `docs/book/figures/ch27/fig01_ct_geometry.{pdf,png}`
- `docs/book/figures/ch27/fig02_acoustic_model.{pdf,png}`
- `docs/book/figures/ch27/fig03_brain_reconstruction.{pdf,png}`
- `docs/book/figures/ch27/fig04_optimization_and_data.{pdf,png}`
- `docs/book/figures/ch27/fig05_simulated_ultrasound_reconstruction.{pdf,png}`
- `docs/book/figures/ch27/fig06_multislice_reconstruction_stack.{pdf,png}`
- `docs/book/figures/ch27/fig07_centroid_pons_thalamus_roi.{pdf,png}`
- `docs/book/figures/ch27/metrics.json`
- `docs/book/figures/ch27/fig08_histotripsy_custom_reconstruction_scenarios.{pdf,png}`
- `docs/book/figures/ch27/fig09_histotripsy_reconstruction_metrics.{pdf,png}`
- `docs/book/figures/ch27/fig10_histotripsy_passive_band_rtm.{pdf,png}`
- `docs/book/figures/ch27/histotripsy_monitoring_metrics.json`

The default CT is:

`data/rire_patient_109/patient_109_ct.nii.gz`

The grid and iteration schedule can be changed without editing the script:

```powershell
$env:KWAVERS_CH27_GRID_SIZE="56"
$env:KWAVERS_CH27_FREQUENCIES_HZ="200000,350000,500000,650000,800000"
$env:KWAVERS_CH27_RECEIVER_OFFSETS="512,384,640,256,768,128,448,576"
$env:KWAVERS_CH27_ITERATIONS="12"
$env:KWAVERS_CH27_STACK_OFFSETS="-8,-6,-4,-2,0,2,4,6,8,10,12,14,16"
$env:KWAVERS_CH27_CENTROID_ROI_HALF_WIDTH_MM="35"
$env:KWAVERS_CH27_ATTENUATION_MODEL="1"
$env:KWAVERS_CH27_NONLINEAR_HARMONIC_MODEL="1"
$env:KWAVERS_CH27_SOURCE_PRESSURE_MPA="0.15"
$env:KWAVERS_CH27_NONLINEAR_BETA="4.5"
$env:KWAVERS_CH27_EDGE_PRESERVING_WEIGHT="0.0001"
$env:KWAVERS_CH27_EDGE_PRESERVING_EPSILON="0.004"
$env:KWAVERS_CH27_EDGE_PRESERVING_STEP="0.12"
$env:KWAVERS_CH27_EDGE_PRESERVING_ITERATIONS="1"
python pykwavers/examples/book/ch27_seismic_fwi_brain_imaging.py
```

`KWAVERS_CH27_STACK_OFFSETS` is relative to the source index in the resampled
3-D inversion volume.  The default stack is
`-8,-6,-4,-2,0,2,4,6,8,10,12,14,16`, which requests thirteen
planes from the same reconstructed 3-D array and records every nonempty
CT-derived brain slice in `metrics.json`.  The centroid ROI figure crops each
valid slice around the CT-derived brain-mask centroid and skips empty mask
planes.  This is a reproducible
deep-midline proxy for inspecting pons-through-thalamus coverage; it is not an
anatomical segmentation.

Run the custom histotripsy-monitoring simulation separately:

```powershell
python pykwavers/examples/book/ch27_histotripsy_fwi_rtm.py
```

Its metrics file records equal-area Dice, AUPRC, and contrast-to-noise ratio for
each reconstruction family. The latest generated fusion metrics are:

| Scenario | Fusion Dice | Fusion AUPRC | Fusion CNR |
| --- | ---: | ---: | ---: |
| compact intrinsic | 0.897 | 0.724 | 3.74 |
| shock elongated | 0.826 | 0.802 | 3.90 |
| multi-packet | 0.933 | 0.995 | 3.96 |

---

## 27.8 Boundary of replication

The published npj Digital Medicine study is a full 3-D time-domain FWI pipeline.
This chapter is a bounded in-repository reproduction of the acquisition and
inversion contract:

- same documented 1024-element hemispherical helmet count;
- CT-derived skull and brain acoustic model;
- finite-frequency source/receiver phase sensitivity;
- weak-Westervelt second-harmonic encoded channels;
- adjoint migration from simulated encoded ultrasound data;
- optimization of a 3-D brain sound-speed volume from simulated helmet data;
- frequency continuation, Sobolev update conditioning, Charbonnier
  edge-preserving proximal regularization, and separate enhanced display output
  for visual inspection;
- multi-slice visualization by slicing the reconstructed simulated 3-D volume;
- centroid-cropped reconstruction visualization for deep midline slices;
- value-semantic verification in the Rust core.

It is not a substitute for the full clinical-scale time-domain pipeline,
measured transducer calibration, skull attenuation calibration, or patient
diagnostic validation.  The histotripsy-monitoring subchapters define the
selected architecture for the next therapy-tracking increment; active 3-D
migration, active 3-D acoustic FWI, attenuation-weighted rows, and weak
second-harmonic encoded rows are implemented in this chapter today.
