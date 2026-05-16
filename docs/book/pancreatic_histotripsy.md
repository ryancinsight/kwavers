# Pancreatic Cancer Histotripsy: PDAC Treatment Planning

Chapter 21f implements treatment-planning simulations for histotripsy of
pancreatic ductal adenocarcinoma (PDAC). It loads an abdominal CT with
pancreas and tumour segmentation from the Medical Segmentation Decathlon
Task07_Pancreas dataset (or a calibrated synthetic phantom when the dataset
is absent) and runs three clinical histotripsy exposure regimes through a
128-element curved abdominal therapy array placed on the anterior skin
surface.

## Clinical Context

Pancreatic ductal adenocarcinoma accounts for approximately 90% of
pancreatic cancers and carries a five-year survival rate below 12% because
most patients present with locally advanced or metastatic disease (Siegel
et al. 2023, CA Cancer J Clin 73:17). Surgery remains the only curative
option but is feasible in fewer than 20% of cases at diagnosis. Histotripsy
offers a non-thermal, non-ionising mechanical ablation mechanism that erodes
tumour tissue through bubble-cloud cavitation without the dose constraints
of thermal methods or the collateral radiation burden of external-beam
therapy (Xu et al. 2021, Nat Rev Urol 18:678).

Preclinical in vitro evidence for pancreatic histotripsy was established by
Mauch et al. (2017, IEEE TUFFC 64:1386) using a 500 kHz single-element
focused transducer producing peak negative pressures up to −16 MPa at
10 cm depth. In vivo porcine pancreatic ablation was demonstrated by
Chen et al. (2022, Ultrasound Med Biol 48:1002) with complete histological
cell destruction in the treated zone and no evidence of pancreatitis or
vascular injury in the 72-hour follow-up window. The HOPE4PANCREAS
investigational programme extends the concept to human subjects
(NCT05370300; first-in-human enrolment ongoing at time of writing).

## Acoustic Access: Subcostal Hepatic Window

Pancreatic histotripsy faces a unique acoustic challenge absent from kidney
and liver applications: the pancreatic head and body lie retroperitoneally
behind the gastric antrum and transverse colon, whose gas-filled lumina
produce near-total acoustic reflection. The preferred clinical access route
is the right subcostal window, directing the beam through the right hepatic
lobe into the retroperitoneal fat surrounding the pancreatic head. The
hepatic parenchyma provides a clear acoustic path (c ≈ 1595 m/s,
α ≈ 0.87 Np/cm at 500 kHz) that attenuates the beam by approximately
2–4 dB over the hepatic transit (Bamber 1989, Ultrasound Med Biol 15:159).

The simulation detects bowel-gas columns along every AP column from the
anterior skin to the PDAC centroid and selects the column minimising gas
voxels while maximising hepatic tissue. This acoustic-window selection
mirrors clinical scanning practice and feeds the transducer placement step.

## Aperture Contract

The therapy array is a 128-element curved abdominal array with 70 mm outer
aperture, 100 mm radius of curvature, and 500 kHz centre frequency. These
parameters are drawn from the public histotripsy literature for retroperitoneal
pancreatic access (Mauch et al. 2017). No proprietary or vendor-specific
element layout is used; the forward model computes a paraxial Gaussian focal
envelope as described in the Mathematical Contract below.

Focal spot sizes at 500 kHz (f# = 100/70 ≈ 1.43, c = 1540 m/s):

```
λ = 3.08 mm
w_lateral  = 1.41 · λ · F#  ≈  6.2 mm FWHM
w_axial    = 7   · λ · F#²  ≈  44  mm FWHM
```

The elongated axial dimension is the primary raster-pitch constraint for
PDAC treatment: the lesion is cigar-shaped along the beam axis with an
aspect ratio of approximately 7:1. Electronic beam-steering interleaves
across multiple focal sub-positions to improve lateral coverage per unit
transducer time.

## Tissue Properties

All acoustic and thermal properties follow Duck (1990) and the IT'IS
Foundation v4.1 database (Hasgall et al. 2022). Cavitation thresholds
follow Vlaisavljevich et al. (2015, 2016) and Maxwell et al. (2013).

α₀ values are in Np/m/MHz (1 Np/m/MHz = 0.08686 dB/cm/MHz).

| Tissue | ρ (kg/m³) | c (m/s) | α₀ (Np/m/MHz) | p_t,1MHz (MPa) | σ (MPa) |
|--------|-----------|---------|----------------|-----------------|---------|
| Fat    | 911       | 1440    | 4.84           | 14.0            | 2.0     |
| Muscle | 1090      | 1588    | 8.05           | 25.0            | 2.0     |
| Liver  | 1079      | 1595    | 8.69           | 20.6            | 4.6     |
| Pancreas | 1040    | 1543    | 6.00           | 22.0            | 3.5     |
| PDAC   | 1060      | 1555    | 8.50           | 26.0            | 3.0     |
| Bone   | 1908      | 4080    | 250            | ∞               | —       |

PDAC exhibits a higher cavitation threshold than normal pancreatic
parenchyma due to its dense desmoplastic stroma (Hruban et al. 2019,
Cold Spring Harb Perspect Med 9:a026575). The value of 26 MPa at 1 MHz
with σ = 3 MPa is estimated from Vlaisavljevich et al. (2016) data for
dense fibrous tissue analogues; no measured PDAC-specific intrinsic
threshold has been published at time of writing.

At 500 kHz the frequency-dependent correction (Vlaisavljevich 2015 Eq. 9)
shifts p_t downward by 1.4 · log₁₀(0.5) ≈ −0.42 MPa, giving an effective
PDAC threshold of approximately 25.6 MPa at the operating frequency.

## Mathematical Contract

### Forward propagation

The acoustic pressure field is computed by the paraxial Rayleigh-Sommerfeld
model with 3-D attenuation. Let (x, y, z) be the AP, RL, and SI axes;
x = 0 is the skin surface and x = x_focus is the PDAC centroid depth.

```
p(x, y, z) = P_source · G_lat(y, z) · G_ax(x) · exp(−∫₀ˣ α(x′,y,z) dx′)
```

where the focal Gaussian envelopes are:

```
G_lat(y, z) = exp(−(r_lat² / (2 σ_lat²)))
G_ax(x)     = exp(−((x − x_f)² / (2 σ_ax²)))
r_lat²       = (y − y_f)² + (z − z_f)²
σ_lat        = w_lat / 2.355,    σ_ax = w_axial / 2.355
```

and the cumulative attenuation integral is evaluated as a running product
of the per-voxel attenuation (in Np/m, converted from dB/cm by
1 dB/cm = 11.513 Np/m).

### Cavitation probability

The intrinsic threshold is tissue- and frequency-dependent
(Vlaisavljevich 2015/2016, Maxwell 2013):

```
p_t(f, T) = p_{t,1MHz} + 1.4 MPa · log₁₀(f / 1 MHz) − 0.3 MPa · max(0, T − 20)
P_cav(x)  = 0.5 · (1 + erf((|p(x)| − p_t) / (σ · √2)))
```

Temperature correction (−0.3 MPa/°C) uses the transient temperature
computed from the per-pulse heat deposition.

### Thermal model

Per-pulse heat deposition:

```
Q_pulse = 2 · α · G_shock · I_eff
I_eff   = (P_source · η)² / (2 ρ c),    η = P_pp / P_pnp
dT_pulse = Q_pulse · t_on / (ρ · c_p)
```

Steady-state temperature rise from Pennes bioheat equation with a
Gaussian focal kernel of radius w_f = 1.41 λ F# / 2.355:

```
ΔT_ss = Q_avg · w_f² / (4 κ + W_b ρ_b c_b w_f²)
Q_avg  = Q_pulse · (t_on · PRF)
```

Thermal dose CEM43 (Sapareto & Dewey 1984):

```
CEM43 = R^(43 − T) · t_treatment / 60,    R = 0.5 (T ≥ 43 °C), 0.25 (T < 43 °C)
```

Thermal ablation threshold: CEM43 ≥ 240 min (Dewey 1994).

### Raster superposition

Per-shot lesion footprints are superposed on a farthest-point raster
whose pitch is half the per-shot half-extent, ensuring > 60% overlap.
Raster centres are eroded from the tumour mask by one pitch so each
per-shot footprint does not spill outside the PDAC outline. Spillover
into healthy pancreatic parenchyma is reported as a confinement metric.
Total treatment time accounts for electronic beam-steering interleaving
across multiple focal sub-positions.

## Exposure Regimes

Three regimes adapted for 500 kHz pancreatic access:

**μs intrinsic-threshold** (500 kHz, 30 MPa PNP, 2 μs pulses, 200 Hz PRF):
The gold-standard histotripsy mechanism for precise, confined ablation.
At 500 kHz with 30 MPa, the erf-CDF gives P_cav ≈ 0.97 in PDAC
(σ = 3 MPa, corrected threshold ≈ 25.6 MPa). Per-spot PRF is limited to
200 Hz by the ~5 ms residual-bubble dissolution time (Vlaisavljevich 2015).
100 pulses/spot drives accumulated P_cav ≥ 0.99 per raster point.

**ms shock-vapor** (500 kHz, 15 MPa PNP, 10 ms pulses, 1 Hz PRF):
Nonlinear shock formation at 500 kHz / 15 MPa drives focal temperature to
boiling (≥ 100 °C) within the pulse window. Bubble-cloud nucleation from
vapour cavities then erodes a cigar-shaped lesion. Shock pressure ratio
estimated at 5–8× from published kzk simulations at equivalent f/1.4
geometry (Bessonova & Khokhlova 2013). Per-spot PRF constrained to 1 Hz by
vapour-cavity dissolution time; 8 interleaved sub-spots → 8 Hz effective
transducer PRF.

**ms sub-threshold with PRF dithering** (500 kHz, 18 MPa PNP, 5 ms, 10 Hz):
Below the shock-boiling threshold but above 65% of the intrinsic threshold,
PRF dithering (Bader 2018, Mancia 2020) drives cumulative cavitation-cloud
erosion without stable thermal elevation. Fifty pulses/spot at 10 Hz per
spot achieves full erosion; 8-spot interleave gives 80 Hz effective PRF.

## Figures

- `fig01_pdac_histotripsy_overview.png`: CT + labels, pressure field,
  lesion mask, and cavitation-dose heatmap for each regime.
- `fig02_pdac_thermal.png`: Transient temperature, steady-state temperature,
  and log₁₀(CEM43) for each regime.
- `embedded_figures.md`: Base64-inlined PNGs and tabulated metrics.
- `metrics.json`: Scenario-level coverage, lesion volume, treatment time,
  peak temperature, array parameters, and dataset provenance.

## Scope Limits

All figures are model-consistent forward simulation outputs, not measured
hardware data. The forward model is a paraxial single-frequency Gaussian
focal envelope with 3-D cumulative attenuation; it does not include
off-axis diffraction, shear-wave coupling, nonlinear waveform distortion,
or temporal focusing via phased delays. The raster superposition assumes
ideal electronic beam steering with no inter-shot residual bubble
interference. The bowel-gas shadow model is a binary column check; the
clinical requirement for dynamic real-time guidance to avoid gas pockets is
not captured. PDAC cavitation threshold is estimated from surrogate data;
prospective validation requires in vitro porcine pancreatic histotripsy
measurements at 500 kHz with fresh tissue at body temperature.
The tissue segmentation is derived automatically from CECT HU thresholds;
it does not use radiologist-reviewed contours and may misclassify
peripancreatic structures in patients with atypical anatomy.

## Data Source

MSD Task07_Pancreas: Antonelli et al. (2022, Nature Commun 13:4128).
License: CC-BY-SA 4.0.
Archive: https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar (~7.5 GB total).

The script streams only the first ~60 MB of the archive to retrieve the
first test-set CT (`imagesTs/pancreas_044.nii.gz`, ~28 MB), which is a
real histologically confirmed PDAC staging CT acquired in the portal
venous phase. The file is cached in `data/msd_pancreas_sample/ct_ts.nii.gz`
after the initial download. No pre-computed segmentation label file is
required; tissue labels are derived automatically from CECT HU values
as described in the Auto-segmentation section of the script docstring.

Set `KWAVERS_CH21F_SYNTHETIC=1` to skip the network download and use a
lightweight deterministic volume for CI pipelines.
