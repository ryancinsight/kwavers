# Chapter 29 — Intravascular Ultrasound Imaging and Therapy

This chapter implements a deterministic intravascular ultrasound (IVUS) slice
for coronary wall imaging and localized vessel-wall therapy. The executable
chapter is:

```powershell
python pykwavers/examples/book/ch30_intravascular_ultrasound.py
```

It emits five figures and `metrics.json` under
`docs/book/figures/ch30/`. The generated frames are model-consistent
simulations, not redistributed patient IVUS images and not measured catheter
data.

## Dataset Contract

The validation dataset target is the public IVUS segmentation corpus used by
IVUS-Net and the IVUS Challenge lineage. It is appropriate for this chapter
because it provides B-mode coronary frames with lumen and media-adventitia
boundary labels, which are the acceptance targets for simulated IVUS image
formation and contour recovery. The local executable uses a deterministic
analytic vessel phantom with the same 384 x 384 frame contract so the book can
build without downloading clinical images.

The chapter records the external validation sources in `metrics.json`:

- IVUS-Net code and dataset pointer:
  <https://github.com/Kulbear/ivus-segmentation-icsm2018>
- IVUS-Net paper:
  <https://arxiv.org/abs/1806.03583>
- IVUS Challenge dataset pointer:
  <http://www.cvc.uab.es/IVUSchallenge2011/dataset.html>

## Transducer Design

The design is a dual-frequency catheter:

- A 64-element circumferential imaging ring at `20 MHz`, matching the
  solid-state IVUS frequency class used by current digital catheters.
- A side-looking therapeutic sector at `1.5 MHz`, matching the low-frequency
  intravascular microbubble-delivery design space used for acoustic radiation
  force and delivery pulses.
- A `0.55 mm` catheter radius, `10 mm` maximum imaging radius, and `0.5 mm`
  pullback step.
- A therapy pressure of `300 kPa`, `5%` duty cycle, and `45 s` treatment
  packet for localized model-drug delivery into the plaque sector.

The design is not a claim about a proprietary catheter. It is a reproducible
physics contract: imaging and therapy share the intravascular coordinate frame,
but operate at separate frequencies because wall-resolution imaging and
microbubble displacement have different wavelength and attenuation constraints.

## Mathematical Contract

The radial B-mode frame samples a deterministic reflectivity field:

```text
R(x, y) = |grad(rho c)| + S(x, y)
s(r, theta) = R(r cos theta, r sin theta)
              exp[-2 alpha(f) (r - r_c)]
```

where `S` is seeded Rayleigh speckle, `rho c` is acoustic impedance,
`alpha(f)` is tissue attenuation, and `r_c` is catheter radius. A Gaussian
radial pulse kernel models finite axial pulse length before logarithmic
compression over a fixed `60 dB` display range.

The therapy model computes pressure, intensity, absorbed power, temperature,
and delivery fraction:

```text
p(r, theta) = p0 G(theta - theta0) exp[-(r - r_c) / L]
I = p^2 / (2 rho c)
Q = 2 alpha(f) I D
Delta T = Q tau / (rho c_p)
M = p_MPa / sqrt(f_MHz)
```

The delivery field is restricted to the plaque/fibrous-cap target and is
weighted by acoustic radiation force. The acceptance criteria are
input-sensitive: wall echoes must exceed lumen echoes, the therapy deposition
must peak in the selected plaque sector, the target/off-target deposition ratio
must exceed one, and thermal rise must stay below the model safety bound for
the selected packet.

## Figures

Outputs:

- `fig01_dataset_and_anatomy.{png,pdf}`: dataset contract and analytic vessel
  anatomy.
- `fig02_transducer_design.{png,pdf}`: circumferential imaging ring,
  guidewire lumen, and side-looking therapy sector.
- `fig03_ivus_bmode_simulation.{png,pdf}`: radial A-line frame and
  scan-converted B-mode with lumen and vessel-wall contours.
- `fig04_microbubble_therapy_map.{png,pdf}`: pressure, delivery fraction, and
  temperature maps.
- `fig05_intravascular_usage_sequence.{png,pdf}`: catheter crossing, IVUS
  imaging, localized therapy, and post-treatment check.

## Research Alignment

Current commercial and research constraints motivate the selected parameters:

- Philips Eagle Eye Platinum lists a `20 MHz` digital IVUS catheter with
  `20 mm` maximum imaging diameter and `5F` guide-catheter compatibility:
  <https://www.usa.philips.com/healthcare/product/HC85900P/eagle-eye-platinum-catheter-the-1-choice-of-physicians-for-intravascular-imaging-in-the-us>
- Boston Scientific OptiCross lists coronary IVUS options at `40 MHz` and
  `60 MHz`, so the chapter keeps the dataset anchor at `20 MHz` while
  preserving higher-frequency validation as a documented extension:
  <https://www.bostonscientific.com/gb/en/products/all-products/interventional-cardiology/pci-guidance/single-unit-devices/opticross-coronary-ivus-catheters/p/FP00000330>
- A CMUT IVUS review identifies the common IVUS catheter center-frequency
  range as `20-40 MHz` and distinguishes mechanical and solid-state catheter
  layouts:
  <https://www.nature.com/articles/s41378-020-0181-z>
- Kilroy et al. designed and validated a `1.5 MHz` IVUS transducer for
  acoustic-radiation-force microbubble displacement and vessel-wall delivery:
  <https://pubmed.ncbi.nlm.nih.gov/24569249/>
- Forward-looking intravascular sonothrombolysis work shows why a separate
  lower-frequency therapy aperture is needed when the target is clot
  dissolution rather than coronary wall drug delivery:
  <https://www.nature.com/articles/s41598-017-03492-4>

The next implementation increment is a real external-dataset loader that maps
IVUS-Net contour files into the same lumen/media-adventitia contract and runs
the synthetic forward model against measured B-mode frames for contour-level
differential validation.
