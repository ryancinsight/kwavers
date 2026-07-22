# 35. Seismic Imaging: Acoustic and Elastic Full-Waveform Inversion

Seismic imaging applies exploration-geophysics algorithms — acoustic and elastic full-waveform inversion (FWI), reverse-time migration (RTM), and adjoint-state methods — to transcranial ultrasound. The wave equation is identical; the domain (skull, brain, soft tissue) and frequency range (0.5–2 MHz) differ.

## Acoustic FWI

Recover the sound-speed map `c(x)` from observed ultrasound data by minimising the L2 misfit between measured and synthetic waveforms. The adjoint-state method computes the gradient `∂J/∂c` without explicit sensitivity matrices.

## Elastic FWI

Extend to the full elastic wave equation to recover both P-wave `c_P` and S-wave `c_S` velocities, enabling shear-modulus reconstruction for stiff-lesion detection (elastography).

## Transcranial Applications

- Skull CT template alignment (MOFI, ADR 016)
- Brain anomaly reconstruction (Guasch 2020)
- Stiff-lesion shear-wave imaging

## Examples

- [Seismic Imaging Demo](examples/seismic_imaging_demo.md) — 2D quasi-3D brain FWI
- [Seismic Imaging 3D Demo](examples/seismic_imaging_3d_demo.md) — true 3D with MNI atlas
