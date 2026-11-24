<a id="kw-med-abs-001"></a>

## KW-MED-ABS-001 - Medium absorption storage-units divergence (Rust source vs pykwavers wheel) [minor] - todo

priority: research; needs: none; scope: kwavers Medium absorption field / pykwavers bindings; milestone: M-imaging-capability

- **Finding (Build 16, `B16_ATT.md` S3):** the Medium absorption field's units diverge between the Rust source documentation and the installed pykwavers wheel (0.1.0). The source documents the stored field as Np/m at 1 MHz; empirically the wheel's per-voxel array takes the k-Wave prefactor DIRECTLY in dB/(MHz^y*cm): array value 1.0 reproduced `set_alpha_coeff(1.0)` exactly at 2.45/3.7 MHz (40.47 / 56.96 Np/m effective), while the documented Np/m conversion (11.51) over-attenuated ~8.5x (280.8 Np/m).
- **Acceptance test:** `b16/validate_att.py` - homogeneous tissue, linear PSTD, Hilbert-envelope decay; control-corrected alpha within ~5% of the analytic law at 2.45 and 3.7 MHz (achieved -3.9% / -4.7% under the wheel convention).
- **Ask:** pin ONE convention in the source, document it at the Medium field, and add a regression test at the validate_att conditions so a wheel rebuild cannot silently flip the units. Until then, every consumer must state which convention it assumed (`b16/gen_rf16.py` carries a loud docstring to this effect).
