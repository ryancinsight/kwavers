# RIRE Patient 109 CT/MR

This directory contains converted NIfTI CT and MR volumes from the
Retrospective Image Registration Evaluation Project (RIRE) patient 109
dataset.

Source: https://rire.insight-journal.org/download_data

Original file downloaded:

- `patient_109/ct.tar.gz`
- PyScience mirror of RIRE patient 109 MR-T1/MR-T2 MetaImage data, used only
  because the original IPFS MR archive was unavailable from the local runtime.

Local converted file:

- `patient_109_ct.nii.gz`
- `patient_109_mr_t1.nii.gz`
- `patient_109_mr_t2.nii.gz`

License: Creative Commons Attribution 3.0 United States.

Conversion:

- CT: decompressed `image.bin.Z` from the RIRE MetaImage package and
  interpreted the raw buffer as big-endian signed 16-bit CT data with
  dimensions `512 x 512 x 41`.
- MR: interpreted the mirrored MetaImage raw buffers as big-endian signed
  16-bit MR data with `x` as the fastest-varying index. Conversion reshapes
  the buffer as `(z, y, x)` and transposes to NIfTI `(x, y, z)`.
- Wrote NIfTI files in the RIRE `L:P:H` coordinate convention. Voxel spacing is
  `0.404341 x 0.404341 x 3.0 mm` for CT and `0.82 x 0.82 x 3.0 mm` for MR.

Scope: these CT/MR volumes supply the same-patient registration pair for
Chapter 25 HIFU planning. The CT supplies skull acoustics; the MR supplies the
subject image used as the atlas-registration intermediate. The UPenn-GBM sample
still supplies tumor MRI and segmentation when a same-patient CT-backed GBM
case is unavailable.
