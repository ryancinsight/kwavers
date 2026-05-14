# UPenn-GBM Local Execution Sample

This directory contains a minimal executable subject slice from the
UPenn-GBM dataset for Chapter 25 GBM BBB-opening subspot planning.

Source: https://github.com/data-nih/tcia/releases/tag/upenn-gbm

Downloaded subject: `sub-002`

Files:

- `sub-002/sub-002_ce-gd_T1w.nii.gz`
- `sub-002/sub-002_FLAIR.nii.gz`
- `sub-002/sub-002_T1w.nii.gz`
- `sub-002/sub-002_T2w.nii.gz`
- `sub-002/sub-002_seg.nii.gz`

License: CC BY 4.0, as published by the TCIA/NIH data release.

Scope: this sample provides real co-registered MRI volumes and tumor
segmentation for the Chapter 25 BBB-opening branch. It does not include CT.
The modality bridge marks this case as MRI-space GBM subspot geometry only:
sample CT registration is visual QC, and skull acoustics remain non-CT-backed
until a same-patient CT or QC-accepted synthetic CT NIfTI artifact is supplied.
