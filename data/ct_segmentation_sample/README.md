# CT-Space Segmentation Execution Sample

This sample exists to exercise the Chapter 25 BBB-opening path where a
segmentation is already defined in CT space.

Files:

- `../rire_patient_109/patient_109_ct.nii.gz`
- `segmentation.nii.gz`

The CT is the converted RIRE patient 109 head CT.  The segmentation is a
deterministic ellipsoidal target generated in the CT coordinate frame.  It is
not a clinical tumor annotation and is not labeled as GBM.  Its purpose is to
verify the valid execution contract: CT plus segmentation is sufficient for
BBB subspot planning and skull-acoustic phase/attenuation correction.

For clinical GBM execution, replace this sample with a same-patient CT and
segmentation from CFB-GBM or another licensed CT-backed cohort.
