# Example: Seismic Imaging 3D Demo

**Crate**: `kwavers`
**Run**: `cargo run -p kwavers --example seismic_imaging_3d_demo --features "dicom ritk"`
**Source**: [`crates/kwavers/examples/seismic_imaging_3d_demo.rs`](../../../crates/kwavers/examples/seismic_imaging_3d_demo.rs)

## What This Example Demonstrates

True 3D transcranial ultrasound FWI — extends the 2D quasi-3D demo to full 3D (NX=64, NY=48, NZ=64) with Fibonacci-sphere acquisition geometry, trilinear CT resampling, 3D MNI atlas brain velocity, and T1 MRI tissue mapping.

## References

- Aubry 2003: JASA 113(1) — skull bone-volume-fraction acoustic model
- Marsac 2017: J. Ther. Ultrasound — transcranial FWI protocol
- Guasch 2020: npj Digital Medicine — 3D brain FWI pipeline
- Duck 1990: "Physical Properties of Tissue" — soft-tissue velocities
- Treeby & Cox 2010: JASA — fractional-Laplacian absorption
