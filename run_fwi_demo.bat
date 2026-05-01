@echo off
cd /d D:\kwavers\kwavers
cargo run --release --example seismic_imaging_demo --features "dicom ritk" > D:\kwavers\kwavers\examples\output\seismic_demo_run.log 2>&1
