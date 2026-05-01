@echo off
cd /d D:\kwavers\kwavers
"C:\Users\RyanClanton\.cargo\bin\cargo.exe" run --release --example seismic_imaging_3d_demo --features "dicom ritk" > D:\kwavers\kwavers\examples\output\seismic_3d_demo_run.log 2>&1
