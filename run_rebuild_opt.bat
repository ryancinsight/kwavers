@echo off
set RUSTC=C:\Users\RyanClanton\.rustup\toolchains\nightly-x86_64-pc-windows-gnu\bin\rustc.exe
cd /d D:\kwavers\kwavers
"C:\Users\RyanClanton\.cargo\bin\cargo.exe" build --release --example seismic_imaging_3d_demo --features "dicom ritk" --target-dir D:\kwavers\target_opt > D:\kwavers\kwavers\examples\output\rebuild_opt.log 2>&1
echo BUILD_EXIT_CODE=%ERRORLEVEL% >> D:\kwavers\kwavers\examples\output\rebuild_opt.log
