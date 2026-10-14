//! Emit scalar implementations from one canonical set of operation bodies.

use std::{env, fs, io, path::PathBuf};

fn main() -> io::Result<()> {
    let relative = "src/inverse/fwi/frequency_domain/born_kernel/arithmetic/operations.rs.in";
    println!("cargo:rerun-if-changed={relative}");
    println!("cargo:rerun-if-changed=build.rs");
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative);
    let bodies = fs::read_to_string(&source).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("reading {}: {error}", source.display()),
        )
    })?;
    let directory = env::var_os("OUT_DIR")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Cargo did not provide OUT_DIR"))?;
    let destination = PathBuf::from(directory).join("native_arithmetic.rs");
    let mut generated = String::new();
    for scalar in [
        "f32",
        "f64",
        "::eunomia::F16",
        "::eunomia::Bf16",
        "::eunomia::F32",
        "::eunomia::F64",
    ] {
        generated.push_str(&format!(
            "impl sealed::Sealed for {scalar} {{}}\nimpl Arithmetic for {scalar} {{\n{bodies}\n}}\n"
        ));
    }
    fs::write(&destination, generated).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("writing {}: {error}", destination.display()),
        )
    })
}
