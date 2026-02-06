fn main() {
    // This is the build script for kwavers
    // It will be run by Cargo before compiling the project

    // Tell Cargo to rerun the build script if any files in src/ change
    println!("cargo:rerun-if-changed=src/");
}
