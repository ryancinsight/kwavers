import os

source = r"d:\kwavers\kwavers\src\solver\multiphysics\fluid_structure.rs"
dest_dir = r"d:\kwavers\kwavers\src\solver\multiphysics\fluid_structure"

with open(source, "r", encoding="utf-8") as f:
    lines = f.readlines()

os.makedirs(dest_dir, exist_ok=True)

with open(os.path.join(dest_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.writelines(lines[0:69]) # header and docs
    f.write("\n")
    f.write("pub mod interface;\n")
    f.write("pub mod coefficients;\n")
    f.write("pub mod solver;\n\n")
    f.write("pub use interface::FsiInterface;\n")
    f.write("pub use coefficients::ReflectionTransmissionCoefficients;\n")
    f.write("pub use solver::FluidStructureSolver;\n")

# lines[69:73] contains imports
imports = "".join(lines[69:74])

with open(os.path.join(dest_dir, "interface.rs"), "w", encoding="utf-8") as f:
    f.write("use crate::core::error::{KwaversError, KwaversResult};\n")
    f.write("use ndarray::Array3;\n\n")
    f.writelines(lines[75:198])
    f.write("\n#[cfg(test)]\nmod tests {\n    use super::*;\n")
    f.writelines(lines[715:752])
    f.write("}\n")

with open(os.path.join(dest_dir, "coefficients.rs"), "w", encoding="utf-8") as f:
    f.write("use crate::math::fft::Complex64;\n")
    f.write("use super::interface::FsiInterface;\n\n")
    f.writelines(lines[199:341])
    f.write("\n#[cfg(test)]\nmod tests {\n    use super::*;\n")
    f.writelines(lines[752:832]) # Normal reflection, energy cons, oblique refl
    f.write("}\n")

with open(os.path.join(dest_dir, "solver.rs"), "w", encoding="utf-8") as f:
    f.write("use crate::core::error::{KwaversError, KwaversResult};\n")
    f.write("use ndarray::{Array1, Array3};\n")
    f.write("use super::interface::FsiInterface;\n")
    f.write("use super::coefficients::ReflectionTransmissionCoefficients;\n\n")
    f.writelines(lines[342:709])
    f.write("\n#[cfg(test)]\nmod tests {\n    use super::*;\n")
    f.writelines(lines[832:984]) # ghost cell traction and velocity check tests
    f.write("}\n")

print("fluid_structure extracted successfully!")
