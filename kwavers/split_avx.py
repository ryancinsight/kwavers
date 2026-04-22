import os

source_file = r"d:\kwavers\kwavers\src\solver\forward\fdtd\avx512_stencil.rs"
dest_dir = r"d:\kwavers\kwavers\src\solver\forward\fdtd\simd\avx512"

os.makedirs(dest_dir, exist_ok=True)

with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_lines(start, end):
    # start and end are 1-indexed inclusive
    return "".join(lines[start-1:end])

# Create mod.rs
with open(os.path.join(dest_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.write(get_lines(1, 63))
    f.write("pub mod pressure;\n")
    f.write("pub mod velocity;\n\n")
    f.write(get_lines(64, 115))
    # inject pub(super)
    struct_body = get_lines(116, 137).replace("    config:", "    pub(super) config:")\
                                     .replace("    nx:", "    pub(super) nx:")\
                                     .replace("    ny:", "    pub(super) ny:")\
                                     .replace("    nz:", "    pub(super) nz:")\
                                     .replace("    pressure_coeff:", "    pub(super) pressure_coeff:")\
                                     .replace("    velocity_coeff:", "    pub(super) velocity_coeff:")\
                                     .replace("    pressure_central_coeff:", "    pub(super) pressure_central_coeff:")\
                                     .replace("    simd_config:", "    pub(super) simd_config:")
    f.write(struct_body)
    f.write(get_lines(138, 195))
    f.write(get_lines(882, 892)) # get_metrics
    f.write("}\n\n")
    f.write(get_lines(893, 908)) # Avx512Metrics
    f.write("\n#[cfg(test)]\nmod tests {\n    use super::*;\n    use ndarray::Array3;\n\n")
    f.write(get_lines(913, 946))
    f.write("}\n")

# Create pressure.rs
with open(os.path.join(dest_dir, "pressure.rs"), "w", encoding="utf-8") as f:
    f.write("use super::Avx512StencilProcessor;\n")
    f.write("use crate::core::error::{KwaversError, KwaversResult};\n")
    f.write("use ndarray::Array3;\n\n")
    f.write("impl Avx512StencilProcessor {\n")
    f.write(get_lines(196, 666))
    f.write("}\n\n")
    f.write(get_lines(909, 912))
    f.write(get_lines(947, 972))

# Create velocity.rs
with open(os.path.join(dest_dir, "velocity.rs"), "w", encoding="utf-8") as f:
    f.write("use super::Avx512StencilProcessor;\n")
    f.write("use crate::core::error::{KwaversError, KwaversResult};\n")
    f.write("use ndarray::Array3;\n\n")
    f.write("impl Avx512StencilProcessor {\n")
    f.write(get_lines(668, 881))
    f.write("}\n")
