import os

source_file = r"d:\kwavers\kwavers\src\solver\forward\fdtd\simd_stencil.rs"
dest_dir = r"d:\kwavers\kwavers\src\solver\forward\fdtd\simd\generic"

os.makedirs(dest_dir, exist_ok=True)

with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_lines(start, end):
    return "".join(lines[start-1:end])

def fix_upper(text):
    return text.replace("I-1", "nx-1").replace("J-1", "ny-1").replace("K-1", "nz-1").replace(" I ", " nx ").replace(" J ", " ny ").replace(" K ", " nz ")

# mod.rs
with open(os.path.join(dest_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.write(get_lines(1, 63))
    f.write("pub mod pressure;\npub mod velocity;\n\n")
    f.write(get_lines(64, 115))
    struct_body = get_lines(116, 140).replace("    config:", "    pub(super) config:")\
                                     .replace("    pressure_coeff:", "    pub(super) pressure_coeff:")\
                                     .replace("    velocity_coeff:", "    pub(super) velocity_coeff:")\
                                     .replace("    nx:", "    pub(super) nx:")\
                                     .replace("    ny:", "    pub(super) ny:")\
                                     .replace("    nz:", "    pub(super) nz:")\
                                     .replace("    num_tiles_x:", "    pub(super) num_tiles_x:")\
                                     .replace("    num_tiles_y:", "    pub(super) num_tiles_y:")\
                                     .replace("    num_tiles_z:", "    pub(super) num_tiles_z:")\
                                     .replace("    vel_scratch:", "    pub(super) vel_scratch:")\
                                     .replace("    pres_scratch:", "    pub(super) pres_scratch:")
    f.write(struct_body)
    f.write(get_lines(141, 183))
    f.write(get_lines(443, 458))
    f.write("\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n")
    f.write(get_lines(463, 476))
    f.write(get_lines(590, 610))

# pressure.rs
with open(os.path.join(dest_dir, "pressure.rs"), "w", encoding="utf-8") as f:
    f.write("use super::SimdStencilProcessor;\n")
    f.write("use crate::core::error::{KwaversError, KwaversResult};\n")
    f.write("use ndarray::Array3;\n\n")
    f.write("impl SimdStencilProcessor {\n")
    p_code = get_lines(184, 264)
    # Apply naming convention fix to gap audit
    p_code = fix_upper(p_code)
    f.write(p_code)
    f.write(get_lines(319, 387))
    f.write("}\n\n")
    f.write(get_lines(459, 461))
    f.write("    use super::super::SimdStencilConfig;\n    use super::*;\n\n")
    f.write(get_lines(477, 491))
    f.write(get_lines(506, 560))
    f.write("}\n")

# velocity.rs
with open(os.path.join(dest_dir, "velocity.rs"), "w", encoding="utf-8") as f:
    f.write("use super::SimdStencilProcessor;\n")
    f.write("use crate::core::error::{KwaversError, KwaversResult};\n")
    f.write("use ndarray::Array3;\n\n")
    f.write("impl SimdStencilProcessor {\n")
    v_code = get_lines(266, 318)
    v_code = fix_upper(v_code)
    f.write(v_code)
    f.write(get_lines(389, 442))
    f.write("}\n\n")
    f.write(get_lines(459, 461))
    f.write("    use super::super::SimdStencilConfig;\n    use super::*;\n\n")
    f.write(get_lines(492, 504))
    f.write(get_lines(561, 588))
    f.write("}\n")
