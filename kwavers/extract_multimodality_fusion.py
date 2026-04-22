import os

source_file = r"d:\kwavers\kwavers\src\domain\imaging\multimodality_fusion.rs"
output_dir = r"d:\kwavers\kwavers\src\domain\imaging\multimodality_fusion"

with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_lines(start, end):
    return "".join(lines[start-1:end])

os.makedirs(output_dir, exist_ok=True)

# 1. mod.rs
mod_content = get_lines(1, 43) + """
pub mod fusion;
pub mod image;
pub mod manager;
pub mod parameters;
pub mod registration;
pub mod transform;

pub use fusion::*;
pub use image::*;
pub use manager::*;
pub use parameters::*;
pub use registration::*;
pub use transform::*;
"""
with open(os.path.join(output_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.write(mod_content)

# 2. image.rs
image_content = """use ndarray::Array3;

""" + get_lines(63, 108) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(1007, 1011) + "\n}\n"
with open(os.path.join(output_dir, "image.rs"), "w", encoding="utf-8") as f:
    f.write(image_content)

# 3. transform.rs
transform_content = """use crate::core::error::KwaversResult;
use ndarray::Array2;

""" + get_lines(110, 236) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(1014, 1024) + "\n" + get_lines(1067, 1148) + "\n}\n"
with open(os.path.join(output_dir, "transform.rs"), "w", encoding="utf-8") as f:
    f.write(transform_content)

# 4. parameters.rs
params_content = get_lines(238, 287) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(1027, 1031) + "\n" + get_lines(1055, 1058) + "\n}\n"
with open(os.path.join(output_dir, "parameters.rs"), "w", encoding="utf-8") as f:
    f.write(params_content)

# 5. registration.rs
reg_content = """use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array2;
use super::{ImageData, RegistrationTransform, TransformationType};

""" + get_lines(289, 745) + """
#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ImageModality;

""" + get_lines(1034, 1038) + "\n" + get_lines(1061, 1065) + "\n" + get_lines(1150, 1247) + "\n}\n"
with open(os.path.join(output_dir, "registration.rs"), "w", encoding="utf-8") as f:
    f.write(reg_content)

# 6. fusion.rs
fusion_content = """use crate::core::error::KwaversResult;
use ndarray::Array3;
use super::{FusionParameters, FusionMethod, ImageData, RegistrationTransform};

""" + get_lines(747, 887) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(1041, 1045) + "\n}\n"
with open(os.path.join(output_dir, "fusion.rs"), "w", encoding="utf-8") as f:
    f.write(fusion_content)

# 7. manager.rs
mgr_content = """use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::collections::HashMap;
use super::{
    FusionEngine, FusionParameters, ImageData, RegistrationEngine,
    RegistrationParams, RegistrationTransform,
};

""" + get_lines(48, 61) + "\n" + get_lines(889, 1000) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(1048, 1052) + "\n}\n"
with open(os.path.join(output_dir, "manager.rs"), "w", encoding="utf-8") as f:
    f.write(mgr_content)

print("Extraction completed!")
