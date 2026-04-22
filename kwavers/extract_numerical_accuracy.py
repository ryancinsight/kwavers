import os
import re

source_file = r"d:\kwavers\kwavers\src\solver\validation\numerical_accuracy.rs"
output_dir = r"d:\kwavers\kwavers\src\solver\validation\numerical_accuracy"

with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_lines(start, end):
    return "".join(lines[start-1:end])

os.makedirs(output_dir, exist_ok=True)

# 1. mod.rs
mod_content = """//! Numerical methods validation tests
//!
//! References:
//! - Treeby & Cox (2010) - "MATLAB toolbox"
//! - Gear & Wells (1984) - "Multirate linear multistep methods"
//! - Berger & Oliger (1984) - "Adaptive mesh refinement"
//! - Persson & Peraire (2006) - "Sub-cell shock capturing"

#[cfg(test)]
pub(crate) mod helpers;
#[cfg(test)]
mod pstd;
#[cfg(test)]
mod amr;
#[cfg(test)]
mod shock;
#[cfg(test)]
mod time_integration;
#[cfg(test)]
mod mms;
"""
with open(os.path.join(output_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.write(mod_content)

# 2. helpers.rs
helpers_content = """use ndarray::{Array1, Array3};
use std::f64::consts::PI;

""" + get_lines(22, 77).replace("fn compute_laplacian_1d", "pub(crate) fn compute_laplacian_1d") \
                       .replace("fn compute_laplacian_3d", "pub(crate) fn compute_laplacian_3d") \
+ "\n" + get_lines(512, 557).replace("fn compute_phase_error_lsq", "pub(crate) fn compute_phase_error_lsq")

with open(os.path.join(output_dir, "helpers.rs"), "w", encoding="utf-8") as f:
    f.write(helpers_content)

# 3. pstd.rs
pstd_content = """#[cfg(test)]
mod tests {
    use crate::domain::grid::Grid;
    use crate::domain::medium::core::CoreMedium;
    use crate::domain::medium::HomogeneousMedium;
    use crate::solver::pstd::PSTDSolver;
    use crate::solver::pstd::PSTDConfig as PstdConfig;
    use ndarray::{Array2, Array3};
    use std::f64::consts::PI;
    use super::super::helpers::*;

    const CFL_NUMBER: f64 = 0.3;
    const PPW_MINIMUM: usize = 6;

""" + get_lines(81, 260) + "\n" + get_lines(559, 1024) + "\n}\n"

with open(os.path.join(output_dir, "pstd.rs"), "w", encoding="utf-8") as f:
    f.write(pstd_content)

# 4. time_integration.rs
time_integration_content = """#[cfg(test)]
mod tests {
    use ndarray::Array3;
    use super::super::helpers::*;

    const CFL_NUMBER: f64 = 0.3;

""" + get_lines(261, 397) + "\n" + get_lines(1119, 1187) + "\n}\n"

with open(os.path.join(output_dir, "time_integration.rs"), "w", encoding="utf-8") as f:
    f.write(time_integration_content)

# 5. amr.rs
amr_content = """#[cfg(test)]
mod tests {
    use crate::domain::grid::Grid;
    use crate::solver::amr::AMRSolver;
    use ndarray::Array3;
    use std::f64::consts::PI;

""" + get_lines(399, 434) + "\n}\n"

with open(os.path.join(output_dir, "amr.rs"), "w", encoding="utf-8") as f:
    f.write(amr_content)

# 6. shock.rs
shock_content = """#[cfg(test)]
mod tests {
    use ndarray::Array3;

""" + get_lines(436, 504) + "\n}\n"

with open(os.path.join(output_dir, "shock.rs"), "w", encoding="utf-8") as f:
    f.write(shock_content)

# 7. mms.rs
mms_content = """#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use std::f64::consts::PI;
    use super::super::helpers::*;

""" + get_lines(1057, 1090) + "\n" + get_lines(1221, 1271) + "\n}\n"

with open(os.path.join(output_dir, "mms.rs"), "w", encoding="utf-8") as f:
    f.write(mms_content)

print("Extraction completed!")
