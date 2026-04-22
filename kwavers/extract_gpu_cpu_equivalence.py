import os

source_file = r"d:\kwavers\kwavers\src\solver\validation\gpu_cpu_equivalence.rs"
output_dir = r"d:\kwavers\kwavers\src\solver\validation\gpu_cpu_equivalence"

with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_lines(start, end):
    return "".join(lines[start-1:end])

os.makedirs(output_dir, exist_ok=True)

# 1. mod.rs
mod_content = get_lines(1, 68) + """
pub mod constants;
pub mod report;
pub mod validator;
pub mod runner;
pub mod ieee754;

pub use constants::*;
pub use report::*;
pub use validator::*;
pub use runner::*;
pub use ieee754::*;
"""
with open(os.path.join(output_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.write(mod_content)

# 2. constants.rs
constants_content = get_lines(79, 104) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(893, 900) + "\n}\n"
with open(os.path.join(output_dir, "constants.rs"), "w", encoding="utf-8") as f:
    f.write(constants_content)

# 3. report.rs
report_content = "use std::fmt;\n\n" + get_lines(106, 226) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(849, 891) + "\n" + get_lines(1093, 1108) + "\n}\n"
with open(os.path.join(output_dir, "report.rs"), "w", encoding="utf-8") as f:
    f.write(report_content)

# 4. validator.rs
validator_content = """use ndarray::{Array3, Zip};
use crate::core::error::ValidationError;
use super::{EquivalenceReport, DEFAULT_ABSOLUTE_TOLERANCE, DEFAULT_RELATIVE_TOLERANCE, MEASUREMENT_STEPS, WARMUP_STEPS};

""" + get_lines(228, 365) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(935, 1037) + "\n" + get_lines(1048, 1091) + "\n}\n"
with open(os.path.join(output_dir, "validator.rs"), "w", encoding="utf-8") as f:
    f.write(validator_content)

# 5. runner.rs
runner_content = """use crate::core::error::{KwaversError, ValidationError};
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::{Array3, Array2};
use crate::solver::forward::fdtd::{FdtdConfig, FdtdSolver, KSpaceCorrectionMode};
use crate::solver::interface::Solver;
use super::{EquivalenceReport, EquivalenceValidator};

""" + get_lines(367, 626) + """
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::medium::HomogeneousMedium;

""" + get_lines(744, 843) + "\n" + get_lines(1039, 1046) + "\n" + get_lines(1110, 1124) + "\n}\n"
with open(os.path.join(output_dir, "runner.rs"), "w", encoding="utf-8") as f:
    f.write(runner_content)

# 6. ieee754.rs
ieee_content = get_lines(628, 735) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(902, 933) + "\n}\n"
with open(os.path.join(output_dir, "ieee754.rs"), "w", encoding="utf-8") as f:
    f.write(ieee_content)

print("Extraction completed!")
