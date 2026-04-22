import os

source_file = r"d:\kwavers\kwavers\src\solver\forward\hybrid\bem_fem_coupling.rs"
output_dir = r"d:\kwavers\kwavers\src\solver\forward\hybrid\bem_fem_coupling"

with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_lines(start, end):
    return "".join(lines[start-1:end])

os.makedirs(output_dir, exist_ok=True)

# 1. mod.rs
mod_content = get_lines(1, 50) + """
pub mod config;
pub mod interface;
pub mod coupler;
pub mod solver;

pub use config::*;
pub use interface::*;
pub use coupler::*;
pub use solver::*;
"""
with open(os.path.join(output_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.write(mod_content)

# 2. config.rs
config_content = get_lines(51, 76) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(826, 833) + "\n}\n"
with open(os.path.join(output_dir, "config.rs"), "w", encoding="utf-8") as f:
    f.write(config_content)

# 3. interface.rs
interface_content = """use crate::core::error::KwaversResult;
use crate::domain::mesh::tetrahedral::TetrahedralMesh;
use nalgebra::Vector3;
use std::collections::HashMap;

""" + get_lines(78, 297) + """
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::tetrahedral::BoundaryType;

""" + get_lines(756, 796) + "\n" + get_lines(835, 934) + "\n}\n"
with open(os.path.join(output_dir, "interface.rs"), "w", encoding="utf-8") as f:
    f.write(interface_content)

# 4. coupler.rs
coupler_content = """use crate::core::error::KwaversResult;
use crate::domain::mesh::tetrahedral::TetrahedralMesh;
use super::{BemFemCouplingConfig, BemFemInterface};
use crate::math::linear_algebra::sparse::solver::Preconditioner;
use crate::math::linear_algebra::sparse::{
    CompressedSparseRowMatrix, CoordinateMatrix, IterativeSolver, SolverConfig,
};
use crate::math::numerics::operators::TrilinearInterpolator;
use crate::solver::forward::bem::solver::{BemConfig, BemSolver};
use nalgebra::{Matrix3, Vector3};
use ndarray::Array1;
use num_complex::{Complex64, ComplexFloat};
use std::collections::HashMap;

""" + get_lines(299, 683) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(798, 809) + "\n" + get_lines(937, 1038) + "\n}\n"
with open(os.path.join(output_dir, "coupler.rs"), "w", encoding="utf-8") as f:
    f.write(coupler_content)

# 5. solver.rs
solver_content = """use crate::core::error::KwaversResult;
use crate::domain::mesh::tetrahedral::TetrahedralMesh;
use num_complex::Complex64;
use super::{BemFemCouplingConfig, BemFemCoupler, BemFemInterface};

""" + get_lines(685, 748) + """
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::tetrahedral::BoundaryType;

""" + get_lines(812, 823) + "\n}\n"
with open(os.path.join(output_dir, "solver.rs"), "w", encoding="utf-8") as f:
    f.write(solver_content)

print("Extraction completed!")
