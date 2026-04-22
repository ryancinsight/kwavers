import os

source_file = r"d:\kwavers\kwavers\src\solver\inverse\pinn\ml\cavitation_coupled.rs"
output_dir = r"d:\kwavers\kwavers\src\solver\inverse\pinn\ml\cavitation_coupled"

with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_lines(start, end):
    return "".join(lines[start-1:end])

os.makedirs(output_dir, exist_ok=True)

# 1. mod.rs
mod_content = get_lines(1, 30) + """
pub mod config;
pub mod domain;
pub mod mie_scattering;

pub use config::*;
pub use domain::*;
pub use mie_scattering::*;
"""
with open(os.path.join(output_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.write(mod_content)

# 2. config.rs
config_content = """use crate::physics::bubble_dynamics::BubbleParameters;

""" + get_lines(31, 79)
with open(os.path.join(output_dir, "config.rs"), "w", encoding="utf-8") as f:
    f.write(config_content)

# 3. domain.rs
domain_content = """use crate::physics::bubble_dynamics::{BubbleState, KellerMiksisModel};
use crate::solver::inverse::pinn::ml::physics::{
    BoundaryComponent, BoundaryConditionSpec, BoundaryPosition, CouplingInterface, CouplingType,
    InitialConditionSpec, PhysicsDomain, PhysicsLossWeights, PhysicsParameters,
    PhysicsValidationMetric,
};
use burn::prelude::ElementConversion;
use burn::tensor::{backend::AutodiffBackend, Tensor};
use std::collections::HashMap;
use super::config::{CavitationCouplingConfig, CavitationCouplingType};
use super::mie_scattering::mie_backscatter_form_function;

""" + get_lines(81, 664) + """
#[cfg(test)]
mod tests {
    use super::*;
    type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

""" + get_lines(871, 930) + "\n}\n"
with open(os.path.join(output_dir, "domain.rs"), "w", encoding="utf-8") as f:
    f.write(domain_content)

# 4. mie_scattering.rs
mie_content = """use num_complex::Complex;

""" + get_lines(666, 864) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(934, 1030) + "\n}\n"
with open(os.path.join(output_dir, "mie_scattering.rs"), "w", encoding="utf-8") as f:
    f.write(mie_content)

print("Extraction completed!")
