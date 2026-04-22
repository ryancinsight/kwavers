import os

source_file = r"d:\kwavers\kwavers\src\solver\inverse\pinn\ml\electromagnetic\residuals.rs"
output_dir = r"d:\kwavers\kwavers\src\solver\inverse\pinn\ml\electromagnetic\residuals"

with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_lines(start, end):
    return "".join(lines[start-1:end])

# Create directory
os.makedirs(output_dir, exist_ok=True)

# 1. mod.rs
mod_content = """//! PINN Electromagnetic Residuals
//! 
//! Extracted into bounded contexts to comply with SRP and SoC.

pub mod constants;
pub mod sources;
pub mod statics;
pub mod scalar_wave;
pub mod te_mode;
pub mod tm_mode;

pub use constants::EPS_FD_F32;
pub use sources::{compute_charge_density, compute_current_density_z};
pub use statics::{electrostatic_residual, magnetostatic_residual, quasi_static_residual};
pub use scalar_wave::wave_propagation_residual;
pub use te_mode::{te_mode_faraday_residual, te_mode_ampere_x_residual, te_mode_ampere_y_residual, te_mode_gauss_residual};
pub use tm_mode::{tm_mode_faraday_x_residual, tm_mode_faraday_y_residual, tm_mode_ampere_z_residual};
"""
with open(os.path.join(output_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.write(mod_content)

# 2. constants.rs
with open(os.path.join(output_dir, "constants.rs"), "w", encoding="utf-8") as f:
    f.write(get_lines(6, 31))

# 3. sources.rs
sources_content = """use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

""" + get_lines(685, 837) + """

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
    use std::collections::HashMap;

""" + get_lines(496, 682) + "\n}\n"

with open(os.path.join(output_dir, "sources.rs"), "w", encoding="utf-8") as f:
    f.write(sources_content)

# 4. statics.rs
statics_content = """use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
use crate::solver::inverse::pinn::ml::BurnPINN2DWave;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use super::sources::{compute_charge_density, compute_current_density_z};
use super::constants::EPS_FD_F32;

""" + get_lines(33, 303)

with open(os.path.join(output_dir, "statics.rs"), "w", encoding="utf-8") as f:
    f.write(statics_content)

# 5. scalar_wave.rs
scalar_content = """use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
use crate::solver::inverse::pinn::ml::BurnPINN2DWave;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use super::constants::EPS_FD_F32;

""" + get_lines(305, 383) + """

#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(386, 495) + "\n}\n"

with open(os.path.join(output_dir, "scalar_wave.rs"), "w", encoding="utf-8") as f:
    f.write(scalar_content)

# 6. te_mode.rs
te_content = """use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
use crate::solver::inverse::pinn::ml::BurnPINN2DWave;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use super::constants::EPS_FD_F32;
use super::sources::compute_charge_density;

""" + get_lines(839, 1018)

with open(os.path.join(output_dir, "te_mode.rs"), "w", encoding="utf-8") as f:
    f.write(te_content)

# 7. tm_mode.rs
tm_content = """use crate::solver::inverse::pinn::ml::physics::PhysicsParameters;
use crate::solver::inverse::pinn::ml::BurnPINN2DWave;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use super::constants::EPS_FD_F32;
use super::sources::compute_current_density_z;

""" + get_lines(1020, 1378)

with open(os.path.join(output_dir, "tm_mode.rs"), "w", encoding="utf-8") as f:
    f.write(tm_content)

print("Extraction completed successfully!")
