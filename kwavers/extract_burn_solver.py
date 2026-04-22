import os

source_file = r"d:\kwavers\kwavers\src\solver\inverse\pinn\ml\burn_wave_equation_3d\solver.rs"
output_dir = r"d:\kwavers\kwavers\src\solver\inverse\pinn\ml\burn_wave_equation_3d\solver"

with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_lines(start, end):
    return "".join(lines[start-1:end])

# Let's cleanly inject `pub(crate)` into private methods that need module-level sharing
def make_pub(block):
    return block.replace("fn scalar_f32", "pub(crate) fn scalar_f32") \
                .replace("fn tensor_column_vec_f32", "pub(crate) fn tensor_column_vec_f32") \
                .replace("fn extract_initial_condition_tensors", "pub(crate) fn extract_initial_condition_tensors") \
                .replace("fn extract_velocity_initial_condition_tensor", "pub(crate) fn extract_velocity_initial_condition_tensor") \
                .replace("fn compute_physics_loss", "pub(crate) fn compute_physics_loss") \
                .replace("fn compute_temporal_derivative_at_t0", "pub(crate) fn compute_temporal_derivative_at_t0") \
                .replace("fn compute_bc_loss_internal", "pub(crate) fn compute_bc_loss_internal")

os.makedirs(output_dir, exist_ok=True)

# 1. mod.rs
mod_content = get_lines(1, 30) + """
pub mod core;
pub mod diagnostics;
pub mod losses;
pub mod ics;
pub mod collocation;
pub mod training;

pub use core::BurnPINN3DWave;
"""
with open(os.path.join(output_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.write(mod_content)

# 2. core.rs
core_content = get_lines(31, 42) + get_lines(174, 284) + "\n" + get_lines(511, 564) + "\n" + make_pub(get_lines(867, 926)) + "\n}\n" + """
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    type TestBackend = Autodiff<NdArray>;

""" + get_lines(1117, 1143) + "\n" + get_lines(1172, 1193) + "\n}\n"
with open(os.path.join(output_dir, "core.rs"), "w", encoding="utf-8") as f:
    f.write(core_content)

# 3. diagnostics.rs
diag_content = """use burn::tensor::{backend::Backend, Tensor};
use crate::core::error::{KwaversError, KwaversResult};
use super::core::BurnPINN3DWave;

""" + get_lines(77, 172) + "\nimpl<B: Backend> BurnPINN3DWave<B> {\n" + get_lines(847, 865) + "\n}\n"
with open(os.path.join(output_dir, "diagnostics.rs"), "w", encoding="utf-8") as f:
    f.write(diag_content)

# 4. losses.rs
losses_content = """use burn::tensor::{backend::Backend, Tensor};
use crate::core::error::KwaversResult;
use super::core::BurnPINN3DWave;
use crate::solver::inverse::pinn::ml::burn_wave_equation_3d::config::BurnLossWeights3D;

""" + get_lines(44, 75) + "\nimpl<B: Backend> BurnPINN3DWave<B> {\n" + make_pub(get_lines(566, 696)) + "\n" + make_pub(get_lines(1001, 1108)) + """
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use crate::solver::inverse::pinn::ml::burn_wave_equation_3d::{config::BurnPINN3DConfig, geometry::Geometry3D};
    type TestBackend = Autodiff<NdArray>;

""" + get_lines(1272, 1307) + "\n}\n"
with open(os.path.join(output_dir, "losses.rs"), "w", encoding="utf-8") as f:
    f.write(losses_content)

# 5. ics.rs
ics_content = """use burn::tensor::{backend::Backend, Tensor, TensorData};
use crate::core::error::{KwaversError, KwaversResult};
use super::core::BurnPINN3DWave;

impl<B: Backend> BurnPINN3DWave<B> {
""" + make_pub(get_lines(698, 845)) + "\n}\n"
with open(os.path.join(output_dir, "ics.rs"), "w", encoding="utf-8") as f:
    f.write(ics_content)

# 6. collocation.rs
colloc_content = """use burn::tensor::{backend::Backend, Tensor, TensorData};
use super::core::BurnPINN3DWave;
use crate::solver::inverse::pinn::ml::burn_wave_equation_3d::config::BurnPINN3DConfig;

impl<B: Backend> BurnPINN3DWave<B> {
""" + make_pub(get_lines(928, 999)) + """
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use crate::solver::inverse::pinn::ml::burn_wave_equation_3d::geometry::Geometry3D;
    use crate::core::error::{SystemError};
    type TestBackend = Autodiff<NdArray>;

""" + get_lines(1195, 1270) + "\n}\n"
with open(os.path.join(output_dir, "collocation.rs"), "w", encoding="utf-8") as f:
    f.write(colloc_content)

# 7. training.rs
train_content = """use burn::tensor::{backend::AutodiffBackend, Tensor, TensorData};
use std::time::Instant;
use crate::core::error::{KwaversError, KwaversResult};
use super::core::BurnPINN3DWave;
use super::losses::LossScales;
use crate::solver::inverse::pinn::ml::burn_wave_equation_3d::config::BurnTrainingMetrics3D;
use crate::solver::inverse::pinn::ml::burn_wave_equation_3d::optimizer::SimpleOptimizer3D;

impl<B: AutodiffBackend> BurnPINN3DWave<B> {
""" + get_lines(286, 509) + """
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use crate::solver::inverse::pinn::ml::burn_wave_equation_3d::{config::BurnPINN3DConfig, geometry::Geometry3D};
    type TestBackend = Autodiff<NdArray>;

""" + get_lines(1145, 1170) + "\n}\n"
with open(os.path.join(output_dir, "training.rs"), "w", encoding="utf-8") as f:
    f.write(train_content)

print("Extraction completed!")
