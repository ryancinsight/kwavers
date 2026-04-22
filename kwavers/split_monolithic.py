import os
import sys
import re

file_path = r"d:\kwavers\kwavers\src\solver\multiphysics\monolithic.rs"
target_dir = r"d:\kwavers\kwavers\src\solver\multiphysics\monolithic"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Gather components
header_docs = "".join(lines[0:69])
imports = "".join(lines[69:83])

c_info = "".join(lines[84:105])
phys_coeff = "".join(lines[106:161])
nk_config = "".join(lines[205:235])

# In coupler struct, we replace private fields with pub(super)
# Wait, let's just do it string-wise
coupler_struct_src = "".join(lines[162:204])
coupler_struct_src = re.sub(r'(\s+)newton_config:', r'\1pub(super) newton_config:', coupler_struct_src)
coupler_struct_src = re.sub(r'(\s+)gmres_config:', r'\1pub(super) gmres_config:', coupler_struct_src)
coupler_struct_src = re.sub(r'(\s+)convergence_history:', r'\1pub(super) convergence_history:', coupler_struct_src)
coupler_struct_src = re.sub(r'(\s+)physics_components:', r'\1pub(super) physics_components:', coupler_struct_src)
coupler_struct_src = re.sub(r'(\s+)physics_coefficients:', r'\1pub(super) physics_coefficients:', coupler_struct_src)
coupler_struct_src = re.sub(r'(\s+)du_scratch:', r'\1pub(super) du_scratch:', coupler_struct_src)
coupler_struct_src = re.sub(r'(\s+)gmres_solver:', r'\1pub(super) gmres_solver:', coupler_struct_src)
coupler_struct_src = re.sub(r'(\s+)grid_spacing:', r'\1pub(super) grid_spacing:', coupler_struct_src)

# impl MonolithicCoupler ... solve_coupled_step is 236 to 465
coupler_impl_top = "".join(lines[236:465])
coupler_impl_bottom = "".join(lines[708:718]) # convergence_history, physics_coefficients

sorted_field_keys = "".join(lines[467:474]).replace('fn sorted_field_keys', 'pub(super) fn sorted_field_keys')
compute_residual = "".join(lines[474:602]).replace('fn compute_residual', 'pub(super) fn compute_residual')
jvp = "".join(lines[602:623]).replace('fn jacobian_vector_product', 'pub(super) fn jacobian_vector_product')
line_search = "".join(lines[623:653]).replace('fn line_search', 'pub(super) fn line_search')

flatten = "".join(lines[653:680]).replace('fn flatten_fields', 'pub(super) fn flatten_fields')
unflatten = "".join(lines[680:703]).replace('fn unflatten_fields', 'pub(super) fn unflatten_fields')
norm = "".join(lines[703:708]).replace('fn norm', 'pub(super) fn norm')

laplacian = "".join(lines[719:794]).replace('fn laplacian_3d', 'pub(super) fn laplacian_3d')
tests = lines[794:] # 795 is mod tests

# We need to distribute tests into proper files.
# But for simplicity, we can keep the tests in mod.rs and import everything, or split them.
# There are specific tests:
test_coupler = ""
test_config = ""
test_residual = ""
test_utils = ""

current_test_block = []
current_test_name = ""
in_test = False
all_tests = []
for line in tests[2:-1]: # skip `#[cfg(test)] mod tests {` and `}`
    if line.strip() == "#[test]" or line.strip() == "#[cfg(feature = \"gpu\")]" or "test_laplacian" in line or "fn test_" in line or "/// " in line:
        pass # we'll group correctly
        
# Actually, splitting tests manually line by line:
# test_monolithic_coupler_creation: 798..808
# test_newton_krylov_config_default: 808..816
# test_physics_coefficients_default: 816..824
# test_flatten_unflatten_round_trip: 824..852
# test_compute_residual_zero_fields: 852..872
# test_laplacian_uniform_field: 872..893
# test_laplacian_unit_vs_nonunit_spacing: 893..913
# test_laplacian_quadratic_field_exact: 913..938
# test_laplacian_zero_field: 938..951
# test_photoacoustic_default_gruneisen_not_one: 951..967
# test_photoacoustic_source_scales_with_gruneisen: 967..1021

test_chunks = {
    'coupler': lines[798:808] + lines[852:872] + lines[967:1021], # it uses compute_residual which will be in residual
    'config': lines[808:824] + lines[951:967],
    'utils': lines[824:852] + lines[872:951],
}

def to_test_module(test_lines):
    if not test_lines: return ""
    return "\n#[cfg(test)]\nmod tests {\n    use super::*;\n" + "".join(test_lines) + "}\n"

# Prepare mod.rs
mod_content = header_docs + """
pub mod config;
pub mod coupler;
pub mod residual;
pub mod utils;

pub use config::{CouplingConvergenceInfo, NewtonKrylovConfig, PhysicsCoefficients};
pub use coupler::MonolithicCoupler;
"""

# Prepare config.rs
config_content = """use crate::core::constants::{
    ACOUSTIC_ABSORPTION_TISSUE, DENSITY_WATER_NOMINAL, GRUNEISEN_WATER_37C,
    OPTICAL_ABSORPTION_TISSUE_NIR, REDUCED_SCATTERING_TISSUE_NIR, SOUND_SPEED_TISSUE,
    SPECIFIC_HEAT_WATER,
};

""" + c_info + phys_coeff + nk_config + to_test_module(test_chunks['config'])

# Prepare utils.rs
utils_content = """use ndarray::{s, Array3};
use std::collections::HashMap;
use crate::domain::field::UnifiedFieldType;

""" + sorted_field_keys + flatten + unflatten + norm + laplacian + to_test_module(test_chunks['utils'])

# Prepare residual.rs
residual_content = """use crate::core::error::KwaversResult;
use crate::domain::field::UnifiedFieldType;
use ndarray::{s, Array3};
use super::coupler::MonolithicCoupler;
use super::utils::{laplacian_3d};

impl MonolithicCoupler {
""" + compute_residual + jvp + line_search + "}\n" + to_test_module(lines[852:872] + lines[967:1021]).replace("use super::*;","use super::*;\n    use crate::solver::integration::nonlinear::GMRESConfig;\n    use super::super::config::NewtonKrylovConfig;\n    use ndarray::{Array3, s};\n")

# Prepare coupler.rs
coupler_content = """use crate::core::error::KwaversResult;
use crate::domain::field::UnifiedFieldType;
use crate::domain::grid::Grid;
use crate::domain::plugin::Plugin;
use crate::solver::integration::nonlinear::{GMRESConfig, GMRESSolver};
use log::{debug, warn};
use ndarray::{s, Array3};
use std::collections::HashMap;
use std::time::Instant;

use super::config::{CouplingConvergenceInfo, NewtonKrylovConfig, PhysicsCoefficients};
use super::utils;

""" + coupler_struct_src + coupler_impl_top[:-2] + coupler_impl_bottom + "}\n" + to_test_module(lines[798:808]).replace("use super::*;", "use super::*;\n    use super::super::config::NewtonKrylovConfig;\n    use crate::solver::integration::nonlinear::GMRESConfig;\n")

# Fix calls inside coupler.rs to utils
coupler_content = coupler_content.replace('Self::sorted_field_keys', 'utils::sorted_field_keys')
coupler_content = coupler_content.replace('Self::flatten_fields', 'utils::flatten_fields')
coupler_content = coupler_content.replace('Self::unflatten_fields', 'utils::unflatten_fields')
coupler_content = coupler_content.replace('Self::norm', 'utils::norm')

# Fix norm inner compute_residual
residual_content = residual_content.replace('Self::norm', 'super::utils::norm')

# Write files
with open(os.path.join(target_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.write(mod_content)

with open(os.path.join(target_dir, "config.rs"), "w", encoding="utf-8") as f:
    f.write(config_content)

with open(os.path.join(target_dir, "utils.rs"), "w", encoding="utf-8") as f:
    f.write(utils_content)

with open(os.path.join(target_dir, "residual.rs"), "w", encoding="utf-8") as f:
    f.write(residual_content)

with open(os.path.join(target_dir, "coupler.rs"), "w", encoding="utf-8") as f:
    f.write(coupler_content)

os.remove(file_path)

print("Split completed successfully!")
