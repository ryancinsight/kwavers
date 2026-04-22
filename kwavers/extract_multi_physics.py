import os

source_file = r"d:\kwavers\kwavers\src\simulation\multi_physics.rs"
output_dir = r"d:\kwavers\kwavers\src\simulation\multi_physics"

with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_lines(start, end):
    return "".join(lines[start-1:end])

# Create directory
os.makedirs(output_dir, exist_ok=True)

# 1. mod.rs
mod_content = get_lines(1, 38) + """
pub mod conservation;
pub mod coupler;
pub mod interface;
pub mod schwarz;
pub mod solver;

pub use conservation::ConservationEnforcer;
pub use coupler::FieldCoupler;
pub use interface::CouplingInterface;
pub use schwarz::SchwarzCoupling;
pub use solver::MultiPhysicsSolver;

""" + get_lines(40, 138)

with open(os.path.join(output_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.write(mod_content)

# 2. interface.rs
interface_content = """use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;

""" + get_lines(634, 686)

with open(os.path.join(output_dir, "interface.rs"), "w", encoding="utf-8") as f:
    f.write(interface_content)

# 3. coupler.rs
coupler_content = """use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::math::numerics::operators::TrilinearInterpolator;
use ndarray::ArrayView3;
use std::collections::HashMap;

use super::{ConservationEnforcer, CoupledPhysicsSolver, CouplingInterface, PhysicsDomain};

""" + get_lines(140, 274)

with open(os.path.join(output_dir, "coupler.rs"), "w", encoding="utf-8") as f:
    f.write(coupler_content)

# 4. conservation.rs
conservation_content = """use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use ndarray::{Array3, ArrayView3};

""" + get_lines(276, 490) + "\n" + get_lines(628, 632) + """

#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(1136, 1213) + "\n}\n"

with open(os.path.join(output_dir, "conservation.rs"), "w", encoding="utf-8") as f:
    f.write(conservation_content)

# 5. schwarz.rs
schwarz_content = """use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::collections::HashMap;

use super::{CoupledPhysicsSolver, FieldCoupler, PhysicsDomain};

""" + get_lines(492, 626) + """

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use ndarray::ArrayView3;

""" + get_lines(1017, 1074) + "\n" + get_lines(1215, 1274) + "\n}\n"

with open(os.path.join(output_dir, "schwarz.rs"), "w", encoding="utf-8") as f:
    f.write(schwarz_content)

# 6. solver.rs
solver_content = """use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::collections::HashMap;

use super::{CoupledPhysicsSolver, CouplingStrategy, FieldCoupler, MultiPhysicsConfig, PhysicsDomain};

""" + get_lines(688, 1011) + """

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use ndarray::ArrayView3;

""" + get_lines(1017, 1134) + "\n" + get_lines(1276, 1307) + "\n}\n"

with open(os.path.join(output_dir, "solver.rs"), "w", encoding="utf-8") as f:
    f.write(solver_content)

print("Extraction completed successfully!")
