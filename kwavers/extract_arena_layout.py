import os

source_file = r"d:\kwavers\kwavers\src\core\arena\layout.rs"
output_dir = r"d:\kwavers\kwavers\src\core\arena\layout"

with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_lines(start, end):
    return "".join(lines[start-1:end])

# Create directory
os.makedirs(output_dir, exist_ok=True)

# 1. mod.rs
mod_content = get_lines(1, 25) + """
pub mod alignment;
pub mod numa_aware;
pub mod packing;
pub mod pool;
pub mod soa;
pub mod tiling;

pub use alignment::*;
pub use numa_aware::*;
pub use packing::*;
pub use pool::*;
pub use soa::*;
pub use tiling::*;
""" + get_lines(37, 102)

with open(os.path.join(output_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.write(mod_content)

# 2. alignment.rs
alignment_content = """use super::CACHE_LINE_SIZE;

""" + get_lines(104, 151) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(1105, 1132) + "\n}\n"
with open(os.path.join(output_dir, "alignment.rs"), "w", encoding="utf-8") as f:
    f.write(alignment_content)

# 3. soa.rs
soa_content = """use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use ndarray::{ArrayView3, ArrayViewMut3};
use rayon::prelude::*;

use crate::core::error::{KwaversError, KwaversResult, SystemError, ValidationError};
use super::{cache_aligned_size, CACHE_LINE_SIZE};

""" + get_lines(153, 471) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(1135, 1153) + "\n" + get_lines(1191, 1225) + "\n" + get_lines(1259, 1273) + "\n}\n"
with open(os.path.join(output_dir, "soa.rs"), "w", encoding="utf-8") as f:
    f.write(soa_content)

# 4. pool.rs
pool_content = """use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use ndarray::{ArrayViewMut3, ArrayView3};

use crate::core::error::{KwaversError, KwaversResult, SystemError, ValidationError};
use super::{cache_aligned_size, CACHE_LINE_SIZE};

""" + get_lines(473, 717) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(1155, 1188) + "\n}\n"
with open(os.path.join(output_dir, "pool.rs"), "w", encoding="utf-8") as f:
    f.write(pool_content)

# 5. tiling.rs
tiling_content = get_lines(719, 922) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(1241, 1256) + "\n}\n"
with open(os.path.join(output_dir, "tiling.rs"), "w", encoding="utf-8") as f:
    f.write(tiling_content)

# 6. numa_aware.rs
numa_content = """use std::alloc::{alloc, Layout};
use std::ptr::NonNull;
use rayon::prelude::*;

use crate::core::error::{KwaversError, KwaversResult, SystemError};
use super::{NumaPolicy, NUMA_ALIGNMENT, CACHE_LINE_SIZE};

""" + get_lines(924, 1023) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(1275, 1286) + "\n}\n"
with open(os.path.join(output_dir, "numa_aware.rs"), "w", encoding="utf-8") as f:
    f.write(numa_content)

# 7. packing.rs
packing_content = """use super::align_up;

""" + get_lines(1025, 1096) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(1228, 1238) + "\n}\n"
with open(os.path.join(output_dir, "packing.rs"), "w", encoding="utf-8") as f:
    f.write(packing_content)

print("Extraction completed!")
