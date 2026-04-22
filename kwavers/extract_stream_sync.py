import os

source_file = r"d:\kwavers\kwavers\src\analysis\visualization\stream_sync.rs"
output_dir = r"d:\kwavers\kwavers\src\analysis\visualization\stream_sync"

with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_lines(start, end):
    return "".join(lines[start-1:end])

os.makedirs(output_dir, exist_ok=True)

# 1. mod.rs
mod_content = get_lines(1, 41) + """
pub mod state;
pub mod pacer;
pub mod budget;
pub mod quality;
pub mod coordinator;

pub use state::*;
pub use pacer::*;
pub use budget::*;
pub use quality::*;
pub use coordinator::*;
"""
with open(os.path.join(output_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.write(mod_content)

# 2. state.rs
state_content = """use std::time::{Duration, Instant};
use tracing::info;

""" + get_lines(43, 89) + "\n" + get_lines(790, 834) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(915, 920) + "\n" + get_lines(929, 964) + "\n}\n"
with open(os.path.join(output_dir, "state.rs"), "w", encoding="utf-8") as f:
    f.write(state_content)

# 3. pacer.rs
pacer_content = """use std::collections::VecDeque;
use std::time::{Duration, Instant};
use parking_lot::{Mutex, RwLock};
use tracing::{info, instrument, trace};
use super::state::PacingStrategy;

""" + get_lines(91, 283) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(868, 872) + "\n" + get_lines(966, 992) + "\n}\n"
with open(os.path.join(output_dir, "pacer.rs"), "w", encoding="utf-8") as f:
    f.write(pacer_content)

# 4. budget.rs
budget_content = """use std::collections::VecDeque;
use std::time::Instant;
use parking_lot::{Mutex, RwLock};
use tracing::{info, trace, warn};

""" + get_lines(285, 429) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(874, 893) + "\n" + get_lines(1004, 1011) + "\n}\n"
with open(os.path.join(output_dir, "budget.rs"), "w", encoding="utf-8") as f:
    f.write(budget_content)

# 5. quality.rs
quality_content = """use std::collections::VecDeque;
use std::time::{Duration, Instant};
use parking_lot::{Mutex, RwLock};
use tracing::{info, instrument, warn};

""" + get_lines(431, 605) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(841, 866) + "\n" + get_lines(895, 913) + "\n" + get_lines(994, 1002) + "\n}\n"
with open(os.path.join(output_dir, "quality.rs"), "w", encoding="utf-8") as f:
    f.write(quality_content)

# 6. coordinator.rs
coord_content = """use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use tokio::sync::Notify;
use tracing::{info, warn};

use super::state::{PacingStrategy, SyncState, SyncStatistics};
use super::pacer::FramePacer;
use super::budget::LatencyBudget;
use super::quality::{QualityController, QualityLevel};

""" + get_lines(607, 788) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(922, 927) + "\n" + get_lines(1013, 1024) + "\n}\n"
with open(os.path.join(output_dir, "coordinator.rs"), "w", encoding="utf-8") as f:
    f.write(coord_content)

print("Extraction completed!")
