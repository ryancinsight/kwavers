import os

source_file = r"d:\kwavers\kwavers\src\gpu\recovery.rs"
output_dir = r"d:\kwavers\kwavers\src\gpu\recovery"

with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

def get_lines(start, end):
    return "".join(lines[start-1:end])

os.makedirs(output_dir, exist_ok=True)

# 1. mod.rs
mod_content = get_lines(1, 32) + """
pub mod checkpoint;
pub mod device_lost;
pub mod error_scope;
pub mod injector;
pub mod manager;
pub mod oom;
pub mod stats;
pub mod timeout;

pub use checkpoint::*;
pub use device_lost::*;
pub use error_scope::*;
pub use injector::*;
pub use manager::*;
pub use oom::*;
pub use stats::*;
pub use timeout::*;

use self::checkpoint::GpuCheckpoint;

""" + get_lines(122, 167)
with open(os.path.join(output_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.write(mod_content)

# 2. checkpoint.rs
ckpt_content = get_lines(47, 120)
with open(os.path.join(output_dir, "checkpoint.rs"), "w", encoding="utf-8") as f:
    f.write(ckpt_content)

# 3. injector.rs
inj_content = """use crate::core::error::{KwaversError, SystemError};
use std::sync::Mutex;
use super::GpuErrorType;

""" + get_lines(169, 256)
with open(os.path.join(output_dir, "injector.rs"), "w", encoding="utf-8") as f:
    f.write(inj_content)

# 4. stats.rs
stats_content = """pub(crate) const RECOVERY_SUCCESS_THRESHOLD: f64 = 0.90;

""" + get_lines(258, 336) + """
/// Global GPU recovery statistics (shared across strategies)
pub static GLOBAL_STATS: std::sync::LazyLock<std::sync::Mutex<GpuRecoveryStats>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(GpuRecoveryStats::default()));

""" + get_lines(340, 354).replace("fn update_avg_latency_us", "pub fn update_avg_latency_us") + """
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::{ErrorContext, KwaversError, SystemError};
    use crate::gpu::recovery::{DeviceLostRecovery, GpuCheckpoint, GpuRecoveryManager};
    use crate::core::error::recovery::RecoveryStrategy;
    use std::sync::{Arc, Mutex};

""" + get_lines(1134, 1154) + "\n}\n"
with open(os.path.join(output_dir, "stats.rs"), "w", encoding="utf-8") as f:
    f.write(stats_content)

# 5. device_lost.rs
dl_content = """use crate::core::error::recovery::{RecoveryResult, RecoveryStrategy};
use crate::core::error::{ErrorContext, KwaversError, KwaversResult, SystemError};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::info;

use super::{GpuCheckpoint, GpuRecoveryAction, GLOBAL_STATS, update_avg_latency_us};

""" + get_lines(356, 528) + """
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::recovery::GpuRecoveryManager;

""" + get_lines(1031, 1071) + "\n}\n"
with open(os.path.join(output_dir, "device_lost.rs"), "w", encoding="utf-8") as f:
    f.write(dl_content)

# 6. oom.rs
oom_content = """use crate::core::error::recovery::{RecoveryResult, RecoveryStrategy};
use crate::core::error::{ErrorContext, KwaversError, KwaversResult, SystemError};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::info;

use super::{GpuCheckpoint, GpuRecoveryAction, GLOBAL_STATS, update_avg_latency_us};

""" + get_lines(530, 701) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(1073, 1103) + "\n}\n"
with open(os.path.join(output_dir, "oom.rs"), "w", encoding="utf-8") as f:
    f.write(oom_content)

# 7. timeout.rs
timeout_content = """use crate::core::error::recovery::{RecoveryResult, RecoveryStrategy};
use crate::core::error::{ErrorContext, KwaversError, SystemError};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::info;

use super::{GpuRecoveryAction, GLOBAL_STATS, update_avg_latency_us};

""" + get_lines(703, 856) + """
#[cfg(test)]
mod tests {
    use super::*;

""" + get_lines(1105, 1115) + "\n}\n"
with open(os.path.join(output_dir, "timeout.rs"), "w", encoding="utf-8") as f:
    f.write(timeout_content)

# 8. manager.rs
mgr_content = """use crate::core::error::recovery::{RecoveryResult, RecoveryStrategy};
use crate::core::error::{ErrorContext, KwaversError};
use std::sync::{Arc, Mutex};
use tracing::warn;

use super::{
    DeviceLostRecovery, GpuCheckpoint, GpuOomRecovery, GpuRecoveryStats,
    TimeoutRecovery, GLOBAL_STATS, RECOVERY_SUCCESS_THRESHOLD,
};

""" + get_lines(858, 988) + """
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::SystemError;

""" + get_lines(1117, 1132) + "\n" + get_lines(1156, 1195) + "\n}\n"
with open(os.path.join(output_dir, "manager.rs"), "w", encoding="utf-8") as f:
    f.write(mgr_content)

# 9. error_scope.rs
err_content = """use crate::profiling::gpu_allocator::GpuError;

""" + get_lines(990, 1025)
with open(os.path.join(output_dir, "error_scope.rs"), "w", encoding="utf-8") as f:
    f.write(err_content)

print("Extraction completed!")
