use super::config::BatchFieldConfig;
use super::soa_buffer::SoAFieldBuffer;
use crate::error::KwaversResult;

/// Pre-allocated field batch for wave simulation (SoA layout).
///
/// Pre-allocates:
/// - Primary fields: pressure (0), velocity_x (1), velocity_y (2), velocity_z (3)
/// - Temporary fields: 2 scratch buffers
#[derive(Debug)]
pub struct BatchFieldHandle {
    pub primary: SoAFieldBuffer<f64>,
    pub temp: SoAFieldBuffer<f64>,
    config: BatchFieldConfig,
}

impl BatchFieldHandle {
    /// Create batch allocation for wave simulation.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn for_wave_simulation(nx: usize, ny: usize, nz: usize) -> KwaversResult<Self> {
        let primary_config = BatchFieldConfig::for_3d_fields(nx, ny, nz, 4);
        let temp_config = BatchFieldConfig::for_3d_fields(nx, ny, nz, 2);
        let primary = SoAFieldBuffer::new(primary_config)?;
        let temp = SoAFieldBuffer::new(temp_config)?;
        Ok(Self {
            primary,
            temp,
            config: primary_config,
        })
    }

    /// Get pressure field (index 0).
    #[inline]
    #[must_use]
    pub fn pressure(&self) -> &[f64] {
        self.primary.field(0)
    }

    /// Get mutable pressure field.
    #[inline]
    pub fn pressure_mut(&mut self) -> &mut [f64] {
        self.primary.field_mut(0)
    }

    /// Get velocity components (indices 1, 2, 3).
    #[must_use]
    pub fn velocity(&self) -> (&[f64], &[f64], &[f64]) {
        (
            self.primary.field(1),
            self.primary.field(2),
            self.primary.field(3),
        )
    }

    /// Get mutable velocity components.
    pub fn velocity_mut(&mut self) -> (&mut [f64], &mut [f64], &mut [f64]) {
        // SAFETY: Non-overlapping field indices guarantee no aliasing.
        let config = self.config;
        let ptr = self.primary.memory.as_ptr() as *mut f64;
        unsafe {
            let vx = std::slice::from_raw_parts_mut(
                ptr.add(config.field_elements),
                config.field_elements,
            );
            let vy = std::slice::from_raw_parts_mut(
                ptr.add(2 * config.field_elements),
                config.field_elements,
            );
            let vz = std::slice::from_raw_parts_mut(
                ptr.add(3 * config.field_elements),
                config.field_elements,
            );
            (vx, vy, vz)
        }
    }

    /// Get mutable temporary buffer 0.
    #[inline]
    pub fn temp_0_mut(&mut self) -> &mut [f64] {
        self.temp.field_mut(0)
    }

    /// Get mutable temporary buffer 1.
    #[inline]
    pub fn temp_1_mut(&mut self) -> &mut [f64] {
        self.temp.field_mut(1)
    }

    /// Total memory usage in bytes.
    #[must_use]
    pub fn total_memory(&self) -> usize {
        self.primary.memory_usage() + self.temp.memory_usage()
    }

    /// Zero all fields.
    pub fn clear(&mut self) {
        self.primary.fill(0.0);
        self.temp.fill(0.0);
    }

    /// Whether fields are cache-line aligned.
    #[must_use]
    pub fn is_cache_efficient(&self) -> bool {
        self.primary.is_cache_aligned() && self.temp.is_cache_aligned()
    }
}
