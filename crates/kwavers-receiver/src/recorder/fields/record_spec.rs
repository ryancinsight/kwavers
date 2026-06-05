//! [`SensorRecordSpec`]: HashSet-backed specification of quantities to record.

use std::collections::HashSet;

use super::record_field::SensorRecordField;

/// A set of sensor record fields specifying what quantities to record.
///
/// Provides helper methods for querying whether specific categories of data
/// need to be accumulated during the time loop.
#[derive(Debug, Clone)]
pub struct SensorRecordSpec {
    fields: HashSet<SensorRecordField>,
}

impl SensorRecordSpec {
    /// Create a spec recording only pressure time series (k-Wave default).
    #[must_use]
    pub fn pressure_only() -> Self {
        let mut fields = HashSet::new();
        fields.insert(SensorRecordField::Pressure);
        Self { fields }
    }

    /// Create a spec from a slice of fields.
    #[must_use]
    pub fn from_fields(fields: &[SensorRecordField]) -> Self {
        Self {
            fields: fields.iter().copied().collect(),
        }
    }

    /// Add a field to the spec.
    pub fn add(&mut self, field: SensorRecordField) {
        self.fields.insert(field);
    }

    /// Returns `true` if the spec contains `field`.
    #[must_use]
    pub fn contains(&self, field: SensorRecordField) -> bool {
        self.fields.contains(&field)
    }

    /// Returns `true` if any velocity time-series recording is requested.
    #[must_use]
    pub fn needs_velocity_time_series(&self) -> bool {
        self.fields.iter().any(|f| f.needs_velocity_time_series())
    }

    /// Returns `true` if any velocity statistics accumulation is requested.
    #[must_use]
    pub fn needs_velocity_stats(&self) -> bool {
        self.fields.iter().any(|f| f.needs_velocity_stats())
    }

    /// Returns `true` if ux statistics accumulation is requested.
    #[must_use]
    pub fn needs_ux_stats(&self) -> bool {
        self.fields.contains(&SensorRecordField::VelocityMaxX)
            || self.fields.contains(&SensorRecordField::VelocityMinX)
            || self.fields.contains(&SensorRecordField::VelocityRmsX)
    }

    /// Returns `true` if uy statistics accumulation is requested.
    #[must_use]
    pub fn needs_uy_stats(&self) -> bool {
        self.fields.contains(&SensorRecordField::VelocityMaxY)
            || self.fields.contains(&SensorRecordField::VelocityMinY)
            || self.fields.contains(&SensorRecordField::VelocityRmsY)
    }

    /// Returns `true` if uz statistics accumulation is requested.
    #[must_use]
    pub fn needs_uz_stats(&self) -> bool {
        self.fields.contains(&SensorRecordField::VelocityMaxZ)
            || self.fields.contains(&SensorRecordField::VelocityMinZ)
            || self.fields.contains(&SensorRecordField::VelocityRmsZ)
    }

    /// Returns `true` if any pressure statistics accumulation is requested.
    #[must_use]
    pub fn needs_pressure_stats(&self) -> bool {
        self.fields.iter().any(|f| f.needs_pressure_stats())
    }

    /// Returns `true` if any acoustic-intensity quantity is requested.
    #[must_use]
    pub fn needs_intensity(&self) -> bool {
        self.records_intensity_x() || self.records_intensity_y() || self.records_intensity_z()
    }

    /// Returns `true` if pressure time-series recording is requested.
    #[must_use]
    pub fn records_pressure(&self) -> bool {
        self.fields.contains(&SensorRecordField::Pressure) || self.needs_intensity()
    }

    /// Returns `true` if ux time-series storage is requested.
    ///
    /// Acoustic intensity uses instantaneous ux samples during
    /// `record_velocity_step`; it does not require storing a separate ux
    /// time-series buffer unless `VelocityX` or `VelocityNonStaggeredX` is also
    /// requested.
    #[must_use]
    pub fn records_ux(&self) -> bool {
        self.fields.contains(&SensorRecordField::VelocityX)
            || self
                .fields
                .contains(&SensorRecordField::VelocityNonStaggeredX)
    }

    /// Returns `true` if uy time-series storage is requested.
    #[must_use]
    pub fn records_uy(&self) -> bool {
        self.fields.contains(&SensorRecordField::VelocityY)
            || self
                .fields
                .contains(&SensorRecordField::VelocityNonStaggeredY)
    }

    /// Returns `true` if uz time-series storage is requested.
    #[must_use]
    pub fn records_uz(&self) -> bool {
        self.fields.contains(&SensorRecordField::VelocityZ)
            || self
                .fields
                .contains(&SensorRecordField::VelocityNonStaggeredZ)
    }

    /// Returns `true` if x-intensity time-series or average is requested.
    #[must_use]
    pub fn records_intensity_x(&self) -> bool {
        self.fields.contains(&SensorRecordField::IntensityX)
            || self.fields.contains(&SensorRecordField::IntensityAvgX)
    }

    /// Returns `true` if y-intensity time-series or average is requested.
    #[must_use]
    pub fn records_intensity_y(&self) -> bool {
        self.fields.contains(&SensorRecordField::IntensityY)
            || self.fields.contains(&SensorRecordField::IntensityAvgY)
    }

    /// Returns `true` if z-intensity time-series or average is requested.
    #[must_use]
    pub fn records_intensity_z(&self) -> bool {
        self.fields.contains(&SensorRecordField::IntensityZ)
            || self.fields.contains(&SensorRecordField::IntensityAvgZ)
    }

    /// Returns `true` if any velocity component is needed at all.
    #[must_use]
    pub fn needs_any_velocity(&self) -> bool {
        self.needs_velocity_time_series() || self.needs_velocity_stats() || self.needs_intensity()
    }

    /// Returns an iterator over all fields in the spec.
    pub fn iter(&self) -> impl Iterator<Item = &SensorRecordField> {
        self.fields.iter()
    }
}

impl Default for SensorRecordSpec {
    fn default() -> Self {
        Self::pressure_only()
    }
}
