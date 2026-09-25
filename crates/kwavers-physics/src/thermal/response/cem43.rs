//! Failure-atomic CEM43 increments and reusable accumulators.
//!
//! The CEM43 law itself is owned by `asclepius::response::thermal::Cem43`; this
//! module holds the kwavers-side scaffolding that the therapy path shares: the
//! stored-temperature scale adapters, the checked per-voxel increment kernel,
//! and [`Cem43Accumulator`] — a single generic accumulator that owns the
//! running dose, a pre-allocated increment scratch field (so the per-timestep
//! update never allocates), an optional bounded temperature history for
//! interval integration, and the running maximum dose.

use aequitas::systems::si::quantities::{ThermodynamicTemperature, Time};
use asclepius::response::thermal::Cem43;
use kwavers_core::{
    constants::{numerical::SECONDS_PER_MINUTE, thermodynamic::KELVIN_OFFSET_C},
    error::{KwaversError, KwaversResult},
};
use leto::{Array3, ArrayView3, ArrayViewMut3};
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Mutex;

use kwavers_core::traversal::zip_mut;

/// Maps a stored field sample to an absolute thermodynamic temperature.
///
/// Implementations are zero-sized, so an accumulator can carry its scale choice
/// in the type without changing its memory layout.
pub trait StoredTemperatureScale: Send + Sync {
    /// Interpret `value` (in the scale's own units) as an absolute temperature.
    fn absolute(value: f64) -> ThermodynamicTemperature<f64>;
}

/// [`StoredTemperatureScale`] for fields stored in degrees Celsius.
#[derive(Debug, Clone, Copy)]
pub struct CelsiusStorage;

impl StoredTemperatureScale for CelsiusStorage {
    #[inline]
    fn absolute(value: f64) -> ThermodynamicTemperature<f64> {
        ThermodynamicTemperature::from_base(value + KELVIN_OFFSET_C)
    }
}

/// [`StoredTemperatureScale`] for fields stored in kelvin.
#[derive(Debug, Clone, Copy)]
pub struct KelvinStorage;

impl StoredTemperatureScale for KelvinStorage {
    #[inline]
    fn absolute(value: f64) -> ThermodynamicTemperature<f64> {
        ThermodynamicTemperature::from_base(value)
    }
}

const _: () = assert!(core::mem::size_of::<CelsiusStorage>() == 0);
const _: () = assert!(core::mem::size_of::<KelvinStorage>() == 0);

/// Fill `increments` with the failure-atomic CEM43 contribution of `temperature`.
///
/// `include` masks the voxels that are allowed to accumulate dose; masked
/// voxels are written as `0.0`. The scratch field is written in place, so a
/// steady-state caller allocates nothing per timestep.
///
/// # Errors
///
/// Returns [`KwaversError::InvalidInput`] when `increments` and `temperature`
/// have different shapes, or when the Asclepius law rejects an observation
/// (for example a non-finite absolute temperature). On rejection the dose
/// tracked by the caller is left untouched.
///
/// # Panics
///
/// Panics only if the internal response-failure lock is poisoned; the lock is
/// never held across a panic.
pub fn checked_cem43_increments<S, F>(
    increments: ArrayViewMut3<'_, f64>,
    temperature: ArrayView3<'_, f64>,
    step: Time<f64>,
    include: F,
) -> KwaversResult<()>
where
    S: StoredTemperatureScale,
    F: Fn(f64) -> bool + Send + Sync,
{
    if increments.shape() != temperature.shape() {
        return Err(KwaversError::InvalidInput(format!(
            "CEM43 increment shape {:?} does not match temperature shape {:?}",
            increments.shape(),
            temperature.shape()
        )));
    }

    let law = Cem43::<f64>::canonical();
    let failure = Mutex::new(None);
    zip_mut(increments, temperature, |increment, &stored| {
        match law.increment(S::absolute(stored), step) {
            Ok(exposure) => {
                *increment = if include(stored) {
                    exposure.get().into_base() / SECONDS_PER_MINUTE
                } else {
                    0.0
                };
            }
            Err(source) => {
                *increment = 0.0;
                let mut first = failure
                    .lock()
                    .expect("invariant: response failure lock is never held across a panic");
                if first.is_none() {
                    *first = Some(source);
                }
            }
        }
    });

    if let Some(source) = failure
        .into_inner()
        .map_err(|_| KwaversError::ConcurrencyError {
            message: "thermal response failure lock was poisoned".to_string(),
        })?
    {
        return Err(KwaversError::InvalidInput(format!(
            "CEM43 update rejected an observation: {source}"
        )));
    }
    Ok(())
}

/// Absolute Celsius temperature at which the canonical CEM43 law is referenced.
///
/// This is the `43 °C` normalisation point of Sapareto & Dewey (1984).
#[must_use]
pub fn cem43_reference_celsius() -> f64 {
    Cem43::<f64>::canonical().reference().into_base() - KELVIN_OFFSET_C
}

/// Instantaneous CEM43 rate at a Celsius temperature.
///
/// The returned value is the equivalent-minutes-per-minute rate: `1.0` at the
/// 43 °C reference, doubling per degree above and quartering per degree below.
///
/// # Errors
///
/// Returns [`KwaversError::InvalidInput`] when the Asclepius law rejects the
/// absolute temperature.
pub fn cem43_rate_per_minute(temperature_c: f64) -> KwaversResult<f64> {
    Cem43::<f64>::canonical()
        .rate(ThermodynamicTemperature::from_base(
            temperature_c + KELVIN_OFFSET_C,
        ))
        .map(|rate| rate.into_base())
        .map_err(|source| {
            KwaversError::InvalidInput(format!("CEM43 rate observation is invalid: {source}"))
        })
}

/// CEM43 equivalent minutes accumulated over `duration_s` at `temperature_c`.
///
/// # Errors
///
/// Returns [`KwaversError::InvalidInput`] when the Asclepius law rejects the
/// absolute temperature or time step.
pub fn cem43_equivalent_minutes(temperature_c: f64, duration_s: f64) -> KwaversResult<f64> {
    Cem43::<f64>::canonical()
        .increment(
            ThermodynamicTemperature::from_base(temperature_c + KELVIN_OFFSET_C),
            Time::from_base(duration_s),
        )
        .map(|exposure| exposure.get().into_base() / SECONDS_PER_MINUTE)
        .map_err(|source| {
            KwaversError::InvalidInput(format!("CEM43 increment observation is invalid: {source}"))
        })
}

/// Reusable CEM43 accumulator over a dense 3-D field.
///
/// `S` selects the stored temperature scale and `N` is the number of
/// temperature samples retained for interval (midpoint) integration; Euler
/// accumulators use `N = 0`. The accumulator pre-allocates its increment
/// scratch field once, so neither update path allocates per timestep.
#[derive(Debug, Clone)]
pub struct Cem43Accumulator<S: StoredTemperatureScale, const N: usize> {
    dose: Array3<f64>,
    increments: Array3<f64>,
    history: VecDeque<Array3<f64>>,
    history_times: VecDeque<f64>,
    max_dose: f64,
    max_time: Time<f64>,
    _scale: PhantomData<S>,
}

impl<S: StoredTemperatureScale, const N: usize> Cem43Accumulator<S, N> {
    /// Allocate an accumulator over a dense field of `shape`.
    #[must_use]
    pub fn new(shape: [usize; 3]) -> Self {
        Self {
            dose: Array3::zeros(shape),
            increments: Array3::zeros(shape),
            history: VecDeque::with_capacity(N),
            history_times: VecDeque::with_capacity(N),
            max_dose: 0.0,
            max_time: Time::from_base(0.0),
            _scale: PhantomData,
        }
    }

    /// Shape of the accumulated dose field.
    #[must_use]
    pub fn shape(&self) -> [usize; 3] {
        self.dose.shape()
    }

    /// Cumulative CEM43 dose field (equivalent minutes).
    #[must_use]
    pub fn dose(&self) -> &Array3<f64> {
        &self.dose
    }

    /// Dose at grid location `(i, j, k)`.
    #[must_use]
    pub fn dose_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.dose[[i, j, k]]
    }

    /// The most recent per-voxel CEM43 increments.
    #[must_use]
    pub fn increments(&self) -> &Array3<f64> {
        &self.increments
    }

    /// Peak dose observed so far.
    #[must_use]
    pub fn max_dose(&self) -> f64 {
        self.max_dose
    }

    /// Simulation time at which the peak dose was last raised.
    #[must_use]
    pub fn max_time(&self) -> Time<f64> {
        self.max_time
    }

    /// Fraction of voxels whose accumulated dose strictly exceeds `threshold`.
    #[must_use]
    pub fn fraction_above(&self, threshold: f64) -> f64 {
        let count = self.dose.iter().filter(|&&dose| dose > threshold).count();
        count as f64 / self.dose.len() as f64
    }

    /// Clear the accumulated dose, history and running maximum.
    pub fn reset(&mut self) {
        self.dose.fill(0.0);
        self.increments.fill(0.0);
        self.history.clear();
        self.history_times.clear();
        self.max_dose = 0.0;
        self.max_time = Time::from_base(0.0);
    }

    /// Accumulate one Euler CEM43 step from a stored `temperature` field.
    ///
    /// `include` masks the voxels allowed to accumulate. When `now` is
    /// supplied the running maximum dose (and the time it was reached) is
    /// tracked.
    ///
    /// # Errors
    ///
    /// Propagates the rejection returned by [`checked_cem43_increments`]; the
    /// committed dose and running maximum are unchanged on failure.
    pub fn accumulate<F>(
        &mut self,
        temperature: &Array3<f64>,
        dt: Time<f64>,
        include: F,
        now: Option<Time<f64>>,
    ) -> KwaversResult<()>
    where
        F: Fn(f64) -> bool + Send + Sync,
    {
        checked_cem43_increments::<S, _>(
            self.increments.view_mut(),
            temperature.view(),
            dt,
            include,
        )?;
        zip_mut(
            self.dose.view_mut(),
            self.increments.view(),
            |dose, &increment| *dose += increment,
        );
        if let Some(now) = now {
            let updated_max = self.dose.iter().copied().fold(0.0_f64, f64::max);
            if updated_max > self.max_dose {
                self.max_dose = updated_max;
                self.max_time = now;
            }
        }
        Ok(())
    }

    /// Accumulate one interval (trapezoidal) CEM43 step.
    ///
    /// The first observation only validates and records `temperature`. Each
    /// later observation integrates the midpoint between the retained previous
    /// sample and `temperature` over `time_s - previous_time_s` seconds.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::DimensionMismatch`] when `temperature` does not
    /// match the dose shape, or [`KwaversError::InvalidInput`] when the
    /// Asclepius law rejects an observation. On rejection the committed dose
    /// and retained history are unchanged.
    ///
    /// # Panics
    ///
    /// Panics only if a dose or temperature array violates the dense-storage
    /// invariant required by the zero-copy accumulation path.
    pub fn accumulate_interval(
        &mut self,
        temperature: Array3<f64>,
        time_s: f64,
    ) -> KwaversResult<()> {
        if temperature.shape() != self.dose.shape() {
            return Err(KwaversError::DimensionMismatch(format!(
                "CEM43 interval temperature shape {:?} does not match dose shape {:?}",
                temperature.shape(),
                self.dose.shape()
            )));
        }

        let law = Cem43::<f64>::canonical();
        let Some(previous) = self.history.back() else {
            // First observation: validate only, then remember for the interval.
            for &stored in temperature.iter() {
                law.rate(S::absolute(stored)).map_err(|source| {
                    KwaversError::InvalidInput(format!(
                        "CEM43 interval observation is invalid: {source}"
                    ))
                })?;
            }
            self.remember(temperature, time_s);
            return Ok(());
        };

        let previous_time_s = *self
            .history_times
            .back()
            .expect("invariant: retained history and times stay in step");
        let step = Time::from_base(time_s - previous_time_s);
        {
            let previous = previous
                .as_slice()
                .expect("invariant: retained temperature history is dense");
            let current = temperature
                .as_slice()
                .expect("invariant: interval temperature measurement is dense");
            let increments = self
                .increments
                .as_slice_mut()
                .expect("invariant: CEM43 increment field is dense");

            for ((increment, &previous_stored), &current_stored) in
                increments.iter_mut().zip(previous).zip(current)
            {
                let average = previous_stored.midpoint(current_stored);
                *increment = law
                    .increment(S::absolute(average), step)
                    .map_err(|source| {
                        KwaversError::InvalidInput(format!(
                            "CEM43 interval observation is invalid: {source}"
                        ))
                    })?
                    .get()
                    .into_base()
                    / SECONDS_PER_MINUTE;
            }
        }

        zip_mut(
            self.dose.view_mut(),
            self.increments.view(),
            |dose, &increment| *dose += increment,
        );
        self.remember(temperature, time_s);
        Ok(())
    }

    /// Push `temperature` onto the retained history, evicting the oldest
    /// samples so at most `N` are kept.
    fn remember(&mut self, temperature: Array3<f64>, time_s: f64) {
        self.history.push_back(temperature);
        self.history_times.push_back(time_s);
        while self.history.len() > N {
            self.history.pop_front();
            self.history_times.pop_front();
        }
    }
}
