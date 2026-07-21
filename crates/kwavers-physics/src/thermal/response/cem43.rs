//! Failure-atomic CEM43 increments over consumer-owned fields.

use aequitas::systems::si::quantities::{ThermodynamicTemperature, Time};
use asclepius::response::thermal::Cem43;
use kwavers_core::{
    constants::{numerical::SECONDS_PER_MINUTE, thermodynamic::KELVIN_OFFSET_C},
    error::{KwaversError, KwaversResult},
};
use leto::{ArrayView3, ArrayViewMut3};
use std::sync::Mutex;

use crate::parallel::zip_mut_ref;

pub(crate) trait StoredTemperatureScale: Send + Sync {
    fn absolute(value: f64) -> ThermodynamicTemperature<f64>;
}

pub(crate) struct CelsiusStorage;

impl StoredTemperatureScale for CelsiusStorage {
    #[inline]
    fn absolute(value: f64) -> ThermodynamicTemperature<f64> {
        ThermodynamicTemperature::from_base(value + KELVIN_OFFSET_C)
    }
}

pub(crate) struct KelvinStorage;

impl StoredTemperatureScale for KelvinStorage {
    #[inline]
    fn absolute(value: f64) -> ThermodynamicTemperature<f64> {
        ThermodynamicTemperature::from_base(value)
    }
}

const _: () = assert!(core::mem::size_of::<CelsiusStorage>() == 0);
const _: () = assert!(core::mem::size_of::<KelvinStorage>() == 0);

pub(crate) fn checked_cem43_increments<S, F>(
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
    zip_mut_ref(increments, temperature, |increment, &stored| {
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
