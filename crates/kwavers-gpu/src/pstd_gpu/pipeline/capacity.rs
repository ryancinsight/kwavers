//! Capacity check of `pml_xyz` and `shifts_all`, two buffers of a GPU PSTD plan.
//!
//! `wgpu` rejects a buffer larger than `max_buffer_size` when it is created
//! and a storage binding larger than `max_storage_buffer_binding_size` when
//! its bind group is created. The provider makes both through calls that
//! return no error, so either rejection panics in the uncaptured-error
//! handler. `pml_xyz` and `shifts_all` are checked before any buffer exists.

use kwavers_core::error::{KwaversError, KwaversResult};

/// Bytes of one `f32`, the element type of `pml_xyz` and `shifts_all`.
const ELEMENT_BYTES: u64 = 4;
const _: () = assert!(std::mem::size_of::<f32>() == 4);

/// The two device limits a storage buffer of the plan must fit, in bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::pstd_gpu) struct StorageBufferLimits {
    /// Largest buffer the device creates.
    pub(in crate::pstd_gpu) max_buffer_size: u64,
    /// Largest storage binding the device accepts in a bind group.
    pub(in crate::pstd_gpu) max_storage_buffer_binding_size: u64,
}

impl StorageBufferLimits {
    /// Read the limits `device` enables.
    pub(in crate::pstd_gpu) fn enabled_on(device: &wgpu::Device) -> Self {
        let limits = device.limits();
        Self {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        }
    }
}

/// Element counts of `pml_xyz` and `shifts_all`, with `None` for a count beyond `usize`.
///
/// `pml_xyz` packs the three PML arrays, three values per grid point; every
/// field, medium, k-space and absorption buffer holds one value per point.
/// `shifts_all` holds four values per axis sample and exceeds `pml_xyz` on
/// some grids, every `N`x1x1 grid among them. Not planned here: the workspace
/// of the FFT plan, which Hephaestus sizes, and the buffers a run sizes.
fn checked_buffers(shape: [usize; 3]) -> [(&'static str, Option<usize>); 2] {
    let [nx, ny, nz] = shape;
    let points = nx.checked_mul(ny).and_then(|xy| xy.checked_mul(nz));
    let axis_samples = nx.checked_add(ny).and_then(|xy| xy.checked_add(nz));
    [
        ("pml_xyz", points.and_then(|points| points.checked_mul(3))),
        (
            "shifts_all",
            axis_samples.and_then(|samples| samples.checked_mul(4)),
        ),
    ]
}

/// Refuse a plan whose `pml_xyz` or `shifts_all` exceeds a limit the device enables.
///
/// # Errors
///
/// [`KwaversError::ResourceLimitExceeded`] naming the buffer, the bytes it
/// needs, and the limit it exceeds; the same variant when the element or
/// byte count of a buffer overflows.
pub(in crate::pstd_gpu) fn check_plan_capacity(
    shape: [usize; 3],
    limits: StorageBufferLimits,
) -> KwaversResult<()> {
    let [nx, ny, nz] = shape;
    for (buffer, elements) in checked_buffers(shape) {
        let bytes = elements
            .and_then(|elements| u64::try_from(elements).ok())
            .and_then(|elements| elements.checked_mul(ELEMENT_BYTES))
            .ok_or_else(|| KwaversError::ResourceLimitExceeded {
                message: format!(
                    "GPU PSTD buffer `{buffer}` of a {nx}x{ny}x{nz} grid has an element or byte count that overflows"
                ),
            })?;
        for (limit, limit_bytes) in [
            ("max_buffer_size", limits.max_buffer_size),
            (
                "max_storage_buffer_binding_size",
                limits.max_storage_buffer_binding_size,
            ),
        ] {
            if bytes > limit_bytes {
                return Err(KwaversError::ResourceLimitExceeded {
                    message: format!(
                        "GPU PSTD buffer `{buffer}` needs {bytes} bytes; the device enables {limit} = {limit_bytes} bytes"
                    ),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check_plan_capacity, StorageBufferLimits};
    use kwavers_core::error::{KwaversError, KwaversResult};

    /// `pml_xyz` of a 16x16x16 grid: three `f32` values for each of 4096 points.
    const PACKED_PML_BYTES: u64 = 3 * 4096 * 4;

    fn limits(max_buffer_size: u64, max_storage_buffer_binding_size: u64) -> StorageBufferLimits {
        StorageBufferLimits {
            max_buffer_size,
            max_storage_buffer_binding_size,
        }
    }

    fn refusal(result: KwaversResult<()>) -> String {
        match result {
            Err(KwaversError::ResourceLimitExceeded { message }) => message,
            other => panic!("expected ResourceLimitExceeded, got {other:?}"),
        }
    }

    #[test]
    fn a_plan_that_meets_both_limits_exactly_is_admitted() {
        let result = check_plan_capacity([16, 16, 16], limits(PACKED_PML_BYTES, PACKED_PML_BYTES));

        assert!(matches!(result, Ok(())), "{result:?}");
    }

    #[test]
    fn one_byte_under_the_packed_pml_buffer_names_the_exceeded_limit() {
        let under = PACKED_PML_BYTES - 1;

        assert_eq!(
            refusal(check_plan_capacity([16, 16, 16], limits(under, PACKED_PML_BYTES))),
            "GPU PSTD buffer `pml_xyz` needs 49152 bytes; the device enables max_buffer_size = 49151 bytes"
        );
        assert_eq!(
            refusal(check_plan_capacity([16, 16, 16], limits(PACKED_PML_BYTES, under))),
            "GPU PSTD buffer `pml_xyz` needs 49152 bytes; the device enables max_storage_buffer_binding_size = 49151 bytes"
        );
    }

    /// A 1x1x2 grid packs 3 * 2 PML values (24 bytes) and holds
    /// 4 * (1 + 1 + 2) shift values (64 bytes).
    #[test]
    fn the_shift_buffer_governs_a_grid_of_few_points() {
        let admitted = check_plan_capacity([1, 1, 2], limits(64, 64));
        assert!(matches!(admitted, Ok(())), "{admitted:?}");

        assert_eq!(
            refusal(check_plan_capacity([1, 1, 2], limits(63, 64))),
            "GPU PSTD buffer `shifts_all` needs 64 bytes; the device enables max_buffer_size = 63 bytes"
        );
    }

    #[test]
    fn a_point_count_beyond_usize_is_refused() {
        let result = check_plan_capacity([usize::MAX, 2, 1], limits(u64::MAX, u64::MAX));

        assert_eq!(
            refusal(result),
            format!(
                "GPU PSTD buffer `pml_xyz` of a {}x2x1 grid has an element or byte count that overflows",
                usize::MAX
            )
        );
    }
}
