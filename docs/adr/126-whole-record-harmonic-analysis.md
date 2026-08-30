# 126. Define harmonic detection as whole-record analysis

- Status: Accepted
- Date: 2026-08-30
- Item: `KW-HARMONIC-CONFIG-CONTRACT-2026-08-30` in `backlog.md`

## Context

`HarmonicDetectionConfig` exposed six public fields, but only
`fundamental_frequency` and `n_harmonics` reached computation. The other four
had been definition-only since the type was introduced:

- `fft_window_size` did not select the transform size;
- `fft_overlap` did not segment or overlap records;
- `min_snr_db` did not filter any result; and
- `enable_phase_unwrapping` did not alter returned phases.

The implementation already had one coherent contract: the supplied time extent
was multiplied by a symmetric Hann window and transformed once per spatial
point. It returned principal complex phases and reported SNR without filtering.
Leaving contradictory controls public made callers believe that behavior could
be changed when it could not.

## Decision

Harmonic detection analyzes one complete caller-supplied time record at every
spatial point. The time extent is the FFT size. The returned time vector has the
same length, and the frequency vector contains the `N / 2 + 1` non-negative FFT
bin frequencies.

The detector applies one symmetric Hann window and one FFT to each record. It
returns principal phases in `[-π, π]`. SNR is descriptive output: every higher
harmonic is returned, and its SNR uses the established ±10-bin neighborhood in
the full normalized `N`-bin spectrum. The requested `n_harmonics` includes the
fundamental; the result stores the remaining `n_harmonics - 1` components in its
higher-harmonic arrays.

Remove `fft_window_size`, `fft_overlap`, `min_snr_db`, and
`enable_phase_unwrapping` from `HarmonicDetectionConfig`. Also remove the inert
configuration argument from
`HarmonicDisplacementField::compute_nonlinearity_parameter`.

## Migration

| Removed surface | Migration |
| --- | --- |
| `fft_window_size` | Supply a record with the desired time extent. |
| `fft_overlap` | Segment the source into overlapping records explicitly and analyze each record. |
| `min_snr_db` | Filter the returned `harmonic_snrs` using the caller's threshold. |
| `enable_phase_unwrapping` | Unwrap returned principal phases downstream along the domain-selected spatial or temporal axis. |
| `compute_nonlinearity_parameter(&config)` | Call `compute_nonlinearity_parameter()`; the ratio never depended on detector configuration. |

## Alternatives rejected

**Implement all four controls.** Window size and overlap require a segmented
output dimension and a policy for combining segments. Phase unwrapping requires
an axis and continuity model. An SNR threshold requires a missing-value or
rejection representation. The existing booleans and scalars do not specify
those contracts, and no in-repository caller configures them.

**Keep and document them as ignored.** An ignored public control remains a
wrong-result hazard. It also preserves two competing descriptions of the
operation: caller-selected segmentation versus the actual whole-record FFT.

**Deprecate before removal.** The workspace has no active caller of the four
fields. A compatibility surface would retain the defect without preserving any
behavior.

## Consequences

This is a public breaking change and requires migration for external struct
literals and direct nonlinearity calls. Valid numerical output, FFT allocation
behavior, and the existing Criterion workload remain unchanged. Tests pin even
and odd record metadata, full-spectrum SNR near Nyquist, principal phase range,
and dense-versus-strided complete-result equivalence.
