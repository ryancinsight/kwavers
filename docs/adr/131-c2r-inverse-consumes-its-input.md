# ADR 131: The complex-to-real inverse consumes its input spectrum

Status: Accepted

Board item: [KW-NATIVE-R2C-2026-09-11](../../backlog.md#kw-native-r2c)

## Context

`Fft3dInOutExt::inverse_c2r_into(&half_in, &mut out, &mut scratch)` is the PSTD
path's only way back from the `(nx, ny, nz/2+1)` half spectrum. Until #770 it
emulated a half-spectrum inverse on apollo's full complex plan and ignored
`scratch`. #770 routed it onto apollo's `RealFftData::inverse_3d_half_into`,
which transforms its spectrum in place; to keep `half_in` borrowed immutably,
the adapter copied it into `scratch` first and transformed the copy.

That copy was the remaining cost over the transform. With #770 in place,
`fft3d_baseline` (two rounds at 2–15% host load) read the r2c round trip below
the complex one at 32³ and 64³, but 21.3–22.1 µs against 20.6 µs at 16³, while
apollo's own real pair reads 19.8 µs per 16³ round trip. The copy of `half_in`
is the difference.

Every one of the 31 call sites — 26 in `kwavers-solver`, 3 in the
`kwavers-math` FFT tests, 2 in `fft3d_baseline` — passes an input spectrum that
is wholly overwritten before any later read: `grad_k`, `p_k`, and `ux_k` are
each rewritten by the next `forward_r2c_into` or gradient write, and their only
other reads are `.shape()`. No site relies on the contents of `scratch`.

## Decision

`inverse_c2r_into(&self, half_in: &mut Array3<Complex64>, out: &mut Array3<f64>)`
transforms `half_in` in place and overwrites it. The `scratch` argument is
removed, and every call site passes its spectrum by `&mut`. A strided `half_in`
or `out` stages through one contiguous copy, a cold path no solver site takes.

## Alternatives

- **Keep the copy into `scratch`.** Correct at every call site, but it pays one
  half-volume copy per inverse (37 KB at 16³, 2.1 MB at 64³) to protect inputs
  no caller reads again, and keeps an argument whose only role is receiving
  that copy.
- **An apollo inverse that reads an immutable spectrum**, fusing the copy into
  its first pass. It keeps the kwavers signature, but adds a second inverse
  entry point to apollo whose only consumer is this adapter, and it still writes
  a full half-volume workspace the caller supplies: the copy moves rather than
  disappears.

## Consequences

- Breaking for `kwavers-math` consumers: pass the spectrum by `&mut` and drop
  the scratch; a caller that still needs the spectrum copies it before the call
  (CHANGELOG, Unreleased).
- The spectrum passed in holds no meaning afterwards. The solver's buffers
  already hold none after each inverse; a future call site that reads its input
  spectrum after the inverse copies it first, as the trait documentation says.
- The 16³ wall-clock confirmation is open on the board item: every run this
  change was measured in ran at 49–100% host load.
