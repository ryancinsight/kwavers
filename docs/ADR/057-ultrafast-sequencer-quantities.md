# ADR 057: Type Ultrafast Sequencer Physical Metrics

- Status: Accepted
- Date: 2026-07-24
- Scope: `kwavers-transducer::ultrafast::sequencer`

## Context

Ultrafast transmission schedules exposed time, angle, pulse-repetition rate,
frame rate, sound speed, and imaging depth as `f64` values. Unit-bearing names
and comments did not prevent a caller from supplying a value in another unit or
from confusing an angle with an unrelated dimensionless scalar.

## Decision

Use Aequitas `Time`, `Angle`, `Frequency`, `Velocity`, and `Length` for the
sequencer configuration and schedule outputs. Convert to base SI scalars only
inside the existing PRF and event-timing arithmetic. Event and transducer
indices remain `usize` because they are structural counts, not physical
metrics. Input angles are canonical radians through Aequitas `Radian`.

## Alternatives rejected

- Keep raw scalars with `_s`, `_rad`, `_hz`, or `_m` names: rejected because
  names do not enforce unit or dimension correctness.
- Type only the schedule outputs: rejected because the constructor and PRF
  override would remain untyped physical boundaries.
- Add a scalar compatibility facade: rejected because all in-tree callers are
  migrated on this branch and the old signatures are superseded.

## Verification

The sequencer tests preserve the analytical PRF bound `c/(2z_max)`, event
spacing `1/PRF`, sequential/interleaved ordering, flash/STA zero-angle
contracts, and compounded frame-rate relation. Full Cargo verification is
pending restoration of the peer Coeus path required by the workspace graph.
