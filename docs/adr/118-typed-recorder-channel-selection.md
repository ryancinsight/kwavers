# 118. Typed recorder channel selection

- Status: Accepted
- Date: 2026-08-21
- Item: `backlog.md#kw-lint-rec-01`
- Revision: retroactive record of the implementation in `refactor/kwavers-lint-struct-bools`

## Context

`RecorderConfig` and `Recorder` each carried five independent channel booleans. The shape made
invalid combinations easy to construct, triggered `struct_excessive_bools`, and exposed boolean
arguments in builder methods. The channels are one domain choice that should have one owner.

## Decision

Represent channel selection with the zero-allocation `RecordingChannels` bitset. Selectors use the
closed `RecorderChannel` enum, and enable/disable transitions use `RecordingState`. Both the
configuration and runtime recorder store the same channel set; all recording branches query it.
Builder methods accept `RecordingState` rather than bare booleans.

## Alternatives rejected

- Keep five booleans and add per-site lint suppressions: preserves boolean blindness and leaves two
  divergent representations.
- Use a heap-backed set or dynamic trait: adds allocation or dispatch to a per-step recording path
  without a present requirement.

## Migration

Callers replace `true`/`false` channel arguments with `RecordingState::Enabled` or
`RecordingState::Disabled`, and struct literals replace the five fields with `channels`. This is a
public contract change; the branch is the migration boundary and no compatibility shim remains.

## Verification

Receiver warning-denied Clippy, Nextest (48/48), doctests (1 passed), Rustdoc, workspace
warning-denied Clippy, and the exact production `struct_excessive_bools` scan pass. The scan falls
from 14 to 12 sites; the remaining sites are outside the receiver bounded context.
