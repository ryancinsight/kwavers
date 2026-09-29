# ADR-134: IVUS imaging operations consume input records

Status: Accepted

## Context

The IVUS RF operation accepted ten positional arguments. The operation's
metrics and therapy operations have the same failure mode: related values are
easy to reorder at call sites, and adding a value changes every caller's
signature. The public Rust surface is used by the Python conversion boundary
and by first-party examples and tests.

## Decision

The RF and metrics operations receive operation-specific input records. The
therapy operations receive distinct input records that share the tissue-mask
and dose value objects both operations consume. The records borrow array
inputs and store scalar values by value, so callers retain ownership and the
computation performs no input copies.

The public signature change is breaking. In-repository callers migrate in the
same increment. No forwarding function or deprecated re-export is retained.
External callers construct the record named in the migration guide.

## Rejected alternatives

- Retaining the positional function would preserve an error-prone contract and
  force every future input addition through another argument list.
- Keeping a positional compatibility wrapper would duplicate the public
  contract and leave two call shapes to document and test.
- One shared record for all operations would mix bounded contexts and permit
  fields irrelevant to a given equation.

## Consequences

The input records make ownership and field meaning explicit and keep borrowed
arrays zero-copy. The migration requires a major-version classification for
the public function signatures. The affected operations retain their
value-semantic positive and invalid-input tests.

## Overturning evidence

If a shipped consumer requires independent ownership of every input or the
operation gains a distinct execution boundary, revise the record to the
smallest owned or boundary-specific type and update the migration contract.
