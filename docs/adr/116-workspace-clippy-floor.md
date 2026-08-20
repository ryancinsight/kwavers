# 116. One workspace clippy floor, and which of the two candidates it is

- Status: Accepted
- Date: 2026-08-20
- Item: `backlog.md#kw-lint-112`

## Context

`main` has no `[workspace.lints.clippy]` table. Every member builds at clippy's
default lint set, so `pedantic` has never been enforced anywhere in the
workspace. Twenty-one of twenty-four members already declare
`[lints] workspace = true` (the remaining three land in
[#431](https://github.com/ryancinsight/kwavers/pull/431)) — the inheritance
plumbing is live and the table it points at is empty.

Two candidate tables were authored independently, neither aware of the other.

**A — `origin/codex/kwavers-floatelement-roots`.** `pedantic` at `deny` with no
`clippy::all` group, all four hygiene lints (`unwrap_used`, `print_stdout`,
`print_stderr`, `dbg_macro`) at `deny`, and 74 `allow` entries in six
categorised sections carrying counts split total-vs-production:

| category | sites | in production |
| --- | ---: | ---: |
| Documentation | 5628 | 4396 |
| Numeric conversion | 9460 | 5666 |
| Signature and shape | 1365 | 1269 |
| Style | 3095 | 1517 |
| Raw-pointer / layout | 49 | 43 |

**B — [#423](https://github.com/ryancinsight/kwavers/pull/423).** The canonical
Atlas template (`all` + `pedantic` at `warn`, `priority = -1`, apollo's allow
set verbatim), plus a debt block of 29 lints counted at 1390 total.

The two numbers measure different things and neither is wrong. A's ~19,600 is
the debt above *bare* pedantic. B's 1390 is the residual above the *Atlas
template floor* — the template's allow set already covers A's two largest
categories (`doc_markdown` 4905, `cast_precision_loss` 5745). Comparing 1390
against 19,600 as if they ranked the same quantity is the trap here.

Three facts decide it rather than the table designs alone.

**A never landed and is stranded.** Its PR (#364) merged on 13 August; the floor
table is on the branch but not in that PR, and five days later it is still not
on `main`. Its `unwrap_used = "deny"` requires per-site
`#![expect(clippy::unwrap_used, reason = "ratchet KWAVERS-UNWRAP-1")]` across
4523 existing sites. A change of that size is the most likely reason it stalled.

**B is landable now.** All three CI clippy invocations report zero warnings
against it, including the `-p kwavers --features pinn --lib` job, which lints
every path member and therefore the whole workspace.

**`KW-LINT-112` mischaracterises A.** It describes the branch as carrying
"`clippy::all` + `clippy::pedantic` at `warn` … `print_stdout` and `dbg_macro`
at `warn` rather than `deny`", citing the ~4,000-`unwrap()`/244-production
figures. That is a description of B. A is `pedantic = "deny"` with all four
hygiene lints denied. The instruction "either land that branch or adopt its
table verbatim" was therefore written against the wrong artifact.

## Decision

**Land B as the floor, and treat A as the target state rather than a rival.**

1. B's table is the floor, because a floor that is enforced today beats a
   stricter one that has not been enforceable for five days. It also keeps the
   Atlas template as the single upstream definition (apollo is SSOT), which A's
   hand-rolled categorisation does not.

2. **Rescue A's measurements into B's debt block.** A's production-vs-test split
   is strictly better data than B's totals: it separates debt that ships from
   debt that only exists in tests, which is what decides burn-down order. The
   counts are recovered into the scratchpad and folded into the debt block as
   each category is worked; the branch is not a durable home for them.

3. **A's levels are the ratchet target, recorded as such.** `pedantic` moves
   `warn` → `deny` and the hygiene four move to `deny` as their counts reach
   zero. That is the same direction A chose, reached incrementally instead of in
   one 4523-site commit.

4. **One item, one ID.** `KW-LINT-1` (filed on #423) and `KW-LINT-112` are the
   same work. `KW-LINT-112` is the surviving ID — it matches the existing
   `KW-LINT-0xx`/`1xx` series, whereas `KW-LINT-1` collides with it by reading
   as its ancestor. The mischaracterisation in its body is corrected in the same
   change.

## Alternatives rejected

**Land A instead.** It is the stricter design and, on the merits of the table
alone, the better one. Rejected because adopting it means either landing 4523
per-site `#![expect]` attributes in one change — untestable as a unit, and a
merge conflict against every open PR — or setting `unwrap_used = "deny"` while
4523 sites violate it, which does not compile. Its levels are adopted as the
ratchet target instead, which reaches the same end state without that step.

**Merge the two tables into a third.** This is what `KW-LINT-112` warns against
("do not author a second table"), and the warning is right: three definitions is
worse than two. B *is* the merge — the Atlas template with A's measurement
discipline folded into its debt block.

**Wait, and land neither until the debt is burned down.** The debt is 19,600
sites above bare pedantic. Nothing gets enforced for as long as that takes, and
new code keeps adding to it. A ratchet baseline exists precisely so enforcement
starts before the burn-down finishes.

## Consequences

The workspace gets an enforced floor now, at `warn`, with new code in the two
CI-gated crates held to it immediately — those crates trip none of the debt
lints and are clean on the merits rather than by the baseline. Burn-down runs
per lint under `KW-LINT-112`: drive a count to zero in production code, delete
its line, and the floor tightens. `pedantic = "deny"` and the hygiene four
follow as the counts clear.

`origin/codex/kwavers-floatelement-roots` can be pruned once its counts are
folded in; its only unlanded content is the table, and the table is superseded
by this decision. Its rescued form lives in the scratchpad until then.
