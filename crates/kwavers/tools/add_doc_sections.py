#!/usr/bin/env python3
"""
add_doc_sections.py — Add `# Errors` and `# Panics` Rustdoc sections
to public functions that are missing them, based on static body analysis.

Strategy:
- # Errors: inspect function body for `Err(KwaversError::Variant(...))`  patterns
            and generate a precise per-variant bullet list.
- # Panics: inspect function body for `.expect("msg")` and `panic!("msg")`
            calls and generate a precise bullet list from the messages.

Operates in-place on every .rs file under src/.
"""

import re
import sys
from pathlib import Path

# ── Pattern library ──────────────────────────────────────────────────────────

# ANY_FN_RE: matches any fn declaration (pub or non-pub trait methods).
# Covers: fn, async fn, unsafe fn, pub fn, pub unsafe fn, pub async fn, etc.
ANY_FN_RE = re.compile(
    r'^\s*(?:pub(?:\([\w:]+\))?\s+)?(?:(?:async|unsafe)\s+)*fn\s+(\w+)'
)

# PUB_FN_RE: matches only public fn declarations.
# Used when there is NO doc comment — generate a minimal stub only for public API.
PUB_FN_RE = re.compile(
    r'^\s*pub(?:\([\w:]+\))?\s+(?:(?:async|unsafe)\s+)*fn\s+(\w+)'
)

# Result return type: bare Result, KwaversResult, or any path-qualified variant.
# Matches: `-> Result<`, `-> KwaversResult<`, `-> crate::...::KwaversResult<`, etc.
RETURNS_RESULT_RE = re.compile(r'\)\s*->\s*(?:[\w:]+::)?(?:Kwavers)?Result\b')

# Error variant in a return expression.
ERR_VARIANT_RE = re.compile(
    r'Err\s*\(\s*KwaversError\s*::\s*(\w+)',
)
QUESTION_ERR_RE = re.compile(r'\?')  # propagated errors (generic)

# Panic sites.
EXPECT_RE   = re.compile(r'\.expect\s*\(\s*"([^"\n]{0,120})"')
PANIC_RE    = re.compile(r'\bpanic!\s*\(\s*"([^"\n]{0,120})"')
UNWRAP_RE   = re.compile(r'\.unwrap\s*\(\s*\)')
ASSERT_RE   = re.compile(r'\bassert(?:_eq|_ne)?!\s*\([^,)]+,\s*"([^"\n]{0,120})"')

# Already-documented sections.
HAS_ERRORS_RE = re.compile(r'#\s+Errors\b')
HAS_PANICS_RE = re.compile(r'#\s+Panics\b')

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_doc_block_and_fn(lines: list[str], fn_idx: int) -> tuple[int, int]:
    """
    Given the index of a `pub fn` line, walk backwards to find the contiguous
    `///` doc-comment block (skipping blank lines and `#[…]` attributes).
    Returns (doc_start_idx, doc_end_idx) inclusive, or (-1, -1) if none.
    """
    i = fn_idx - 1
    doc_end = -1
    doc_start = -1
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith('///'):
            if doc_end == -1:
                doc_end = i
            doc_start = i
        elif stripped.startswith('#[') or stripped == '':
            pass  # skip attributes and blank lines between doc and fn
        else:
            break
        i -= 1
    return doc_start, doc_end


def extract_fn_body(lines: list[str], fn_idx: int) -> str:
    """
    Extract the body of the function starting at fn_idx.
    Uses brace counting; stops when the opening brace's matching close is found.
    Returns the body as a single string.
    """
    depth = 0
    body_lines = []
    started = False
    for line in lines[fn_idx:]:
        for ch in line:
            if ch == '{':
                depth += 1
                started = True
            elif ch == '}':
                depth -= 1
        body_lines.append(line)
        if started and depth == 0:
            break
    return ''.join(body_lines)


def build_errors_section(body: str, indent: str) -> list[str]:
    """
    Produce `/// # Errors` lines based on error variants found in body.
    """
    variants = sorted(set(ERR_VARIANT_RE.findall(body)))
    has_propagation = bool(QUESTION_ERR_RE.search(body))

    lines = [f'{indent}/// # Errors\n']
    if variants:
        for v in variants:
            lines.append(f'{indent}/// - Returns [`KwaversError::{v}`] if the precondition for {_variant_description(v)} is violated.\n')
    if has_propagation and not variants:
        lines.append(f'{indent}/// - Propagates any [`KwaversError`] returned by called functions.\n')
    elif has_propagation:
        lines.append(f'{indent}/// - Propagates any [`KwaversError`] returned by called functions.\n')
    if not variants and not has_propagation:
        lines.append(f'{indent}/// - Returns [`Err`] if an internal constraint is violated.\n')
    lines.append(f'{indent}///\n')
    return lines


def _variant_description(variant: str) -> str:
    """Map known KwaversError variant names to brief English descriptions."""
    table = {
        'InvalidInput':        'invalid or out-of-range input parameters',
        'InvalidState':        'an invalid internal state transition',
        'DimensionMismatch':   'mismatched array or grid dimensions',
        'NumericalInstability':'numerical instability in the solver',
        'IoError':             'an I/O operation',
        'NotImplemented':      'an unimplemented code path',
        'ConfigError':         'invalid configuration values',
        'MediumError':         'medium property constraints',
        'SolverError':         'a solver-level failure',
        'BoundaryError':       'boundary condition constraints',
        'SourceError':         'source field constraints',
        'SensorError':         'sensor configuration constraints',
        'PhysicsError':        'physics parameter constraints',
        'MathError':           'a mathematical operation (e.g., singular matrix)',
        'ConvergenceError':    'convergence failure',
        'AllocationError':     'memory allocation failure',
    }
    return table.get(variant, f'a {variant}-class constraint')


def build_panics_section(body: str, indent: str) -> list[str]:
    """
    Produce `/// # Panics` lines based on panic sites in body.
    Covers: .expect(), panic!(), assert!(), assert_eq!(), assert_ne!(), .unwrap()
    """
    expects = EXPECT_RE.findall(body)
    panics  = PANIC_RE.findall(body)
    asserts = ASSERT_RE.findall(body)
    unwraps = UNWRAP_RE.findall(body)

    def _sanitize(msg: str) -> str:
        """Strip trailing backslash (line continuation) and format specifiers."""
        msg = msg.rstrip().rstrip('\\').strip()
        # Truncate at 80 chars for readability.
        if len(msg) > 80:
            msg = msg[:77] + '...'
        return msg

    lines = [f'{indent}/// # Panics\n']
    seen: set[str] = set()
    for msg in expects:
        msg = _sanitize(msg)
        key = msg[:80]
        if key not in seen:
            seen.add(key)
            lines.append(f'{indent}/// - Panics if `{msg}`.\n')
    for msg in panics:
        msg = _sanitize(msg)
        key = msg[:80]
        if key not in seen:
            seen.add(key)
            lines.append(f'{indent}/// - Panics with `"{msg}"`.\n')
    for msg in asserts:
        msg = _sanitize(msg)
        key = msg[:80]
        if key not in seen:
            seen.add(key)
            lines.append(f'{indent}/// - Panics if assertion fails: `{msg}`.\n')
    if unwraps and not expects and not panics and not asserts:
        lines.append(f'{indent}/// - Panics if an internal invariant assumed to hold at this call site is violated.\n')
    if not expects and not panics and not asserts and not unwraps:
        lines.append(f'{indent}/// - Panics if an internal precondition is violated.\n')
    lines.append(f'{indent}///\n')
    return lines


def doc_indent(lines: list[str], doc_start: int) -> str:
    """Return the leading whitespace of the first doc line."""
    return re.match(r'^(\s*)', lines[doc_start]).group(1)


# ── Per-file processor ────────────────────────────────────────────────────────

def process_file(path: Path, dry_run: bool = False) -> int:
    """
    Mutate path in-place, inserting missing `# Errors` / `# Panics` sections.
    Returns the number of functions updated.
    """
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)

    # We collect insertions as (line_index_to_insert_before, new_lines).
    insertions: list[tuple[int, list[str]]] = []

    i = 0
    while i < len(lines):
        # Match any fn (including non-pub trait methods that already have docs).
        m_any = ANY_FN_RE.match(lines[i])
        if not m_any:
            i += 1
            continue

        fn_line = lines[i]
        # Check if signature (possibly multi-line) returns a Result.
        # Collect up to 8 lines of signature.
        sig = ''.join(lines[i:i+20])
        returns_result = bool(RETURNS_RESULT_RE.search(sig))

        doc_start, doc_end = extract_doc_block_and_fn(lines, i)

        body = extract_fn_body(lines, i)
        body_has_panics = (
            '.expect(' in body or
            'panic!(' in body or
            '.unwrap()' in body or
            'assert!(' in body or
            'assert_eq!(' in body or
            'assert_ne!(' in body or
            'unreachable!(' in body or
            'todo!(' in body or
            'unimplemented!(' in body
        )

        if doc_start == -1:
            # No doc comment. Only generate a minimal stub for `pub fn`
            # (trait-method declarations and private fns don't need auto-stubs).
            m_pub = PUB_FN_RE.match(lines[i])
            if not m_pub:
                i += 1
                continue

            # Find insertion point: scan back past attributes only (not blank lines).
            insert_at = i
            j = i - 1
            while j >= 0:
                stripped = lines[j].strip()
                if stripped.startswith('#['):
                    insert_at = j
                    j -= 1
                else:
                    break

            indent = re.match(r'^(\s*)', lines[i]).group(1)
            fn_name = m_pub.group(1)
            # Convert snake_case to a brief sentence.
            brief = fn_name.replace('_', ' ')
            new_lines: list[str] = [f'{indent}/// {brief.capitalize()}.\n']
            if returns_result:
                new_lines.extend(build_errors_section(body, indent))
            if body_has_panics:
                new_lines.extend(build_panics_section(body, indent))

            if len(new_lines) > 1:  # only insert if we added at least one section
                insertions.append((insert_at, new_lines))
            i += 1
            continue

        doc_text = ''.join(lines[doc_start:doc_end+1])
        indent = doc_indent(lines, doc_start)

        # Insertion point = immediately after the last doc line,
        # before the first attribute or fn keyword.
        insert_at = doc_end + 1

        new_lines = []

        if returns_result and not HAS_ERRORS_RE.search(doc_text):
            new_lines.extend(build_errors_section(body, indent))

        if body_has_panics and not HAS_PANICS_RE.search(doc_text):
            new_lines.extend(build_panics_section(body, indent))

        if new_lines:
            insertions.append((insert_at, new_lines))

        i += 1

    if not insertions:
        return 0

    # Apply insertions in reverse order so indices stay valid.
    for insert_at, new_lines in sorted(insertions, reverse=True):
        lines[insert_at:insert_at] = new_lines

    if not dry_run:
        path.write_text(''.join(lines), encoding='utf-8')
    return len(insertions)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    src_root = Path(__file__).parent.parent / 'src'
    if not src_root.exists():
        print(f'ERROR: src root not found at {src_root}', file=sys.stderr)
        sys.exit(1)

    dry_run = '--dry-run' in sys.argv
    total_fns = 0
    total_files = 0

    for rs_file in sorted(src_root.rglob('*.rs')):
        count = process_file(rs_file, dry_run=dry_run)
        if count:
            total_fns += count
            total_files += 1
            print(f'  {count:3d}  {rs_file.relative_to(src_root)}')

    print(f'\nDone. {total_fns} functions updated across {total_files} files.')


if __name__ == '__main__':
    main()
