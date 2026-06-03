#!/usr/bin/env python3
"""
fix_doc_blank_lines.py — Remove blank lines that appear between the end of a
`///` doc-comment block and the next non-blank, non-doc line (fn, struct, #[attr]).

This fixes `empty_line_after_doc_comments` lint instances introduced when the
add_doc_sections.py script inserted doc blocks immediately before a blank line
that preceded the `pub fn` declaration.
"""

import re
import sys
from pathlib import Path


def fix_file(path: Path, dry_run: bool = False) -> int:
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)

    new_lines: list[str] = []
    changed = 0
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # Is this a doc comment line?
        if stripped.startswith('///'):
            # Look ahead: collect the entire doc block
            doc_block_end = i
            j = i
            while j < len(lines) and lines[j].strip().startswith('///'):
                doc_block_end = j
                j += 1
            # j now points to the first non-doc line after the block.
            # Skip any blank lines between doc block end and the next item.
            k = j
            while k < len(lines) and lines[k].strip() == '':
                k += 1
            # If there were blank lines and what follows is not another doc
            # block (i.e., it's an attribute or fn declaration), remove them.
            if k > j and k < len(lines):
                next_stripped = lines[k].strip()
                follows_code = (
                    next_stripped.startswith('#[') or
                    re.match(r'(pub(\([\w:]+\))?\s+)?(async\s+)?fn\b', next_stripped) or
                    re.match(r'(pub(\([\w:]+\))?\s+)?(struct|enum|trait|impl|type|const|static|mod)\b', next_stripped)
                )
                if follows_code:
                    # Output doc block lines as-is, then skip the blank lines.
                    new_lines.extend(lines[i:j])
                    changed += k - j  # number of blank lines removed
                    i = k
                    continue
        new_lines.append(lines[i])
        i += 1

    if changed and not dry_run:
        path.write_text(''.join(new_lines), encoding='utf-8')
    return changed


def main() -> None:
    src_root = Path(__file__).parent.parent / 'src'
    dry_run = '--dry-run' in sys.argv
    total_removed = 0
    total_files = 0
    for rs_file in sorted(src_root.rglob('*.rs')):
        removed = fix_file(rs_file, dry_run=dry_run)
        if removed:
            total_removed += removed
            total_files += 1
            print(f'  {removed:3d}  {rs_file.relative_to(src_root)}')
    print(f'\nDone. Removed {total_removed} blank lines across {total_files} files.')


if __name__ == '__main__':
    main()
