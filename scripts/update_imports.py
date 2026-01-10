#!/usr/bin/env python3
"""
Import path updater for Rust module migrations.
Updates all occurrences of old import paths to new paths.

Usage:
    python3 update_imports.py <old_path> <new_path>

Example:
    python3 update_imports.py domain/core/error core/error
"""

import sys
import re
from pathlib import Path
from typing import Tuple, List

def convert_path_to_module(path: str) -> str:
    """Convert filesystem path to Rust module path."""
    return path.replace('/', '::')

def find_rust_files(root: Path = Path("src")) -> List[Path]:
    """Find all Rust source files."""
    return list(root.rglob("*.rs"))

def update_imports_in_file(
    file_path: Path,
    old_module: str,
    new_module: str
) -> Tuple[bool, int]:
    """
    Update import statements in a single file.
    Returns (changed, count) tuple.
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"⚠️  Warning: Could not read {file_path}: {e}")
        return False, 0

    original = content
    count = 0

    # Pattern 1: use crate::old::module::*
    pattern1 = rf'(use\s+crate::){re.escape(old_module)}(::|\s|;)'
    replacement1 = rf'\g<1>{new_module}\g<2>'
    content, n1 = re.subn(pattern1, replacement1, content)
    count += n1

    # Pattern 2: use super::old::module (less common but possible)
    # Only if the old module is at root level
    if '::' not in old_module:
        pattern1b = rf'(use\s+super::){re.escape(old_module)}(::|\s|;)'
        replacement1b = rf'\g<1>{new_module}\g<2>'
        content, n1b = re.subn(pattern1b, replacement1b, content)
        count += n1b

    # Pattern 3: crate::old::module in other contexts (type annotations, etc.)
    pattern2 = rf'(crate::){re.escape(old_module)}(::)'
    replacement2 = rf'\g<1>{new_module}\g<2>'
    content, n2 = re.subn(pattern2, replacement2, content)
    count += n2

    # Pattern 4: pub use crate::old::module
    pattern3 = rf'(pub\s+use\s+crate::){re.escape(old_module)}(::|\s|;)'
    replacement3 = rf'\g<1>{new_module}\g<2>'
    content, n3 = re.subn(pattern3, replacement3, content)
    count += n3

    # Pattern 5: pub(crate) use crate::old::module
    pattern4 = rf'(pub\(crate\)\s+use\s+crate::){re.escape(old_module)}(::|\s|;)'
    replacement4 = rf'\g<1>{new_module}\g<2>'
    content, n4 = re.subn(pattern4, replacement4, content)
    count += n4

    # Pattern 6: crate::old::module at end of line or before closing delimiter
    pattern5 = rf'(crate::){re.escape(old_module)}(\s|;|\)|\]|\}|,)'
    replacement5 = rf'\g<1>{new_module}\g<2>'
    content, n5 = re.subn(pattern5, replacement5, content)
    count += n5

    changed = content != original

    if changed:
        try:
            file_path.write_text(content, encoding='utf-8')
        except Exception as e:
            print(f"❌ Error: Could not write {file_path}: {e}")
            return False, 0

    return changed, count

def main():
    if len(sys.argv) != 3:
        print("Usage: update_imports.py <old_path> <new_path>")
        print("Example: update_imports.py domain/core/error core/error")
        sys.exit(1)

    old_path = sys.argv[1]
    new_path = sys.argv[2]

    old_module = convert_path_to_module(old_path)
    new_module = convert_path_to_module(new_path)

    print(f"🔍 Searching for imports of: {old_module}")
    print(f"📝 Replacing with: {new_module}")
    print()

    rust_files = find_rust_files()
    total_files_changed = 0
    total_changes = 0
    changed_files = []

    for file_path in rust_files:
        changed, count = update_imports_in_file(file_path, old_module, new_module)
        if changed:
            total_files_changed += 1
            total_changes += count
            changed_files.append((file_path, count))
            print(f"  ✓ {file_path.relative_to('src')}: {count} changes")

    print()
    print(f"✅ Updated {total_changes} imports in {total_files_changed} files")

    if total_files_changed > 0:
        print()
        print("📋 Summary of changes:")
        for file_path, count in sorted(changed_files, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {file_path.relative_to('src')}: {count} changes")
        if len(changed_files) > 10:
            print(f"  ... and {len(changed_files) - 10} more files")

    # Verification step
    print()
    print("🔍 Verifying no old imports remain...")
    remaining = 0
    for file_path in rust_files:
        content = file_path.read_text(encoding='utf-8')
        # Look for old module path (but not in comments)
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith('//'):
                continue
            if old_module in line and 'crate::' in line:
                remaining += 1
                if remaining <= 5:  # Show first 5 occurrences
                    print(f"  ⚠️  {file_path.relative_to('src')}:{line_num}: {line.strip()[:80]}")

    if remaining > 0:
        print(f"⚠️  Warning: Found {remaining} potential remaining references")
        print("   (May include comments or false positives)")
    else:
        print("✅ No old import paths detected")

if __name__ == "__main__":
    main()
