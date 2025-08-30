#!/usr/bin/env python3
"""Add #[derive(Debug)] to all structs missing it."""

import re
import subprocess
import sys

def get_missing_debug_structs():
    """Get all structs missing Debug implementation."""
    result = subprocess.run(
        ["bash", "-c", "source /usr/local/cargo/env && cargo build --lib 2>&1"],
        capture_output=True,
        text=True
    )
    
    structs = []
    for line in result.stderr.split('\n'):
        if "does not implement `std::fmt::Debug`" in line:
            # Next lines contain the struct location
            continue
        match = re.search(r'-->\s+(src/[^:]+):(\d+)', line)
        if match:
            file_path = match.group(1)
            line_num = int(match.group(2))
            structs.append((file_path, line_num))
    
    return structs

def add_debug_derive(file_path, struct_line):
    """Add #[derive(Debug)] to a struct."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the struct declaration
    target_line = struct_line - 1  # Convert to 0-indexed
    
    # Check if already has derive
    if target_line > 0 and '#[derive' in lines[target_line - 1]:
        # Add Debug to existing derive
        lines[target_line - 1] = lines[target_line - 1].replace(')]', ', Debug)]')
        if ', Debug, Debug)]' in lines[target_line - 1]:
            lines[target_line - 1] = lines[target_line - 1].replace(', Debug, Debug)]', ', Debug)]')
    else:
        # Add new derive line
        indent = len(lines[target_line]) - len(lines[target_line].lstrip())
        lines.insert(target_line, ' ' * indent + '#[derive(Debug)]\n')
    
    with open(file_path, 'w') as f:
        f.writelines(lines)

def main():
    structs = get_missing_debug_structs()
    print(f"Found {len(structs)} structs missing Debug")
    
    # Group by file
    by_file = {}
    for file_path, line_num in structs:
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(line_num)
    
    # Process each file
    for file_path, line_nums in by_file.items():
        print(f"Processing {file_path}: {len(line_nums)} structs")
        # Sort in reverse to avoid line number shifts
        for line_num in sorted(line_nums, reverse=True):
            try:
                add_debug_derive(file_path, line_num)
            except Exception as e:
                print(f"  Error at line {line_num}: {e}")
    
    print("Done!")

if __name__ == "__main__":
    main()