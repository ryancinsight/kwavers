#!/usr/bin/env python3
"""
Script to fix validation error format changes in the codebase.
Converts old format with field/value/min/max strings to new format with numeric values.
"""

import os
import re
import sys

def fix_out_of_range_errors(content):
    """Fix OutOfRange validation errors to use the new format."""
    
    # Pattern for OutOfRange with field, value, min, max as strings
    pattern = r'''ValidationError::OutOfRange\s*\{\s*
        field:\s*"([^"]+)"\.to_string\(\),?\s*
        value:\s*([^,]+?)(?:\.to_string\(\))?,?\s*
        min:\s*([^,]+?)(?:\.to_string\(\))?,?\s*
        max:\s*([^,}]+?)(?:\.to_string\(\))?,?\s*
    \}'''
    
    def replace_match(match):
        field = match.group(1)
        value = match.group(2).strip()
        min_val = match.group(3).strip()
        max_val = match.group(4).strip()
        
        # Clean up the values
        value = value.replace('.to_string()', '').strip(',')
        min_val = min_val.replace('.to_string()', '').replace('"', '').strip(',')
        max_val = max_val.replace('.to_string()', '').replace('"', '').strip(',')
        
        # Convert string literals to numeric
        if min_val == '"0.0"' or min_val == '0.0':
            min_val = '0.0'
        if max_val == '"inf"':
            max_val = 'f64::INFINITY'
            
        # Build the replacement - add comment with field name for clarity
        return f'''ValidationError::OutOfRange {{
                value: {value},
                min: {min_val},
                max: {max_val},
            }} /* field: {field} */'''
    
    # Apply the fix
    content = re.sub(pattern, replace_match, content, flags=re.MULTILINE | re.DOTALL | re.VERBOSE)
    
    return content

def fix_too_small_errors(content):
    """Fix TooSmall grid errors to use the correct format."""
    
    # Pattern for TooSmall with reason field
    pattern = r'''GridError::TooSmall\s*\{\s*
        (?:nx:\s*(\w+),?\s*)?
        (?:ny:\s*(\w+),?\s*)?
        (?:nz:\s*(\w+),?\s*)?
        (?:spatial_order:\s*(\w+),?\s*)?
        (?:min:\s*(\w+),?\s*)?
        reason:\s*format!\([^)]+\),?\s*
    \}'''
    
    def replace_match(match):
        nx = match.group(1) or 'grid.nx'
        ny = match.group(2) or 'grid.ny'
        nz = match.group(3) or 'grid.nz'
        min_val = match.group(5) or '2'
        
        return f'''GridError::TooSmall {{
                nx: {nx},
                ny: {ny},
                nz: {nz},
                min: {min_val},
            }}'''
    
    content = re.sub(pattern, replace_match, content, flags=re.MULTILINE | re.DOTALL | re.VERBOSE)
    
    return content

def process_file(filepath):
    """Process a single Rust file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original = content
        content = fix_out_of_range_errors(content)
        content = fix_too_small_errors(content)
        
        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    return False

def main():
    """Main function to process all Rust files."""
    fixed_count = 0
    
    for root, dirs, files in os.walk('/workspace/src'):
        for file in files:
            if file.endswith('.rs'):
                filepath = os.path.join(root, file)
                if process_file(filepath):
                    fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()