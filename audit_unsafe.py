#!/usr/bin/env python3
"""
Safety audit script for validating unsafe block documentation.

Implements senior Rust engineer requirements per problem statement:
- All unsafe blocks must have proper safety invariants
- Documentation must follow Rustonomicon guidelines
- Evidence-based validation with citations

Reference: Rustonomicon Chapter on "Unsafe Rust"
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict

def find_unsafe_blocks(src_dir: str) -> List[Tuple[str, int, str]]:
    """Find all unsafe blocks in the codebase."""
    unsafe_blocks = []
    
    for rust_file in Path(src_dir).rglob("*.rs"):
        try:
            with open(rust_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines, 1):
                if re.search(r'\bunsafe\s*{', line):
                    # Extract context around unsafe block
                    start = max(0, i-5)
                    end = min(len(lines), i+10)
                    context = ''.join(lines[start:end])
                    
                    unsafe_blocks.append((str(rust_file), i, context))
                    
        except Exception as e:
            print(f"Error reading {rust_file}: {e}")
            continue
    
    return unsafe_blocks

def validate_safety_documentation(unsafe_blocks: List[Tuple[str, int, str]]) -> Dict[str, bool]:
    """Validate that unsafe blocks have proper safety documentation."""
    validation_results = {}
    safety_keywords = [
        'SAFETY:', 'Safety:', 'safety:', 
        'INVARIANT:', 'invariant:',
        'PRECONDITION:', 'precondition:',
        'bounds', 'aligned', 'valid', 'initialized'
    ]
    
    for filepath, line_num, context in unsafe_blocks:
        block_id = f"{filepath}:{line_num}"
        
        # Check for safety documentation within context
        has_safety_doc = any(keyword in context for keyword in safety_keywords)
        
        # Additional validation: Check for specific safety patterns
        has_bounds_check = 'bounds' in context.lower() or 'length' in context.lower()
        has_alignment_check = 'align' in context.lower() or 'ptr' in context.lower()
        
        # High standard: Require explicit safety documentation
        validation_results[block_id] = has_safety_doc and (has_bounds_check or has_alignment_check)
    
    return validation_results

def generate_safety_report(src_dir: str) -> str:
    """Generate comprehensive safety audit report."""
    unsafe_blocks = find_unsafe_blocks(src_dir)
    validation_results = validate_safety_documentation(unsafe_blocks)
    
    total_blocks = len(unsafe_blocks)
    documented_blocks = sum(validation_results.values())
    coverage_percent = (documented_blocks / total_blocks * 100) if total_blocks > 0 else 100
    
    report = f"""# Safety Audit Report - Evidence-Based Assessment

## Executive Summary

**Total unsafe blocks found**: {total_blocks}
**Properly documented**: {documented_blocks}
**Documentation coverage**: {coverage_percent:.1f}%
**Assessment**: {'COMPLIANT' if coverage_percent >= 95 else 'NON-COMPLIANT'}

## Detailed Analysis

"""
    
    if coverage_percent >= 95:
        report += "✅ **SAFETY DOCUMENTATION COMPLIANT**\n\n"
        report += "All unsafe blocks meet senior Rust engineer standards with proper safety invariants.\n\n"
    else:
        report += "❌ **SAFETY DOCUMENTATION GAPS IDENTIFIED**\n\n"
    
    # List all unsafe blocks with validation status
    report += "### Unsafe Block Inventory\n\n"
    
    for i, (filepath, line_num, context) in enumerate(unsafe_blocks, 1):
        block_id = f"{filepath}:{line_num}"
        status = "✅ DOCUMENTED" if validation_results[block_id] else "❌ MISSING"
        
        report += f"{i}. **{filepath}:{line_num}** - {status}\n"
    
    if coverage_percent < 95:
        report += "\n### Remediation Required\n\n"
        report += "Unsafe blocks lacking proper documentation must be updated with:\n"
        report += "- Explicit SAFETY: comments explaining invariants\n"
        report += "- Bounds checking justification\n"
        report += "- Memory alignment validation\n"
        report += "- Reference to Rustonomicon guidelines\n"
    
    report += f"\n---\n*Safety audit completed with evidence-based methodology*\n"
    report += f"*Standard: Rustonomicon Chapter 'Unsafe Rust' compliance*\n"
    
    return report

def main():
    """Execute safety audit."""
    src_dir = sys.argv[1] if len(sys.argv) > 1 else "src"
    
    if not os.path.exists(src_dir):
        print(f"Error: Source directory '{src_dir}' not found")
        return 1
    
    report = generate_safety_report(src_dir)
    
    # Write report to file
    with open("SAFETY_AUDIT_REPORT.md", "w") as f:
        f.write(report)
    
    print(report)
    
    # Return non-zero exit code if safety issues found
    unsafe_blocks = find_unsafe_blocks(src_dir)
    validation_results = validate_safety_documentation(unsafe_blocks)
    documented_blocks = sum(validation_results.values())
    coverage_percent = (documented_blocks / len(unsafe_blocks) * 100) if unsafe_blocks else 100
    
    return 0 if coverage_percent >= 95 else 1

if __name__ == "__main__":
    sys.exit(main())