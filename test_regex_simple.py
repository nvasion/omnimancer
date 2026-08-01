#!/usr/bin/env python3

import re

# Test the exact regex pattern used in the implementation
test_lines = [
    "VERDICT: FAIL",
    "VERDICT: PASS",
    "- VERDICT: PASS tail line"
]

for line in test_lines:
    print(f"Testing line: {repr(line)}")
    
    # This is the pattern from my implementation
    match = re.search(r"VERDICT:?\s*(PASS|FAIL)", line, re.IGNORECASE)
    print(f"  Regex match: {match}")
    if match:
        print(f"    Groups: {match.groups()}")
        print(f"    Group 1: {repr(match.group(1))}")
    print()