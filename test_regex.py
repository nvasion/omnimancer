#!/usr/bin/env python3

import re

# Test the exact line that should match
line = "- VERDICT: PASS tail line"

print("Testing line:", repr(line))

# Try various patterns
patterns = [
    r"VERDICT:?\s*(PASS|FAIL)",
    r"VERDICT:?\s*(PASS|FAIL)",
    r"[Vv][Ee][Rr][Dd][Ii][Cc]:?\s*(PASS|FAIL)",
    r".*?(VERDICT:?\s*(PASS|FAIL)).*?"
]

for i, pattern in enumerate(patterns):
    match = re.search(pattern, line, re.IGNORECASE)
    print(f"Pattern {i+1}: {pattern}")
    if match:
        print(f"  Match found: {match.groups()}")
        if len(match.groups()) >= 2:
            print(f"  Group 2: {repr(match.group(2))}")
    else:
        print("  No match")

# Let's manually test what the regex should match
print("\nManual inspection:")
print("Line contains 'VERDICT:'?", "VERDICT:" in line.upper())
print("Line matches pattern:", re.search(r"VERDICT:?\s*(PASS|FAIL)", line, re.IGNORECASE))