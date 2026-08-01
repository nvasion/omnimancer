#!/usr/bin/env python3

import re

# Test the actual patterns used in the implementation
lines_to_test = [
    "## Session: 2026-08-01 03:10",
    "### Spawned: aabbccdd - 03:11", 
    "### Complete: aabbccdd",
    "### Died: 11223344 - crash",
    "### Synthesis - 03:30",
    "- VERDICT: PASS tail line",
    "VERDICT: FAIL"
]

patterns = {
    "session": r"^## Session:(.*)",
    "spawned": r"^### Spawned:\s*([0-9a-f]{8})",
    "complete": r"^### Complete:\s*([0-9a-f]{8})",
    "died": r"^### Died:\s*([0-9a-f]{8})",
    "synthesis": r"^### Synthesis",
    "verdict": r"VERDICT:?\s*(PASS|FAIL)"
}

for i, line in enumerate(lines_to_test):
    print(f"\nLine {i+1}: {repr(line)}")
    for name, pattern in patterns.items():
        match = re.match(pattern, line)
        if match and name != "verdict":
            print(f"  {name} pattern match: {match.groups()}")
        elif name == "verdict":
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                print(f"  {name} pattern match: {match.groups()}")