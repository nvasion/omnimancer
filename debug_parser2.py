#!/usr/bin/env python3

import tempfile
from pathlib import Path
from omnimancer.tui.fleet.sources import AgentsLogParser

# Replicate the exact test case
log_content = """## Session: 2026-08-01 03:10
### Spawned: aabbccdd - 03:11
### Complete: aabbccdd
### Died: 11223344 - crash
### Synthesis - 03:30
- VERDICT: PASS tail line"""

print("Original content:")
print(repr(log_content))

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    log_file = tmp_path / "agents.log"
    log_file.write_text(log_content)
    
    # Read it back to make sure it's stored correctly
    content_read = log_file.read_text()
    print("\nContent read back:")
    print(repr(content_read))
    
    # Check each line individually
    lines = content_read.split('\n')
    print(f"\nNumber of lines: {len(lines)}")
    for i, line in enumerate(lines):
        print(f"Line {i}: {repr(line)}")
        # Test if this line would match
        import re
        match = re.search(r"VERDICT:?\s*(PASS|FAIL)", line, re.IGNORECASE)
        if match:
            print(f"  -> Would match: {match.group(1)}")
        else:
            print(f"  -> No match")
    
    p = AgentsLogParser(log_file)
    entries = p.poll()
    
    print(f"\nNumber of entries found: {len(entries)}")
    for i, entry in enumerate(entries):
        print(f"{i+1}: {entry}")