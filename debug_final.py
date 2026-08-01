#!/usr/bin/env python3

import tempfile
from pathlib import Path
from omnimancer.tui.fleet.sources import AgentsLogParser

# Debug the failing test case specifically
log_content = """## Session: 2026-08-01 03:10
### Spawned: aabbccdd - 03:11
### Complete: aabbccdd
### Died: 11223344 - crash
### Synthesis - 03:30
- VERDICT: PASS tail line"""

print("Testing the exact failing case:")
print("Content:", repr(log_content))

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    log_file = tmp_path / "agents.log"
    log_file.write_text(log_content)
    
    # Read and split manually to see what happens
    content = log_file.read_text()
    lines = content.split('\n')
    print(f"\nSplit into {len(lines)} lines:")
    for i, line in enumerate(lines):
        print(f"  {i}: {repr(line)}")
    
    # Now test the parser
    p = AgentsLogParser(log_file)
    entries = p.poll()
    print(f"\nParser returned {len(entries)} entries:")
    for i, entry in enumerate(entries):
        print(f"  {i+1}: {entry}")
        
    print(f"\nExpected 6 entries, got {len(entries)}")