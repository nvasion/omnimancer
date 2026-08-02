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

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    log_file = tmp_path / "agents.log"
    log_file.write_text(log_content)
    
    p = AgentsLogParser(log_file)
    entries = p.poll()
    
    print("Number of entries found:", len(entries))
    for i, entry in enumerate(entries):
        print(f"{i+1}: {entry}")
        
    # Print the raw content
    print("\nRaw content:")
    print(repr(log_content))