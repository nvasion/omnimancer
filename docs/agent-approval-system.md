# Omnimancer Agent Approval System

The Omnimancer Agent Approval System provides secure, user-controlled file modification workflows with comprehensive visual feedback and risk assessment. This system ensures that all file operations require explicit user consent while providing clear visibility into what changes will be made.

## Overview

The approval system consists of several key components working together to provide a safe and intuitive file modification experience:

- **File Content Display**: Rich visualization of file contents with syntax highlighting
- **Diff Rendering**: Side-by-side, unified, and inline diff views for modifications
- **Risk Assessment**: Automatic evaluation of operation risk levels
- **User Approval Flow**: Interactive prompts with clear approval/denial options
- **Workflow Management**: State tracking and operation orchestration

## Key Features

### 🔒 **Security First**
- **Explicit Consent Required**: No file modifications without user approval
- **Risk Assessment**: Operations categorized by potential impact (Low, Medium, High, Critical)
- **Visual Confirmation**: Clear display of exactly what will change
- **Audit Trail**: Complete logging of user decisions and operations

### 🎨 **Rich Visual Interface**
- **Syntax Highlighting**: Automatic language detection and code highlighting
- **Multiple Diff Views**: Choose between unified, side-by-side, or inline diffs
- **Color-Coded Changes**: Visual distinction between additions, deletions, and modifications
- **File Statistics**: Display file size, line counts, and metadata

### ⚡ **Intuitive Interaction**
- **Keyboard Shortcuts**: Quick approval with Y/N, detailed view with D
- **Batch Operations**: Handle multiple files with selective or bulk approval
- **Smart Defaults**: Safe fallbacks for timeouts and edge cases
- **Graceful Cancellation**: Easy to cancel or modify operations

## Getting Started

### Basic Usage

When an AI agent attempts to modify files, the approval system automatically activates:

```bash
# Example: Agent wants to create a new file
>>> Please create a Python script to analyze CSV data

🔍 File Operation Approval Required

┌─────────────────────────────────────────────────────────┐
│ 📄 Creating New File                                   │
│ Path: /home/user/data_analyzer.py                      │
│ Risk Level: Low                                         │
│ Size: 1,247 bytes (45 lines)                          │
└─────────────────────────────────────────────────────────┘

[Y] Approve  [N] Deny  [D] View Details  [Q] Quit
Your choice: 
```

### Interactive Options

| Key | Action | Description |
|-----|--------|-------------|
| `Y` or `Enter` | **Approve** | Execute the operation |
| `N` or `Esc` | **Deny** | Cancel the operation |
| `D` | **Details** | View full file content or diff |
| `A` | **Approve & Remember** | Approve and skip similar future prompts |
| `Q` | **Quit** | Cancel all pending operations |

### File Operation Types

#### 1. **File Creation**
When creating new files, you'll see:
- Complete file content with syntax highlighting
- File statistics (size, lines, encoding)
- Risk assessment based on file type and content

#### 2. **File Modification**
For existing file changes, you'll see:
- Side-by-side comparison of old and new content
- Highlighted changes with line numbers
- Summary of additions, deletions, and modifications

#### 3. **File Deletion**
When deleting files, you'll see:
- Content of the file being deleted
- Warning about permanent deletion
- File metadata and last modified date

#### 4. **Batch Operations**
For multiple file operations:
- Summary table of all operations
- Risk distribution across files
- Option to approve all, selective approval, or deny all

## Risk Assessment

The system automatically evaluates the risk level of each operation:

### 🟢 **Low Risk**
- Text files, documentation
- Small changes to non-critical files
- Creating new files in safe directories

### 🟡 **Medium Risk**
- Configuration files
- Scripts in user directories
- Moderate-sized changes to existing files

### 🟠 **High Risk**
- System configuration files
- Large file modifications
- Operations affecting multiple files

### 🔴 **Critical Risk**
- System files or directories
- Files with elevated permissions
- Operations that could affect system stability

## Advanced Features

### Diff View Modes

The system supports multiple ways to view file changes:

#### **Unified Diff** (Default)
```diff
- old line content
+ new line content
  unchanged content
```

#### **Side-by-Side**
```
Original               │ Modified
─────────────────────────────────────
old line content      │ new line content
unchanged content     │ unchanged content
```

#### **Inline Changes**
```python
def function_name():     # Changed from: old_function_name()
    return "new value"   # Changed from: "old value"
```

### Batch Approval Options

When handling multiple files:

```bash
📦 Batch Operations (5 files)

┌─────────────────────────────────────────────────────────┐
│ Operation Summary                                        │
├─────────────────┬─────────┬──────────────────────────────┤
│ Type           │ Count   │ Risk Distribution            │
├─────────────────┼─────────┼──────────────────────────────┤
│ Create         │    3    │ Low: 2, Medium: 1           │
│ Modify         │    1    │ Low: 1                      │
│ Delete         │    1    │ High: 1                     │
└─────────────────┴─────────┴──────────────────────────────┘

[A] Approve All  [S] Selective  [N] Deny All  [D] Details
```

### Approval History

The system maintains a complete audit trail:

```bash
>>> /approval-history

Recent Approvals:
2025-01-15 14:30:22 - APPROVED - Create data_analyzer.py (Low Risk)
2025-01-15 14:25:15 - DENIED   - Modify /etc/hosts (Critical Risk)
2025-01-15 14:20:08 - APPROVED - Update README.md (Low Risk)
```

## Configuration

### Environment Variables

Control system behavior with environment variables:

```bash
# Approval timeout (default: 300 seconds)
export OMNIMANCER_APPROVAL_TIMEOUT=600

# Default approval for low-risk operations (default: false)
export OMNIMANCER_AUTO_APPROVE_LOW_RISK=true

# Enable approval history logging (default: true)
export OMNIMANCER_LOG_APPROVALS=true

# Approval log file location
export OMNIMANCER_APPROVAL_LOG=~/.omnimancer/approvals.log
```

### Configuration File

Add approval settings to `~/.omnimancer/config.json`:

```json
{
  "agent": {
    "approval": {
      "timeout_seconds": 300,
      "auto_approve_low_risk": false,
      "log_decisions": true,
      "show_risk_assessment": true,
      "diff_type": "unified",
      "syntax_highlighting": true
    }
  }
}
```

## Security Considerations

### Safe Defaults
- **Deny by Default**: Timeout results in operation denial
- **Explicit Consent**: All operations require user action
- **Risk Awareness**: Clear risk indicators for all operations
- **Audit Trail**: Complete logging of decisions and operations

### Best Practices
1. **Review Changes Carefully**: Always check the diff before approving
2. **Understand Risk Levels**: Pay special attention to high/critical risk operations
3. **Use Selective Approval**: For batch operations, review each file individually
4. **Monitor Approval History**: Regularly review the audit trail
5. **Configure Timeouts**: Set appropriate timeout values for your workflow

### Permission Requirements
- **File System Access**: Required for file operations
- **Log File Access**: For audit trail maintenance
- **Configuration Access**: For saving approval preferences

## Troubleshooting

### Common Issues

#### **Approval Timeout**
```
⚠️ Approval timeout after 300 seconds
Operation denied for safety
```
**Solution**: Increase timeout or respond more quickly to prompts

#### **Permission Denied**
```
❌ Permission denied: Cannot write to /etc/config.conf
```
**Solution**: Run with appropriate permissions or choose different location

#### **Large File Warning**
```
⚠️ Large file operation (>10MB)
This may take significant time to display
```
**Solution**: Use the summary view or split operation into smaller parts

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
export OMNIMANCER_DEBUG_APPROVAL=true
omn --debug
```

## Developer Reference

### API Integration

Integrate the approval system in your code:

```python
from omnimancer.core.agent.file_modification_workflow import FileModificationWorkflow
from omnimancer.core.agent.file_content_display import UnifiedFileContentDisplay

# Initialize workflow
workflow = FileModificationWorkflow()

# Execute single file operation
result = await workflow.execute_single_file_workflow(
    file_path="/path/to/file.py",
    operation="create",
    new_content="print('Hello, World!')"
)

if result.final_result == WorkflowResult.APPROVED_AND_APPLIED:
    print("File created successfully!")
```

### Component Overview

| Component | Purpose | Location |
|-----------|---------|----------|
| `FileModificationWorkflow` | Main orchestrator | `omnimancer/core/agent/file_modification_workflow.py` |
| `UnifiedFileContentDisplay` | Content visualization | `omnimancer/core/agent/file_content_display.py` |
| `EnhancedDiffRenderer` | Diff generation | `omnimancer/core/agent/diff_renderer.py` |
| `ApprovalDialog` | User interaction | `omnimancer/core/agent/approval_dialog.py` |
| `ProposedChangesIntegration` | Change management | `omnimancer/core/agent/proposed_changes_integration.py` |

## Examples

### Example 1: Simple File Creation

```bash
>>> Create a Python script that prints "Hello World"

🔍 File Operation Approval Required
📄 Creating: hello.py
📊 Risk Level: Low
📏 Size: 22 bytes (1 line)

┌─────────────────────────────────────────┐
│ print("Hello, World!")                  │
└─────────────────────────────────────────┘

[Y] Approve  [N] Deny  [D] Details
Your choice: Y

✅ File created successfully: hello.py
```

### Example 2: File Modification with Diff

```bash
>>> Add error handling to the function

🔍 File Operation Approval Required
✏️ Modifying: calculator.py
📊 Risk Level: Low
📏 Changes: +5 lines, -0 lines

┌─────────────────────────────────────────┐
│ @@ -1,4 +1,9 @@                         │
│  def divide(a, b):                      │
│ +    if b == 0:                         │
│ +        raise ValueError("Cannot divide by zero") │
│ +    return a / b                       │
│ -    return a / b                       │
└─────────────────────────────────────────┘

[Y] Approve  [N] Deny  [D] Details
Your choice: Y

✅ File modified successfully: calculator.py
```

### Example 3: Batch Operations

```bash
>>> Refactor the project structure

🔍 Batch Operation Approval Required
📦 5 files affected

┌─────────────────────────────────────────┐
│ 1. CREATE  src/utils.py        (Low)    │
│ 2. MODIFY  main.py             (Low)    │
│ 3. MOVE    helper.py → lib/    (Medium) │
│ 4. DELETE  old_script.py       (Low)    │
│ 5. CREATE  tests/test_utils.py (Low)    │
└─────────────────────────────────────────┘

[A] Approve All  [S] Selective  [N] Deny All
Your choice: S

Select files to approve (space to toggle, enter to confirm):
[✓] 1. CREATE  src/utils.py        (Low)
[✓] 2. MODIFY  main.py             (Low)
[ ] 3. MOVE    helper.py → lib/    (Medium)
[✓] 4. DELETE  old_script.py       (Low)
[✓] 5. CREATE  tests/test_utils.py (Low)

✅ 4 operations approved, 1 skipped
```

## Changelog

### Version 2.0
- Added comprehensive approval system
- Implemented risk assessment
- Added multiple diff view modes
- Introduced batch operation support

### Version 2.1
- Enhanced visual interface
- Added approval history
- Improved error handling
- Added configuration options

---

**For more information**, see the [Security Guide](security.md) and [File Modification Interaction Flow](file-modification-interaction-flow.md).