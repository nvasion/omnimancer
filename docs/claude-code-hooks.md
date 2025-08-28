# Claude Code Test Enforcement Hooks

This document explains the global Claude Code hooks that enforce test writing for code changes.

## Overview

The test enforcement hooks automatically run after every `Edit` or `Write` operation in Claude Code, checking if your code changes have corresponding test coverage.

## Hook Files

- **Global Hook Script**: `~/.claude/global-test-hook.js`
- **Post-Edit Hook**: `~/.claude/hooks/post-file-edit.sh`  
- **Post-Create Hook**: `~/.claude/hooks/post-file-create.sh`

## How It Works

1. When you edit or create a file using Claude Code
2. The hook checks if the file is a source code file (not in test directories)
3. It looks for corresponding test files using common naming patterns
4. If no tests are found, it provides helpful suggestions

## Supported File Types

- **Python**: `.py` files (looks for `tests/test_*.py`)
- **JavaScript/TypeScript**: `.js`, `.ts`, `.jsx`, `.tsx` files (looks for `*.test.js`, `*.spec.js`)
- **Other languages**: Basic support for common source file extensions

## Test File Patterns

The hook looks for test files in these patterns:

### Python
- `tests/test_{module_name}.py`
- `tests/{module_path}/test_{module_name}.py` 
- `{module_dir}/test_{module_name}.py`

### JavaScript/TypeScript
- `tests/{module_name}.test.{ext}`
- `__tests__/{module_name}.test.{ext}`
- `{module_dir}/{module_name}.test.{ext}`

## What the Hook Does

✅ **When tests exist**: Shows a green checkmark with the test file location

❌ **When tests are missing**: Shows suggestions for:
- Where to create test files
- What to name them
- Best practices for testing

## Hook Output Examples

### Good - Tests Found
```
🧪 Test Enforcement Hook (Edit): omnimancer/core/engine.py
✅ Found test coverage in: tests/test_engine.py
```

### Needs Tests
```
🧪 Test Enforcement Hook (Write): omnimancer/core/new_feature.py
❌ No corresponding test files found

💡 Suggestions:
   • Create tests/test_new_feature.py
   • Add test functions like: def test_new_feature_functionality():
   • Consider using pytest for testing

📋 Best Practices:
   • Write tests before or immediately after code changes
   • Aim for high test coverage of new functionality
   • Follow existing test patterns in the project
   • Test both happy paths and edge cases
```

## Disabling the Hook

If needed, you can disable the hooks by setting them to non-executable:

```bash
chmod -x ~/.claude/hooks/post-file-edit.sh
chmod -x ~/.claude/hooks/post-file-create.sh
```

To re-enable:
```bash
chmod +x ~/.claude/hooks/post-file-edit.sh
chmod +x ~/.claude/hooks/post-file-create.sh
```

## Customization

You can customize the hook behavior by editing `~/.claude/global-test-hook.js`:

- Modify file patterns to match your project structure
- Add support for additional languages
- Change the test file naming conventions
- Adjust the verbosity of output

## Project-Specific Overrides

You can also create project-specific hooks in `.claude/hooks/` within your project directory to override the global behavior.

## Benefits

- **Consistent Test Coverage**: Ensures every code change is accompanied by tests
- **Educational**: Teaches proper test file naming and organization
- **Non-Intrusive**: Only shows reminders, doesn't block your work
- **Flexible**: Works with multiple languages and project structures

The hooks are designed to be helpful reminders that encourage good testing practices without being overly restrictive.