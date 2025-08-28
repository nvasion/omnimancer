# Custom Commands for Omnimancer

This directory contains example custom commands that demonstrate how to extend Omnimancer with your own slash commands, similar to Claude Code's command system.

## Features

- **Dynamic Command Loading**: Commands are loaded automatically from the user's config directory
- **Autocomplete Support**: Full tab-completion for command names and arguments
- **Multiple Command Types**: Support for Python handlers, shell scripts, and JSON definitions
- **Argument Validation**: Define argument types, choices, and requirements
- **Rich Formatting**: Use Rich library for beautiful terminal output

## Installation

1. Create your commands directory:
```bash
mkdir -p ~/.config/omnimancer/commands
```

2. Copy example commands to your commands directory:
```bash
cp examples/custom_commands/*.py ~/.config/omnimancer/commands/
cp examples/custom_commands/*.sh ~/.config/omnimancer/commands/
cp examples/custom_commands/*.json ~/.config/omnimancer/commands/
```

3. Make shell scripts executable:
```bash
chmod +x ~/.config/omnimancer/commands/*.sh
```

## Command Types

### 1. Python Commands

Create a `.py` file with a `COMMAND_INFO` dictionary and a `handle_command` function:

```python
#!/usr/bin/env python3
COMMAND_INFO = {
    'name': 'mycommand',
    'description': 'My custom command',
    'arguments': [
        {
            'name': 'arg1',
            'type': 'string',
            'description': 'First argument',
            'choices': ['option1', 'option2'],  # Optional
            'required': False,
            'default': 'option1'
        }
    ]
}

def handle_command(args, **kwargs):
    """Handle the command."""
    console = kwargs.get('console')
    engine = kwargs.get('engine')
    
    # Your command logic here
    return "Command output"

# Or for async commands:
async def handle_command(args, **kwargs):
    """Handle the command asynchronously."""
    # Your async command logic here
    return "Command output"
```

### 2. Shell Script Commands

Create a `.sh` file with metadata in comments:

```bash
#!/bin/bash
# NAME: mycommand
# DESCRIPTION: My shell script command
# ARG: arg1:string:First argument
# ARG: arg2:file:File path argument

# Your script logic here
echo "Hello from shell script!"
```

### 3. JSON Command Definitions

Create a `.json` file that references a script:

```json
{
  "name": "mycommand",
  "description": "My JSON-defined command",
  "arguments": [
    {
      "name": "input",
      "type": "file",
      "description": "Input file path"
    }
  ],
  "script": "mycommand_handler.py"
}
```

## Argument Types

- `string`: Text input
- `file`: File path (autocompletes file names)
- `directory`: Directory path (autocompletes directory names)
- Custom types can be defined with `choices` array

## Example Commands

### `/greet [name] [style]`
A friendly greeting command with customizable messages.
- `name`: Person to greet (default: "friend")
- `style`: Greeting style - formal, casual, or enthusiastic

### `/timestamp [format]`
Display current timestamp in various formats.
- `format`: iso, unix, human, or all

### `/wordcount [type]`
Analyze conversation history and provide statistics.
- `type`: words, messages, tokens, or all

### `/custom [action] [command]`
Manage custom commands.
- `action`: list, reload, info, or path
- `command`: Command name (for info action)

## Using Custom Commands

Once installed, your custom commands will:
1. Appear in tab-completion when you type `/`
2. Show argument suggestions when you press Tab
3. Execute when you enter them

Example:
```
>>> /greet Alice formal
Good day, Alice. How may I assist you today?

>>> /timestamp iso
2024-01-15T10:30:45Z

>>> /custom list
╭─────────────────────────────────╮
│      Custom Commands            │
├─────────────────────────────────┤
│ Command    │ Description        │
├────────────┼────────────────────┤
│ /greet     │ A friendly greeting│
│ /timestamp │ Display timestamp  │
│ /wordcount │ Count words        │
│ /custom    │ Manage commands    │
╰─────────────────────────────────╯
```

## Advanced Features

### Context Access

Commands receive these in `kwargs`:
- `engine`: The Omnimancer engine instance
- `console`: Rich console for formatted output

### Async Support

Python handlers can be async functions for non-blocking operations.

### Dynamic Choices

Argument choices can be functions that return dynamic lists:

```python
def get_model_list():
    # Return current list of models
    return ['model1', 'model2', 'model3']

COMMAND_INFO = {
    'arguments': [{
        'name': 'model',
        'choices': get_model_list
    }]
}
```

## Troubleshooting

- **Commands not loading**: Check the commands directory path with `/custom path`
- **Script not executing**: Ensure shell scripts are executable (`chmod +x`)
- **Python import errors**: Commands run in Omnimancer's Python environment
- **Autocomplete not working**: Reload commands with `/custom reload`

## Creating Your Own Commands

1. Choose a command type (Python, Shell, or JSON)
2. Create the file in `~/.config/omnimancer/commands/`
3. Define metadata (name, description, arguments)
4. Implement the command logic
5. Reload with `/custom reload` or restart Omnimancer

Happy commanding! 🚀