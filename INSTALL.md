# Omnimancer Installation Guide

This guide shows you how to install Omnimancer as a system-wide binary that works from any terminal without needing to activate virtual environments.

## Quick Installation

### Option 1: Install from PyPI (Recommended)

```bash
# Using pipx (recommended)
pipx install omnimancer-cli

# Or using pip
pip install omnimancer-cli
```

### Option 2: Install from Source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/omnimancer.git
   cd omnimancer
   ```

2. **Install with pipx (recommended)**:
   ```bash
   # Install pipx if needed
   pip install pipx
   
   # Install Omnimancer
   pipx install .
   ```

3. **Or install with pip**:
   ```bash
   pip install .
   ```

### Option 3: Development Installation

For development and testing:

```bash
git clone https://github.com/your-repo/omnimancer.git
cd omnimancer
pip install -e ".[dev]"
```

2. **Add to PATH if needed**:
   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
   ```

## Available Commands

After installation, you can use any of these commands:

- **`omnimancer`** - Full command name
- **`omn`** - Short alias 
- **`omniman`** - Alternative alias

## Verification

After installation, you should be able to:

- ✅ Run `omnimancer` or `omn` from any directory
- ✅ Run `omnimancer --help` to see options
- ✅ Run `omnimancer --version` to see version info
- ✅ Use Omnimancer without activating any virtual environment

**Quick test:**
```bash
# All of these should work
omnimancer --version
omn --version
omniman --version
```

## Troubleshooting

### `omnimancer: command not found`

This usually means the installation directory isn't in your PATH:

```bash
# Check where pip installed it
pip show omnimancer-cli | grep Location

# Add the bin directory to PATH
export PATH="$HOME/.local/bin:$PATH"

# Make it permanent
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### `import omnimancer` fails

This suggests a dependency issue:

```bash
# Reinstall with dependencies
pip install --force-reinstall omnimancer-cli

# Or try pipx
pipx install omnimancer-cli
```

### Permission issues

If you get permission errors:

```bash
# Use user installation
pip install --user omnimancer-cli

# Or use pipx (recommended)
pipx install omnimancer-cli
```

### Issues with aliases

If `omn` or `omniman` commands don't work:

```bash
# Check if they're installed
which omnimancer omn omniman

# Reinstall if missing
pip install --force-reinstall omnimancer-cli
```

## Usage

Once installed, start Omnimancer using any of the available commands:

```bash
# Long form
omnimancer

# Short aliases  
omn
omniman
```

From any terminal, in any directory. No virtual environment activation needed!

### First Run Setup

On first run, you'll be guided through the setup wizard:

```bash
omn
🚀 Starting Omnimancer Setup Wizard...

Select a provider to configure:
1. Claude (Anthropic)
2. OpenAI
3. Google Gemini
4. Perplexity AI
...
```

## Uninstalling

To remove Omnimancer:

```bash
# If installed with pip
pip uninstall omnimancer-cli

# If installed with pipx
pipx uninstall omnimancer-cli
```