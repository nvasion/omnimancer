# MCP (Model Context Protocol) Setup Guide

This comprehensive guide covers setting up and configuring MCP servers with Omnimancer CLI to extend AI capabilities with external tools and data sources.

## What is MCP?

Model Context Protocol (MCP) is an open standard developed by Anthropic that enables AI models to securely connect to external data sources and tools. MCP transforms your AI assistant from a text-only interface into a powerful agent capable of:

### Core Capabilities
- **File System Operations**: Read, write, and manage files and directories
- **Web Search**: Access real-time information from the internet
- **Database Queries**: Connect to and query various database systems
- **API Integrations**: Interact with REST APIs and web services
- **Code Execution**: Run scripts and commands in controlled environments
- **Version Control**: Manage Git repositories and track changes
- **Cloud Services**: Integrate with AWS, Google Cloud, Azure services
- **Development Tools**: Access IDEs, build systems, and deployment pipelines

### How MCP Works

```mermaid
graph LR
    A[Omnimancer CLI] --> B[MCP Manager]
    B --> C[MCP Server 1]
    B --> D[MCP Server 2]
    B --> E[MCP Server N]
    C --> F[File System]
    D --> G[Web APIs]
    E --> H[Databases]
```

1. **MCP Servers**: Standalone processes that provide specific tools and capabilities
2. **MCP Manager**: Coordinates communication between Omnimancer and MCP servers
3. **Tool Discovery**: Automatically discovers available tools from connected servers
4. **Secure Execution**: Tools run in controlled environments with user approval

## Prerequisites

### Install UV (Python Package Manager)

Most MCP servers use `uvx` (part of the `uv` Python package manager) for execution:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

## Configuration

### Configuration File Locations

MCP servers can be configured at two levels:

1. **User-level** (applies to all workspaces): `~/.omnimancer/mcp.json`
2. **Workspace-level** (workspace-specific): `./.omnimancer/mcp.json`

If both exist, workspace-level settings take precedence.

### Basic Configuration Structure

```json
{
  "mcpServers": {
    "server-name": {
      "command": "uvx",
      "args": ["package-name", "additional-args"],
      "env": {
        "ENV_VAR": "value"
      },
      "enabled": true,
      "autoApprove": ["tool1", "tool2"]
    }
  }
}
```

## Popular MCP Servers

### 1. Filesystem Server

Provides secure file system operations with configurable access controls.

**Installation:**
```bash
# Server is automatically downloaded when first used
uvx mcp-server-filesystem --help
```

**Configuration:**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "uvx",
      "args": ["mcp-server-filesystem", "/path/to/allowed/directory"],
      "env": {},
      "enabled": true,
      "autoApprove": ["read_file", "list_directory"]
    }
  }
}
```

**Available Tools:**
- `read_file` - Read file contents with encoding detection
- `write_file` - Write content to files with backup creation
- `list_directory` - List directory contents with metadata
- `create_directory` - Create directories recursively
- `move_file` - Move/rename files and directories
- `delete_file` - Delete files with confirmation
- `search_files` - Search for files by name or content
- `get_file_info` - Get detailed file metadata

**Security Features:**
- Restricted to specified directory tree
- Read-only mode available
- File type filtering
- Size limits for operations

**Example Usage:**
```
>>> What files are in my current directory?
🤖 I'll check your current directory for you.

[Tool: list_directory executed]
Found 8 files:
- README.md (2.3 KB, modified 2 hours ago)
- src/ (directory, 15 files)
- tests/ (directory, 8 files)
- package.json (1.2 KB, modified 1 day ago)
```

### 2. Web Search Server

Search the web using Brave Search API for real-time information.

**Setup:**
1. Get a Brave Search API key from [Brave Search API](https://api.search.brave.com/)
2. Set the environment variable:
   ```bash
   export BRAVE_API_KEY="your-api-key"
   ```

**Configuration:**
```json
{
  "mcpServers": {
    "web_search": {
      "command": "uvx",
      "args": ["mcp-server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
      },
      "enabled": true,
      "autoApprove": []
    }
  }
}
```

**Available Tools:**
- `web_search` - Search the web with customizable parameters
- `local_search` - Search for local businesses and services
- `news_search` - Search for recent news articles
- `image_search` - Search for images (URLs only)

**Search Parameters:**
- Query string and filters
- Result count (1-20)
- Safe search settings
- Country/region targeting
- Time-based filtering

**Example Usage:**
```
>>> What's the latest news about AI developments in 2024?
🤖 I'll search for the latest AI news for you.

[Tool: web_search executed]
Found recent articles about AI developments:

1. "Major AI Breakthrough in Language Models" - TechNews
   Published 2 hours ago
   Summary: Researchers announce significant improvements...

2. "AI Regulation Updates from EU Parliament" - EuroTech
   Published 5 hours ago
   Summary: New guidelines for AI development...
```

### 3. SQLite Database Server

Query and manage SQLite databases with full SQL support.

**Configuration:**
```json
{
  "mcpServers": {
    "database": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "/path/to/database.db"],
      "env": {},
      "enabled": true,
      "autoApprove": ["read_query", "list_tables"]
    }
  }
}
```

**Available Tools:**
- `read_query` - Execute SELECT queries with result formatting
- `write_query` - Execute INSERT/UPDATE/DELETE with transaction support
- `list_tables` - List all tables with row counts
- `describe_table` - Get detailed table schema and indexes
- `execute_sql` - Execute arbitrary SQL commands
- `backup_database` - Create database backups
- `analyze_table` - Get table statistics and optimization suggestions

**Features:**
- Transaction support
- Query result formatting (JSON, CSV, table)
- Query performance analysis
- Schema introspection
- Backup and restore capabilities

**Example Usage:**
```
>>> Show me the structure of the users table
🤖 I'll examine the users table structure for you.

[Tool: describe_table executed]
Table: users
Columns:
- id (INTEGER PRIMARY KEY AUTOINCREMENT)
- username (TEXT NOT NULL UNIQUE)
- email (TEXT NOT NULL)
- created_at (DATETIME DEFAULT CURRENT_TIMESTAMP)
- last_login (DATETIME)

Indexes:
- idx_users_email (email)
- idx_users_username (username)

Row count: 1,247
```

### 4. Git Server

Comprehensive Git repository management and version control.

**Configuration:**
```json
{
  "mcpServers": {
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git", "/path/to/repository"],
      "env": {},
      "enabled": true,
      "autoApprove": ["git_status", "git_log", "git_diff"]
    }
  }
}
```

**Available Tools:**
- `git_status` - Check repository status with detailed file information
- `git_log` - View commit history with filtering options
- `git_diff` - Show changes between commits, branches, or working directory
- `git_add` - Stage files for commit
- `git_commit` - Create commits with message validation
- `git_push` - Push changes to remote repositories
- `git_pull` - Pull changes from remote repositories
- `git_branch` - List, create, or delete branches
- `git_checkout` - Switch branches or restore files
- `git_merge` - Merge branches with conflict detection
- `git_stash` - Stash and manage uncommitted changes
- `git_remote` - Manage remote repositories

**Advanced Features:**
- Commit message templates
- Pre-commit hook integration
- Conflict resolution assistance
- Branch management workflows
- Remote repository synchronization

**Example Usage:**
```
>>> What's the current status of my repository?
🤖 I'll check your Git repository status.

[Tool: git_status executed]
Repository Status:
Branch: feature/new-ui (ahead 2 commits)
Remote: origin/main

Modified files:
- src/components/Header.tsx (15 lines changed)
- styles/main.css (3 lines changed)

Untracked files:
- docs/api-guide.md
- tests/header.test.tsx

Staged files: None
```

### 5. AWS Server

Interact with AWS services for cloud resource management.

**Setup:**
1. Configure AWS credentials:
   ```bash
   aws configure
   ```
   Or set environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID="your-key"
   export AWS_SECRET_ACCESS_KEY="your-secret"
   export AWS_DEFAULT_REGION="us-east-1"
   ```

**Configuration:**
```json
{
  "mcpServers": {
    "aws": {
      "command": "uvx",
      "args": ["mcp-server-aws"],
      "env": {},
      "enabled": true,
      "autoApprove": ["list_s3_buckets", "describe_ec2_instances"]
    }
  }
}
```

**Available Tools:**
- `list_s3_buckets` - List S3 buckets with metadata
- `s3_operations` - Upload, download, and manage S3 objects
- `describe_ec2_instances` - List EC2 instances with details
- `ec2_operations` - Start, stop, and manage EC2 instances
- `lambda_functions` - List and invoke Lambda functions
- `cloudwatch_metrics` - Query CloudWatch metrics and logs
- `rds_instances` - Manage RDS database instances
- `iam_operations` - List users, roles, and policies

### 6. Docker Server

Manage Docker containers and images.

**Configuration:**
```json
{
  "mcpServers": {
    "docker": {
      "command": "uvx",
      "args": ["mcp-server-docker"],
      "env": {},
      "enabled": true,
      "autoApprove": ["list_containers", "list_images"]
    }
  }
}
```

**Available Tools:**
- `list_containers` - List running and stopped containers
- `container_operations` - Start, stop, restart containers
- `list_images` - List Docker images with details
- `image_operations` - Build, pull, push, remove images
- `container_logs` - View container logs with filtering
- `container_exec` - Execute commands in running containers
- `docker_stats` - Monitor container resource usage

### 7. Kubernetes Server

Manage Kubernetes clusters and resources.

**Configuration:**
```json
{
  "mcpServers": {
    "kubernetes": {
      "command": "uvx",
      "args": ["mcp-server-kubernetes"],
      "env": {
        "KUBECONFIG": "/path/to/kubeconfig"
      },
      "enabled": true,
      "autoApprove": ["get_pods", "get_services"]
    }
  }
}
```

**Available Tools:**
- `get_pods` - List pods with status and resource usage
- `get_services` - List services and endpoints
- `get_deployments` - List deployments with replica status
- `kubectl_apply` - Apply Kubernetes manifests
- `kubectl_delete` - Delete Kubernetes resources
- `pod_logs` - View pod logs with filtering
- `port_forward` - Set up port forwarding to pods

### 8. PostgreSQL Server

Advanced PostgreSQL database operations.

**Configuration:**
```json
{
  "mcpServers": {
    "postgres": {
      "command": "uvx",
      "args": ["mcp-server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/dbname"
      },
      "enabled": true,
      "autoApprove": ["read_query", "list_tables"]
    }
  }
}
```

**Available Tools:**
- `read_query` - Execute SELECT queries with advanced formatting
- `write_query` - Execute DML operations with transaction support
- `list_schemas` - List database schemas
- `list_tables` - List tables with detailed metadata
- `describe_table` - Get comprehensive table information
- `analyze_performance` - Query performance analysis
- `manage_indexes` - Create and analyze database indexes

### 9. Slack Server

Integrate with Slack for team communication.

**Setup:**
1. Create a Slack app at [api.slack.com](https://api.slack.com/apps)
2. Get bot token and configure permissions
3. Set environment variable:
   ```bash
   export SLACK_BOT_TOKEN="xoxb-your-token"
   ```

**Configuration:**
```json
{
  "mcpServers": {
    "slack": {
      "command": "uvx",
      "args": ["mcp-server-slack"],
      "env": {
        "SLACK_BOT_TOKEN": "${SLACK_BOT_TOKEN}"
      },
      "enabled": true,
      "autoApprove": []
    }
  }
}
```

**Available Tools:**
- `send_message` - Send messages to channels or users
- `list_channels` - List available channels
- `get_channel_history` - Retrieve channel message history
- `upload_file` - Upload files to Slack
- `create_channel` - Create new channels
- `invite_users` - Invite users to channels

### 10. Email Server

Send and manage emails through various providers.

**Configuration:**
```json
{
  "mcpServers": {
    "email": {
      "command": "uvx",
      "args": ["mcp-server-email"],
      "env": {
        "SMTP_HOST": "smtp.gmail.com",
        "SMTP_PORT": "587",
        "SMTP_USER": "${EMAIL_USER}",
        "SMTP_PASS": "${EMAIL_PASS}"
      },
      "enabled": true,
      "autoApprove": []
    }
  }
}
```

**Available Tools:**
- `send_email` - Send emails with attachments
- `send_template_email` - Send emails using templates
- `validate_email` - Validate email addresses
- `check_deliverability` - Check email deliverability status

## Managing MCP Servers

### CLI Commands

Use these commands within Omnimancer to manage MCP servers:

```bash
/mcp status          # Show status of all MCP servers
/mcp reload          # Reload MCP server configurations
/mcp health          # Check health of all servers
/tools               # List all available tools
```

### Auto-Approval

The `autoApprove` setting allows certain tools to run without user confirmation:

```json
{
  "autoApprove": ["read_file", "list_directory", "git_status"]
}
```

**Security Note:** Only auto-approve tools you trust completely, as they can access your system.

## Troubleshooting

### Common Issues

#### "MCP server failed to start"
**Causes:**
- `uv` not installed
- Invalid server configuration
- Missing dependencies

**Solutions:**
1. Install `uv`:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Check configuration syntax
3. Verify server package exists

#### "Tool execution failed"
**Causes:**
- Insufficient permissions
- Invalid arguments
- Server not responding

**Solutions:**
1. Check file/directory permissions
2. Verify tool arguments
3. Restart MCP servers: `/mcp reload`

#### "Environment variable not found"
**Causes:**
- Environment variable not set
- Incorrect variable name

**Solutions:**
1. Set the required environment variable:
   ```bash
   export VAR_NAME="value"
   ```
2. Restart Omnimancer to pick up new environment variables

### Debug Mode

Enable debug logging for MCP issues:

```bash
export OMNIMANCER_DEBUG=1
export MCP_DEBUG=1
omnimancer
```

### Server Logs

Check individual server logs:
```bash
# Most MCP servers log to stderr
uvx mcp-server-filesystem /path 2>&1 | tee server.log
```

## Security Considerations

### File System Access

- Limit filesystem server to specific directories
- Use auto-approve carefully for file operations
- Regularly review tool usage logs

### API Keys

- Store API keys in environment variables, not configuration files
- Use least-privilege API keys when possible
- Rotate API keys regularly

### Network Access

- Be cautious with web search and API tools
- Monitor network requests in debug mode
- Consider using VPN for sensitive operations

## Advanced Configuration

### Custom MCP Servers

You can create custom MCP servers for specific needs:

```json
{
  "mcpServers": {
    "custom": {
      "command": "python",
      "args": ["/path/to/custom-server.py"],
      "env": {
        "CUSTOM_CONFIG": "/path/to/config"
      },
      "enabled": true,
      "autoApprove": []
    }
  }
}
```

### Conditional Server Loading

Load servers based on workspace or environment:

```json
{
  "mcpServers": {
    "development": {
      "command": "uvx",
      "args": ["mcp-server-git", "."],
      "env": {},
      "enabled": "${NODE_ENV:development}",
      "autoApprove": []
    }
  }
}
```

### Server Health Monitoring

Configure health check intervals:

```json
{
  "mcpConfig": {
    "healthCheckInterval": 30,
    "maxRetries": 3,
    "timeout": 10
  }
}
```

## Best Practices

1. **Start Small**: Begin with filesystem and web search servers
2. **Security First**: Only auto-approve safe, read-only operations
3. **Monitor Usage**: Regularly check tool usage and server health
4. **Environment Variables**: Use environment variables for sensitive data
5. **Workspace-Specific**: Use workspace-level configs for project-specific tools
6. **Regular Updates**: Keep MCP servers updated with `uvx` cache clean
7. **Backup Configs**: Version control your MCP configurations

## Examples

### Development Workflow

Configuration for a typical development environment:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "uvx",
      "args": ["mcp-server-filesystem", "."],
      "env": {},
      "enabled": true,
      "autoApprove": ["read_file", "list_directory"]
    },
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git", "."],
      "env": {},
      "enabled": true,
      "autoApprove": ["git_status", "git_log"]
    },
    "web_search": {
      "command": "uvx",
      "args": ["mcp-server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
      },
      "enabled": true,
      "autoApprove": []
    }
  }
}
```

### Data Analysis Workflow

Configuration for data analysis tasks:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "uvx",
      "args": ["mcp-server-filesystem", "/data"],
      "env": {},
      "enabled": true,
      "autoApprove": ["read_file", "list_directory"]
    },
    "database": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "/data/analysis.db"],
      "env": {},
      "enabled": true,
      "autoApprove": ["read_query", "list_tables"]
    },
    "web_search": {
      "command": "uvx",
      "args": ["mcp-server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
      },
      "enabled": true,
      "autoApprove": []
    }
  }
}
```

## Getting Help

- **Omnimancer Issues**: [GitHub Issues](https://github.com/omnimancer-cli/omnimancer/issues)
- **MCP Specification**: [MCP Documentation](https://modelcontextprotocol.io/)
- **Server Registry**: [MCP Server Registry](https://github.com/modelcontextprotocol/servers)