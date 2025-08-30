"""Audit logger for tracking and monitoring agent security events."""

import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional


class AuditEventType(Enum):
    """Types of security audit events."""

    PERMISSION_CHECK = "permission_check"
    PERMISSION_DENIED = "permission_denied"
    COMMAND_EXECUTED = "command_executed"
    COMMAND_BLOCKED = "command_blocked"
    FILE_ACCESS = "file_access"
    FILE_ACCESS_DENIED = "file_access_denied"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    SANDBOX_CREATED = "sandbox_created"
    SANDBOX_VIOLATION = "sandbox_violation"
    SECURITY_ALERT = "security_alert"
    SYSTEM_EVENT = "system_event"


class AuditLevel(Enum):
    """Audit logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents a security audit event."""

    timestamp: datetime
    event_type: AuditEventType
    level: AuditLevel
    message: str
    source: str = "security_manager"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        # Add event hash for integrity checking
        self.metadata["event_hash"] = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate hash for event integrity."""
        # Create hashable content (excluding the hash itself)
        content = {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "level": self.level.value,
            "message": self.message,
            "source": self.source,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "operation_id": self.operation_id,
        }

        # Include metadata except for event_hash
        if self.metadata:
            filtered_metadata = {
                k: v for k, v in self.metadata.items() if k != "event_hash"
            }
            content["metadata"] = filtered_metadata

        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        data["level"] = self.level.value
        return data

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """Handles security audit logging and monitoring."""

    def __init__(
        self,
        log_file: Optional[str] = None,
        log_level: AuditLevel = AuditLevel.INFO,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        enable_async: bool = True,
    ):

        self.log_level = log_level
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_async = enable_async

        # Set up log file path
        if log_file:
            self.log_file = Path(log_file)
        else:
            log_dir = Path.home() / ".omnimancer" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = log_dir / "security_audit.log"

        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Set up Python logger
        self.logger = self._setup_logger()

        # Async logging setup
        if self.enable_async:
            self.log_queue = Queue()
            self.log_thread = threading.Thread(
                target=self._async_log_worker, daemon=True
            )
            self.log_thread.start()
        else:
            self.log_queue = None
            self.log_thread = None

        # Event statistics
        self.event_counts = {event_type: 0 for event_type in AuditEventType}
        self.level_counts = {level: 0 for level in AuditLevel}
        self.start_time = datetime.now(timezone.utc)

        # Log audit logger initialization
        self.log_event(
            AuditEventType.SYSTEM_EVENT,
            AuditLevel.INFO,
            "Security audit logger initialized",
            metadata={
                "log_file": str(self.log_file),
                "log_level": self.log_level.value,
                "async_enabled": self.enable_async,
            },
        )

    def _setup_logger(self) -> logging.Logger:
        """Set up Python logger for audit events."""

        logger = logging.getLogger("omnimancer.security.audit")
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        logger.handlers.clear()

        # File handler with rotation
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
        )

        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(
                logging.WARNING
            )  # Only warnings and above to console
            console_formatter = logging.Formatter(
                "%(asctime)s - SECURITY - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Detailed file formatter
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def _async_log_worker(self) -> None:
        """Worker thread for async logging."""

        while True:
            try:
                event = self.log_queue.get(timeout=1.0)
                if event is None:  # Shutdown signal
                    break
                self._write_event_sync(event)
                self.log_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                # Fallback to stderr for logging errors
                print(f"Audit logging error: {e}", file=__import__("sys").stderr)

    def log_event(
        self,
        event_type: AuditEventType,
        level: AuditLevel,
        message: str,
        source: str = "security_manager",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        operation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a security audit event."""

        # Check if event should be logged based on level
        if not self._should_log_level(level):
            return

        # Create audit event
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            level=level,
            message=message,
            source=source,
            user_id=user_id,
            session_id=session_id,
            operation_id=operation_id,
            metadata=metadata or {},
        )

        # Update statistics
        self.event_counts[event_type] += 1
        self.level_counts[level] += 1

        # Log the event
        if self.enable_async and self.log_queue:
            self.log_queue.put(event)
        else:
            self._write_event_sync(event)

    def _should_log_level(self, level: AuditLevel) -> bool:
        """Check if event level should be logged."""

        level_order = [
            AuditLevel.DEBUG,
            AuditLevel.INFO,
            AuditLevel.WARNING,
            AuditLevel.ERROR,
            AuditLevel.CRITICAL,
        ]

        return level_order.index(level) >= level_order.index(self.log_level)

    def _write_event_sync(self, event: AuditEvent) -> None:
        """Write event to log file synchronously."""

        try:
            # Write to file as JSON
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")

            # Also log to Python logger
            python_level = self._audit_level_to_python_level(event.level)
            log_message = f"[{event.event_type.value.upper()}] {event.message}"
            if event.metadata:
                log_message += f" | Metadata: {json.dumps(event.metadata, default=str)}"

            self.logger.log(python_level, log_message)

        except Exception as e:
            # Fallback logging
            print(
                f"Failed to write audit event: {e}",
                file=__import__("sys").stderr,
            )
            print(f"Event: {event.to_json()}", file=__import__("sys").stderr)

    def _audit_level_to_python_level(self, level: AuditLevel) -> int:
        """Convert audit level to Python logging level."""

        mapping = {
            AuditLevel.DEBUG: logging.DEBUG,
            AuditLevel.INFO: logging.INFO,
            AuditLevel.WARNING: logging.WARNING,
            AuditLevel.ERROR: logging.ERROR,
            AuditLevel.CRITICAL: logging.CRITICAL,
        }

        return mapping.get(level, logging.INFO)

    def log_permission_check(
        self,
        operation: str,
        path: Optional[str] = None,
        allowed: bool = True,
        reason: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a permission check event."""

        event_type = (
            AuditEventType.PERMISSION_CHECK
            if allowed
            else AuditEventType.PERMISSION_DENIED
        )
        level = AuditLevel.INFO if allowed else AuditLevel.WARNING

        message = f"Permission {'granted' if allowed else 'denied'} for {operation}"
        if path:
            message += f" on {path}"

        metadata = {
            "operation": operation,
            "path": path,
            "allowed": allowed,
            "reason": reason,
            **kwargs,
        }

        self.log_event(event_type, level, message, metadata=metadata)

    def log_command_execution(
        self,
        command: str,
        success: bool = True,
        exit_code: Optional[int] = None,
        output: Optional[str] = None,
        sandbox_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a command execution event."""

        event_type = (
            AuditEventType.COMMAND_EXECUTED
            if success
            else AuditEventType.COMMAND_BLOCKED
        )
        level = AuditLevel.INFO if success else AuditLevel.WARNING

        message = f"Command {'executed' if success else 'blocked'}: {command}"

        metadata = {
            "command": command,
            "success": success,
            "exit_code": exit_code,
            "sandbox_id": sandbox_id,
            **kwargs,
        }

        # Don't log full output for security, just indicate if present
        if output:
            metadata["output_length"] = len(output)
            metadata["output_preview"] = output[:100] if len(output) > 100 else output

        self.log_event(event_type, level, message, metadata=metadata)

    def log_file_access(
        self,
        path: str,
        operation: str,
        allowed: bool = True,
        file_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Log a file access event."""

        event_type = (
            AuditEventType.FILE_ACCESS if allowed else AuditEventType.FILE_ACCESS_DENIED
        )
        level = AuditLevel.INFO if allowed else AuditLevel.WARNING

        message = f"File {operation} {'allowed' if allowed else 'denied'}: {path}"

        metadata = {
            "path": path,
            "operation": operation,
            "allowed": allowed,
            "file_size": file_size,
            **kwargs,
        }

        self.log_event(event_type, level, message, metadata=metadata)

    def log_security_alert(
        self,
        alert_type: str,
        description: str,
        severity: AuditLevel = AuditLevel.WARNING,
        **kwargs,
    ) -> None:
        """Log a security alert."""

        message = f"Security alert: {alert_type} - {description}"

        metadata = {
            "alert_type": alert_type,
            "description": description,
            **kwargs,
        }

        self.log_event(
            AuditEventType.SECURITY_ALERT, severity, message, metadata=metadata
        )

    def get_recent_events(
        self,
        count: int = 100,
        event_type: Optional[AuditEventType] = None,
        level: Optional[AuditLevel] = None,
    ) -> List[AuditEvent]:
        """Get recent audit events from the log file."""

        events = []

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Process lines in reverse order (most recent first)
            for line in reversed(
                lines[-count * 2 :]
            ):  # Read more lines to account for filtering
                try:
                    event_data = json.loads(line.strip())

                    # Recreate AuditEvent object
                    event = AuditEvent(
                        timestamp=datetime.fromisoformat(event_data["timestamp"]),
                        event_type=AuditEventType(event_data["event_type"]),
                        level=AuditLevel(event_data["level"]),
                        message=event_data["message"],
                        source=event_data.get("source", "unknown"),
                        user_id=event_data.get("user_id"),
                        session_id=event_data.get("session_id"),
                        operation_id=event_data.get("operation_id"),
                        metadata=event_data.get("metadata", {}),
                    )

                    # Apply filters
                    if event_type and event.event_type != event_type:
                        continue
                    if level and event.level != level:
                        continue

                    events.append(event)

                    if len(events) >= count:
                        break

                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        except (FileNotFoundError, PermissionError):
            pass

        return events

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics."""

        uptime = datetime.now(timezone.utc) - self.start_time

        return {
            "uptime_seconds": int(uptime.total_seconds()),
            "log_file": str(self.log_file),
            "log_level": self.log_level.value,
            "async_enabled": self.enable_async,
            "total_events": sum(self.event_counts.values()),
            "event_type_counts": {k.value: v for k, v in self.event_counts.items()},
            "level_counts": {k.value: v for k, v in self.level_counts.items()},
            "log_file_size": (
                self.log_file.stat().st_size if self.log_file.exists() else 0
            ),
        }

    def shutdown(self) -> None:
        """Shutdown the audit logger gracefully."""

        self.log_event(
            AuditEventType.SYSTEM_EVENT,
            AuditLevel.INFO,
            "Security audit logger shutting down",
        )

        if self.enable_async and self.log_queue and self.log_thread:
            # Signal shutdown and wait for thread
            self.log_queue.put(None)
            self.log_thread.join(timeout=5.0)

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self.shutdown()
        except:
            pass
