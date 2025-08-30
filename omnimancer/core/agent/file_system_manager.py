"""File system operations manager with security, backup, and atomic operations."""

import glob as glob_module
import hashlib
import logging
import mimetypes
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import aiofiles
import aiofiles.os

logger = logging.getLogger(__name__)

from ..security import SecurityManager
from ..security.permission_controller import PermissionOperation
from .approval_manager import EnhancedApprovalManager
from .read_before_write_errors import (
    CallbackError,
    ContentValidationError,
    DiffGenerationError,
    FileReadError,
    FileWriteError,
    ReadBeforeWriteErrorHandler,
)
from .types import Operation, OperationResult, OperationType


class FileOperationError(Exception):
    """Custom exception for file operation errors."""

    pass


class FileSystemManager:
    """Comprehensive file system operations with security, backup, and atomic operations."""

    def __init__(
        self,
        security_manager: Optional[SecurityManager] = None,
        approval_manager: Optional[EnhancedApprovalManager] = None,
        backup_dir: Optional[str] = None,
        max_file_size_mb: int = 100,
        chunk_size: int = 8192,
        require_approval: bool = True,
        error_handler: Optional[ReadBeforeWriteErrorHandler] = None,
        base_path: Optional[str] = None,
    ):

        self.security = security_manager or SecurityManager()
        self.approval_manager = approval_manager
        self.max_file_size_mb = max_file_size_mb
        self.chunk_size = chunk_size
        self.require_approval = require_approval

        # Set up base path for operations (for test compatibility)
        self.base_path = Path(base_path) if base_path else Path.cwd()

        # Set up read-before-write error handler
        self.error_handler = error_handler or ReadBeforeWriteErrorHandler()

        # Set up backup directory
        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            self.backup_dir = Path("/tmp/omnimancer_backups")

        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Operation tracking
        self.active_operations: Dict[str, Dict[str, Any]] = {}

    async def _request_approval_if_needed(self, operation: Operation) -> bool:
        """Request approval for operation if approval manager is configured."""
        if not self.require_approval or not self.approval_manager:
            return True

        # Request approval through the approval manager
        return await self.approval_manager.request_single_approval(operation)

    def _create_operation(
        self,
        operation_type: OperationType,
        description: str,
        data: Dict[str, Any],
        reversible: bool = False,
    ) -> Operation:
        """Create an Operation object for approval workflow."""
        return Operation(
            type=operation_type,
            description=description,
            data=data,
            requires_approval=self.require_approval,
            reversible=reversible,
        )

    async def check_file_exists(
        self, path: Union[str, Path], follow_symlinks: bool = True
    ) -> Dict[str, Any]:
        """
        Check if a file exists with comprehensive error handling and metadata.

        Args:
            path: Path to check for existence
            follow_symlinks: Whether to follow symbolic links (default: True)

        Returns:
            Dict containing:
                - exists: Boolean indicating if file exists
                - path: Resolved path string
                - is_file: Boolean indicating if path points to a file
                - is_directory: Boolean indicating if path points to a directory
                - is_symlink: Boolean indicating if path is a symbolic link
                - size: File size in bytes (if exists and is file)
                - modified_time: Last modified timestamp (if exists)
                - error: Error message if check failed
        """
        try:
            original_path = Path(path)
            # Use resolved path for file operations, but preserve original path for return value
            resolved_path = (
                original_path.resolve() if follow_symlinks else original_path
            )

            # Check if original path is a symlink before resolving
            is_symlink = await aiofiles.os.path.islink(original_path)

            # First check existence using aiofiles for async operation
            exists = await aiofiles.os.path.exists(resolved_path)

            if not exists:
                return {
                    "exists": False,
                    "path": str(original_path),  # Return original path, not resolved
                    "is_file": False,
                    "is_directory": False,
                    "is_symlink": is_symlink,  # May be True for broken symlinks
                    "size": None,
                    "modified_time": None,
                    "error": None,
                }

            # Since path exists, do security validation for access
            # Note: We only do security check for actual access control,
            # not for existence determination
            try:
                security_result = await self.security.secure_file_access(
                    str(resolved_path), "read"
                )
                if not security_result["success"]:
                    # Check if it's a real security denial vs. SecurityManager limitations
                    error_msg = security_result.get("error", "")
                    # If SecurityManager says "does not exist" but we know it exists,
                    # it might be a directory or other filesystem object the SecurityManager doesn't handle
                    if (
                        "does not exist" in error_msg.lower()
                        or "not found" in error_msg.lower()
                    ):
                        # Continue with existence check - SecurityManager may not handle directories properly
                        logger.debug(
                            f"SecurityManager doesn't recognize {resolved_path}, continuing with existence check"
                        )
                    else:
                        # Actual security denial for existing path
                        return {
                            "exists": True,
                            "path": str(
                                original_path
                            ),  # Return original path, not resolved
                            "is_file": False,
                            "is_directory": False,
                            "is_symlink": is_symlink,
                            "size": None,
                            "modified_time": None,
                            "error": f"Access denied: {error_msg}",
                        }
            except Exception as security_error:
                logger.warning(
                    f"Security check failed for {resolved_path}: {security_error}"
                )
                # Continue with existence check despite security error
                logger.debug(
                    f"Continuing existence check despite security error: {security_error}"
                )

            # Gather additional metadata if file exists
            try:
                is_file = await aiofiles.os.path.isfile(resolved_path)
                is_directory = await aiofiles.os.path.isdir(resolved_path)
                # is_symlink was already determined above

                # Get file size and modification time
                stat_result = await aiofiles.os.stat(resolved_path)
                file_size = stat_result.st_size if is_file else None
                modified_time = datetime.fromtimestamp(stat_result.st_mtime)

                return {
                    "exists": True,
                    "path": str(
                        original_path
                    ),  # Return original path to preserve symlink paths
                    "is_file": is_file,
                    "is_directory": is_directory,
                    "is_symlink": is_symlink,
                    "size": file_size,
                    "modified_time": modified_time,
                    "error": None,
                }

            except OSError as os_error:
                # File exists but we can't get metadata (permission issues, etc.)
                logger.warning(
                    f"Could not get metadata for {resolved_path}: {os_error}"
                )
                return {
                    "exists": True,
                    "path": str(
                        original_path
                    ),  # Return original path to preserve symlink paths
                    "is_file": False,
                    "is_directory": False,
                    "is_symlink": is_symlink,
                    "size": None,
                    "modified_time": None,
                    "error": f"Metadata access failed: {str(os_error)}",
                }

        except Exception as e:
            logger.error(f"Error checking file existence for {original_path}: {e}")
            return {
                "exists": False,
                "path": str(original_path if "original_path" in locals() else path),
                "is_file": False,
                "is_directory": False,
                "is_symlink": False,
                "size": None,
                "modified_time": None,
                "error": f"Existence check failed: {str(e)}",
            }

    async def file_exists(self, path: Union[str, Path]) -> bool:
        """
        Simple boolean check for file existence.

        Args:
            path: Path to check for existence

        Returns:
            True if file exists and is accessible, False otherwise
        """
        result = await self.check_file_exists(path)
        return result["exists"] and result["error"] is None

    async def write_file_with_confirmation(
        self,
        path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = "utf-8",
        confirmation_callback: Optional[Callable] = None,
        backup: bool = True,
        atomic: bool = True,
        read_before_write: bool = False,
        user_review_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Write file content with existence check and user confirmation if file exists.

        Args:
            path: Path to write to
            content: Content to write (string or bytes)
            encoding: File encoding (default: 'utf-8')
            confirmation_callback: Optional callback for user confirmation on file overwrite
            backup: Whether to create backup of existing file (default: True)
            atomic: Whether to use atomic write operation (default: True)
            read_before_write: Whether to use read-before-write workflow (default: False)
            user_review_callback: Optional callback for user review when read_before_write=True

        Returns:
            Dictionary with operation result
        """
        original_path = Path(path)
        operation_id = (
            f"write_confirm_{hashlib.md5(str(original_path).encode()).hexdigest()[:8]}"
        )

        try:
            # Check if file exists with detailed information (preserving symlink info)
            file_info = await self.check_file_exists(original_path)

            # If file exists and we have a confirmation callback, ask user
            if file_info["exists"] and confirmation_callback:
                try:
                    confirmation_result = await confirmation_callback(file_info)

                    if not confirmation_result.get("confirmed", False):
                        return {
                            "success": False,
                            "operation_id": operation_id,
                            "path": str(original_path),
                            "reason": "User cancelled operation",
                            "user_decision": confirmation_result,
                            "file_exists": True,
                        }

                    # Handle different user actions
                    user_action = confirmation_result.get("action", "overwrite")
                    if user_action == "backup":
                        backup = True  # Force backup if user requested it
                    elif user_action == "cancel":
                        return {
                            "success": False,
                            "operation_id": operation_id,
                            "path": str(original_path),
                            "reason": confirmation_result.get(
                                "reason", "User cancelled"
                            ),
                            "user_decision": confirmation_result,
                            "file_exists": True,
                        }

                except Exception as e:
                    logger.error(f"Error in confirmation callback: {e}")
                    return {
                        "success": False,
                        "operation_id": operation_id,
                        "path": str(original_path),
                        "reason": f"Confirmation callback error: {str(e)}",
                        "file_exists": file_info["exists"],
                    }

            # Handle symlink replacement if needed
            if file_info["exists"] and file_info.get("is_symlink", False):
                # If overwriting a symlink, remove it first to create a regular file
                logger.debug(
                    f"Removing symlink at {original_path} before writing new file"
                )
                await aiofiles.os.unlink(original_path)

            # Proceed with regular file write
            write_result = await self.write_file(
                path=original_path,
                content=content,
                encoding=encoding,
                backup=backup,
                atomic=atomic,
                read_before_write=read_before_write,
                user_review_callback=user_review_callback,
            )

            # Add confirmation information to result
            write_result.update(
                {
                    "file_existed_before": file_info["exists"],
                    "confirmation_requested": file_info["exists"]
                    and confirmation_callback is not None,
                    "user_confirmed": file_info["exists"]
                    and confirmation_callback is not None,
                }
            )

            return write_result

        except Exception as e:
            logger.error(f"Error in write_file_with_confirmation: {e}")
            return {
                "success": False,
                "operation_id": operation_id,
                "path": str(path),
                "reason": f"Write operation failed: {str(e)}",
                "error": str(e),
            }

    async def read_file(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
        binary: bool = False,
    ) -> Union[str, bytes]:
        """Read file content with security validation and error handling."""

        path = Path(path).resolve()

        # Security validation
        result = await self.security.secure_file_access(str(path), "read")
        if not result["success"]:
            raise FileOperationError(f"Security check failed: {result['error']}")

        try:
            # Check if file exists
            if not await aiofiles.os.path.exists(path):
                raise FileOperationError(f"File not found: {path}")

            # Check file size
            stat_result = await aiofiles.os.stat(path)
            file_size_mb = stat_result.st_size / (1024 * 1024)

            if file_size_mb > self.max_file_size_mb:
                # Use streaming for large files
                return await self._read_large_file(path, binary, encoding)

            # Regular file read
            if binary or self._is_binary_file(path):
                async with aiofiles.open(path, "rb") as f:
                    return await f.read()
            else:
                async with aiofiles.open(path, "r", encoding=encoding) as f:
                    return await f.read()

        except Exception as e:
            raise FileOperationError(f"Failed to read file {path}: {str(e)}")

    async def write_file(
        self,
        path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = "utf-8",
        backup: bool = True,
        atomic: bool = True,
        read_before_write: bool = False,
        user_review_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Write file content with atomic operations, backup, security validation, and approval.

        Args:
            path: Path to write to
            content: Content to write (string or bytes)
            encoding: File encoding (default: 'utf-8')
            backup: Whether to create backup of existing file (default: True)
            atomic: Whether to use atomic write operation (default: True)
            read_before_write: Whether to use read-before-write workflow (default: False)
            user_review_callback: Optional callback for user review when read_before_write=True

        Returns:
            Dictionary with operation result
        """

        path = Path(path).resolve()
        operation_id = f"write_{hashlib.md5(str(path).encode()).hexdigest()[:8]}"

        # If read-before-write is enabled, delegate to that method
        if read_before_write:
            return await self.read_before_write(
                path=path,
                new_content=content,
                encoding=encoding,
                user_review_callback=user_review_callback,
            )

        # Validate content is not None
        if content is None:
            raise FileOperationError(
                f"Content cannot be None for write operation to {path}"
            )

        # Create operation for approval workflow
        file_exists = await aiofiles.os.path.exists(path)
        operation_type = OperationType.FILE_WRITE
        operation_description = (
            f"{'Modify' if file_exists else 'Create'} file: {path.name}"
        )

        # Safely handle content for operation data
        if content is None:
            content_str = ""
            content_length = 0
        elif isinstance(content, str):
            content_str = content[:1000] + "..." if len(content) > 1000 else content
            content_length = len(content)
        else:
            # bytes content
            decoded_content = content.decode(encoding, errors="ignore")
            content_str = (
                decoded_content[:1000] + "..."
                if len(decoded_content) > 1000
                else decoded_content
            )
            content_length = len(content)

        operation = self._create_operation(
            operation_type=operation_type,
            description=operation_description,
            data={
                "path": str(path),
                "content": content_str,
                "content_length": content_length,
                "encoding": encoding,
                "backup": backup,
                "atomic": atomic,
            },
            reversible=file_exists
            and backup,  # Reversible if we have backup capability
        )

        # Request approval if needed
        approved = await self._request_approval_if_needed(operation)
        if not approved:
            raise FileOperationError(
                f"File write operation denied by approval workflow: {path}"
            )

        # Track operation
        self.active_operations[operation_id] = {
            "type": "write",
            "path": path,
            "start_time": datetime.now(),
            "backup_path": None,
            "temp_path": None,
            "approved": approved,
        }

        try:
            # Use shared implementation
            result = await self._execute_write_operation(
                path=path,
                content=content,
                encoding=encoding,
                backup=backup,
                atomic=atomic,
                operation_id=operation_id,
                approved=approved,
            )
            return result

        finally:
            # Clean up temporary files safely
            try:
                if (
                    operation_id in self.active_operations
                    and "temp_path" in self.active_operations[operation_id]
                ):
                    temp_path = self.active_operations[operation_id]["temp_path"]
                    if temp_path and await aiofiles.os.path.exists(temp_path):
                        await aiofiles.os.remove(temp_path)
            except Exception:
                # Ignore cleanup errors - temporary files will be cleaned up eventually
                pass

    async def _execute_write_operation(
        self,
        path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = "utf-8",
        backup: bool = True,
        atomic: bool = True,
        operation_id: str = None,
        approved: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the core file write operation with security, backup, and atomic operations.
        Used by both write_file() and _perform_direct_write() to avoid code duplication.
        """
        try:
            # Security validation
            security_result = await self.security.secure_file_access(
                str(path),
                "write",
                content if isinstance(content, str) else None,
            )
            if not security_result["success"]:
                raise FileOperationError(
                    f"Security check failed: {security_result['error']}"
                )

            # Create backup if file exists and backup is requested
            backup_path = None
            if backup and await aiofiles.os.path.exists(path):
                try:
                    backup_path = await self.create_backup(path)
                    if operation_id:
                        self.active_operations[operation_id][
                            "backup_path"
                        ] = backup_path
                except Exception as e:
                    # Log backup failure but continue with write operation
                    logger.warning(f"Backup creation failed for {path}: {e}")
                    if operation_id:
                        self.active_operations[operation_id]["backup_warning"] = str(e)

            # Atomic write
            if atomic:
                result = await self._atomic_write(path, content, encoding)
            else:
                result = await self._direct_write(path, content, encoding)

            # Update operation tracking if operation_id provided
            if operation_id:
                self.active_operations[operation_id].update(result)
                self.active_operations[operation_id]["success"] = True

            return {
                "success": True,
                "operation_id": operation_id,
                "path": str(path),
                "backup_path": str(backup_path) if backup_path else None,
                "bytes_written": len(
                    content.encode(encoding) if isinstance(content, str) else content
                ),
                "atomic": atomic,
                "approved": approved,
            }

        except Exception as e:
            # Rollback on failure if we have backup info
            if (
                backup
                and operation_id
                and "backup_path" in self.active_operations[operation_id]
            ):
                await self._rollback_operation(operation_id)

            if operation_id:
                self.active_operations[operation_id]["success"] = False
                self.active_operations[operation_id]["error"] = str(e)

            raise FileOperationError(f"Failed to write file {path}: {str(e)}")

    async def _perform_direct_write(
        self,
        path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = "utf-8",
        backup: bool = True,
        atomic: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform direct file write bypassing the main write_file method.
        Used internally by read_before_write to avoid infinite loops.
        """
        path = Path(path).resolve()
        operation_id = f"direct_write_{hashlib.md5(str(path).encode()).hexdigest()[:8]}"

        # Validate content is not None
        if content is None:
            raise FileOperationError(
                f"Content cannot be None for write operation to {path}"
            )

        # Track operation
        self.active_operations[operation_id] = {
            "type": "direct_write",
            "path": path,
            "start_time": datetime.now(),
            "backup_path": None,
            "temp_path": None,
        }

        try:
            # Use shared implementation
            result = await self._execute_write_operation(
                path=path,
                content=content,
                encoding=encoding,
                backup=backup,
                atomic=atomic,
                operation_id=operation_id,
                approved=True,  # Direct writes bypass approval
            )
            return result

        finally:
            # Clean up temporary files safely
            try:
                if (
                    operation_id in self.active_operations
                    and "temp_path" in self.active_operations[operation_id]
                ):
                    temp_path = self.active_operations[operation_id]["temp_path"]
                    if temp_path and await aiofiles.os.path.exists(temp_path):
                        await aiofiles.os.remove(temp_path)
            except Exception:
                # Ignore cleanup errors - temporary files will be cleaned up eventually
                pass

    async def create_backup(self, path: Union[str, Path]) -> Path:
        """Create a timestamped backup of a file."""

        path = Path(path).resolve()

        if not await aiofiles.os.path.exists(path):
            raise FileOperationError(f"Cannot backup non-existent file: {path}")

        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{path.name}.{timestamp}.backup"
        backup_path = self.backup_dir / backup_filename

        # Ensure backup directory exists
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file to backup location
        await self._copy_file(path, backup_path)

        return backup_path

    async def restore_backup(
        self, backup_path: Union[str, Path], target_path: Union[str, Path]
    ) -> bool:
        """Restore a file from backup."""

        backup_path = Path(backup_path)
        target_path = Path(target_path)

        if not await aiofiles.os.path.exists(backup_path):
            raise FileOperationError(f"Backup file not found: {backup_path}")

        # Security validation for target path
        result = await self.security.secure_file_access(str(target_path), "write")
        if not result["success"]:
            raise FileOperationError(f"Security check failed: {result['error']}")

        try:
            await self._copy_file(backup_path, target_path)
            return True
        except Exception as e:
            raise FileOperationError(f"Failed to restore backup: {str(e)}")

    async def create_directory(
        self, path: Union[str, Path], parents: bool = True
    ) -> bool:
        """Create directory with security validation and approval."""

        path = Path(path).resolve()

        # Create operation for approval workflow
        operation = self._create_operation(
            operation_type=OperationType.DIRECTORY_CREATE,
            description=f"Create directory: {path.name}",
            data={"path": str(path), "parents": parents},
            reversible=True,  # Directory creation is reversible (can be deleted)
        )

        # Request approval if needed
        approved = await self._request_approval_if_needed(operation)
        if not approved:
            raise FileOperationError(
                f"Directory creation denied by approval workflow: {path}"
            )

        # Security validation
        perm_operation = PermissionOperation(
            operation_type="file_create", path=str(path)
        )
        validation = await self.security.validate_operation(perm_operation)

        if not validation["allowed"]:
            raise FileOperationError(
                f"Directory creation blocked: {', '.join(validation['reasons'])}"
            )

        try:
            if parents:
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(exist_ok=True)
            return True
        except Exception as e:
            raise FileOperationError(f"Failed to create directory {path}: {str(e)}")

    async def delete_file(
        self, path: Union[str, Path], backup: bool = True
    ) -> Dict[str, Any]:
        """Delete file with optional backup and approval."""

        path = Path(path).resolve()

        # Create operation for approval workflow
        operation = self._create_operation(
            operation_type=OperationType.FILE_DELETE,
            description=f"Delete file: {path.name}",
            data={"path": str(path), "backup": backup},
            reversible=backup,  # Reversible if we create backup
        )

        # Request approval if needed
        approved = await self._request_approval_if_needed(operation)
        if not approved:
            raise FileOperationError(
                f"File delete operation denied by approval workflow: {path}"
            )

        # Security validation
        result = await self.security.secure_file_access(str(path), "delete")
        if not result["success"]:
            raise FileOperationError(f"Security check failed: {result['error']}")

        backup_path = None
        try:
            # Create backup if requested
            if backup and await aiofiles.os.path.exists(path):
                backup_path = await self.create_backup(path)

            # Delete the file
            await aiofiles.os.remove(path)

            return {
                "success": True,
                "path": str(path),
                "backup_path": str(backup_path) if backup_path else None,
                "approved": approved,
            }

        except Exception as e:
            raise FileOperationError(f"Failed to delete file {path}: {str(e)}")

    async def glob_files(self, pattern: str, recursive: bool = True) -> List[Path]:
        """Find files matching glob pattern with security validation."""

        try:
            # Use glob to find matching files
            if recursive:
                matches = glob_module.glob(pattern, recursive=True)
            else:
                matches = glob_module.glob(pattern)

            # Convert to Path objects and validate each
            validated_paths = []
            for match in matches:
                path = Path(match).resolve()

                # Basic security check for each path
                try:
                    result = await self.security.secure_file_access(str(path), "read")
                    if result["success"]:
                        validated_paths.append(path)
                except:
                    # Skip files that fail security validation
                    continue

            return validated_paths

        except Exception as e:
            raise FileOperationError(f"Glob pattern matching failed: {str(e)}")

    async def copy_file(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Copy file with security validation."""

        src = Path(src).resolve()
        dst = Path(dst).resolve()

        # Validate source read access
        read_result = await self.security.secure_file_access(str(src), "read")
        if not read_result["success"]:
            raise FileOperationError(
                f"Source read access denied: {read_result['error']}"
            )

        # Validate destination write access
        write_result = await self.security.secure_file_access(str(dst), "write")
        if not write_result["success"]:
            raise FileOperationError(
                f"Destination write access denied: {write_result['error']}"
            )

        try:
            await self._copy_file(src, dst)
            return True
        except Exception as e:
            raise FileOperationError(f"Failed to copy file: {str(e)}")

    async def move_file(
        self, src: Union[str, Path], dst: Union[str, Path], backup: bool = True
    ) -> Dict[str, Any]:
        """Move file with security validation, approval, and optional backup."""

        src = Path(src).resolve()
        dst = Path(dst).resolve()

        # Create operation for approval workflow
        operation = self._create_operation(
            operation_type=OperationType.FILE_WRITE,  # Move involves writing to new location
            description=f"Move file: {src.name} → {dst.name}",
            data={
                "source_path": str(src),
                "destination_path": str(dst),
                "backup": backup,
            },
            reversible=backup,  # Reversible if we create backup
        )

        # Request approval if needed
        approved = await self._request_approval_if_needed(operation)
        if not approved:
            raise FileOperationError(
                f"File move operation denied by approval workflow: {src} → {dst}"
            )

        # Copy first, then delete source
        await self.copy_file(src, dst)
        delete_result = await self.delete_file(src, backup=backup)

        return {
            "success": True,
            "source": str(src),
            "destination": str(dst),
            "backup_path": delete_result.get("backup_path"),
            "approved": approved,
        }

    async def get_file_info(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive file information."""

        path = Path(path).resolve()

        # Security validation
        result = await self.security.secure_file_access(str(path), "read")
        if not result["success"]:
            raise FileOperationError(f"Security check failed: {result['error']}")

        try:
            if not await aiofiles.os.path.exists(path):
                raise FileOperationError(f"File not found: {path}")

            stat_result = await aiofiles.os.stat(path)

            info = {
                "path": str(path),
                "name": path.name,
                "size": stat_result.st_size,
                "size_mb": stat_result.st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(stat_result.st_mtime),
                "created": datetime.fromtimestamp(stat_result.st_ctime),
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "is_binary": (self._is_binary_file(path) if path.is_file() else False),
                "mime_type": mimetypes.guess_type(str(path))[0],
                "permissions": oct(stat_result.st_mode)[-3:],
            }

            # Add file hash for files
            if (
                path.is_file() and stat_result.st_size < 10 * 1024 * 1024
            ):  # Only for files < 10MB
                content = await self.read_file(path, binary=True)
                info["md5_hash"] = hashlib.md5(content).hexdigest()
                info["sha256_hash"] = hashlib.sha256(content).hexdigest()

            return info

        except Exception as e:
            raise FileOperationError(f"Failed to get file info: {str(e)}")

    async def list_directory(
        self, path: Union[str, Path], recursive: bool = False
    ) -> List[Dict[str, Any]]:
        """List directory contents with file information."""

        path = Path(path).resolve()

        # Security validation
        result = await self.security.secure_file_access(str(path), "read")
        if not result["success"]:
            raise FileOperationError(f"Security check failed: {result['error']}")

        try:
            if not path.is_dir():
                raise FileOperationError(f"Path is not a directory: {path}")

            items = []

            if recursive:
                for item in path.rglob("*"):
                    try:
                        info = await self.get_file_info(item)
                        items.append(info)
                    except:
                        # Skip items that can't be accessed
                        continue
            else:
                for item in path.iterdir():
                    try:
                        info = await self.get_file_info(item)
                        items.append(info)
                    except:
                        # Skip items that can't be accessed
                        continue

            return sorted(items, key=lambda x: (not x["is_dir"], x["name"].lower()))

        except Exception as e:
            raise FileOperationError(f"Failed to list directory: {str(e)}")

    # Private helper methods

    def _is_binary_file(self, path: Path) -> bool:
        """Detect if file is binary."""
        try:
            with open(path, "rb") as f:
                chunk = f.read(8192)
                return b"\x00" in chunk
        except:
            return False

    async def _atomic_write(
        self, path: Path, content: Union[str, bytes], encoding: str
    ) -> Dict[str, Any]:
        """Perform atomic write using temporary file."""

        # Create temporary file in same directory
        temp_path = path.parent / f".{path.name}.tmp.{os.getpid()}"

        try:
            # Write to temporary file
            if isinstance(content, bytes):
                async with aiofiles.open(temp_path, "wb") as f:
                    await f.write(content)
            else:
                async with aiofiles.open(temp_path, "w", encoding=encoding) as f:
                    await f.write(content)

            # Atomic move
            await aiofiles.os.rename(temp_path, path)

            return {
                "method": "atomic",
                "temp_path": temp_path,
            }

        except Exception as e:
            # Clean up temp file on failure
            if await aiofiles.os.path.exists(temp_path):
                await aiofiles.os.remove(temp_path)
            raise e

    async def _direct_write(
        self, path: Path, content: Union[str, bytes], encoding: str
    ) -> Dict[str, Any]:
        """Perform direct write to file."""

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, bytes):
            async with aiofiles.open(path, "wb") as f:
                await f.write(content)
        else:
            async with aiofiles.open(path, "w", encoding=encoding) as f:
                await f.write(content)

        return {
            "method": "direct",
        }

    async def _copy_file(self, src: Path, dst: Path) -> None:
        """Copy file using async operations."""

        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(src, "rb") as src_file:
            async with aiofiles.open(dst, "wb") as dst_file:
                while True:
                    chunk = await src_file.read(self.chunk_size)
                    if not chunk:
                        break
                    await dst_file.write(chunk)

    async def _read_large_file(
        self, path: Path, binary: bool, encoding: str
    ) -> Union[str, bytes]:
        """Read large file in chunks."""

        chunks = []

        if binary:
            async with aiofiles.open(path, "rb") as f:
                while True:
                    chunk = await f.read(self.chunk_size)
                    if not chunk:
                        break
                    chunks.append(chunk)
            return b"".join(chunks)
        else:
            async with aiofiles.open(path, "r", encoding=encoding) as f:
                while True:
                    chunk = await f.read(self.chunk_size)
                    if not chunk:
                        break
                    chunks.append(chunk)
            return "".join(chunks)

    async def _rollback_operation(self, operation_id: str) -> bool:
        """Rollback a failed operation using backup."""

        if operation_id not in self.active_operations:
            return False

        operation = self.active_operations[operation_id]
        backup_path = operation.get("backup_path")
        original_path = operation.get("path")

        if backup_path and original_path and await aiofiles.os.path.exists(backup_path):
            try:
                await self._copy_file(Path(backup_path), Path(original_path))
                return True
            except:
                return False

        return False

    # Git integration methods

    async def git_add(self, path: Union[str, Path]) -> bool:
        """Add file to git staging area."""

        path = Path(path).resolve()

        # Check if we're in a git repository
        try:
            result = await self.security.execute_secure_command(
                ["git", "rev-parse", "--git-dir"], working_dir=str(path.parent)
            )
            if not result["success"]:
                raise FileOperationError("Not in a git repository")
        except:
            raise FileOperationError("Git not available or not in a git repository")

        # Add file to git
        try:
            result = await self.security.execute_secure_command(
                ["git", "add", str(path)], working_dir=str(path.parent)
            )
            return result["success"]
        except Exception as e:
            raise FileOperationError(f"Git add failed: {str(e)}")

    async def git_status(
        self, path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Get git status for path or current directory."""

        working_dir = str(Path(path).parent if path else Path.cwd())

        try:
            result = await self.security.execute_secure_command(
                ["git", "status", "--porcelain"], working_dir=working_dir
            )

            if result["success"]:
                return {
                    "success": True,
                    "status_output": result["stdout"],
                    "changes": (
                        result["stdout"].strip().split("\n")
                        if result["stdout"].strip()
                        else []
                    ),
                }
            else:
                return {
                    "success": False,
                    "error": result["stderr"],
                }
        except Exception as e:
            raise FileOperationError(f"Git status failed: {str(e)}")

    # Context managers and streaming

    @asynccontextmanager
    async def streaming_read(
        self, path: Union[str, Path], chunk_size: Optional[int] = None
    ):
        """Context manager for streaming file reads."""

        path = Path(path).resolve()
        chunk_size = chunk_size or self.chunk_size

        # Security validation
        result = await self.security.secure_file_access(str(path), "read")
        if not result["success"]:
            raise FileOperationError(f"Security check failed: {result['error']}")

        async with aiofiles.open(path, "rb") as f:
            yield f

    async def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a tracked operation."""
        return self.active_operations.get(operation_id)

    def clear_completed_operations(self) -> int:
        """Clear completed operations from tracking."""
        completed = [
            op_id
            for op_id, op in self.active_operations.items()
            if op.get("success") is not None
        ]

        for op_id in completed:
            del self.active_operations[op_id]

        return len(completed)

    # Read-before-write functionality

    async def read_before_write(
        self,
        path: Union[str, Path],
        new_content: Union[str, bytes],
        encoding: str = "utf-8",
        user_review_callback: Optional[Callable] = None,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Read existing file content before write and present to user for review.

        This method implements read-before-write logic by:
        1. Reading current file content if it exists
        2. Presenting both current and new content to user for review
        3. Allowing user to approve, modify, or reject the change

        Args:
            path: Path to the file to be written
            new_content: New content to write to the file
            encoding: File encoding for text files
            user_review_callback: Callback function for user review interface
            max_retries: Maximum number of retry attempts for recoverable errors

        Returns:
            Dict with operation result and user decision
        """
        return await self._read_before_write_with_recovery(
            path=path,
            new_content=new_content,
            encoding=encoding,
            user_review_callback=user_review_callback,
            max_retries=max_retries,
            retry_count=0,
        )

    async def _read_before_write_with_recovery(
        self,
        path: Union[str, Path],
        new_content: Union[str, bytes],
        encoding: str = "utf-8",
        user_review_callback: Optional[Callable] = None,
        max_retries: int = 2,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Internal method with error recovery for read-before-write operations.
        """
        path = Path(path).resolve()
        operation_type = (
            "create" if not await aiofiles.os.path.exists(path) else "modify"
        )

        try:
            # Read current content if file exists
            current_content = None
            file_exists = await aiofiles.os.path.exists(path)

            if file_exists:
                try:
                    current_content = await self.read_file(path, encoding=encoding)
                except Exception as e:
                    # Handle file read error
                    read_error = FileReadError(str(path), e)
                    error_result = self.error_handler.handle_error(
                        read_error, retry_count, max_retries
                    )

                    if error_result.get("should_retry"):
                        return await self._read_before_write_with_recovery(
                            path,
                            new_content,
                            encoding,
                            user_review_callback,
                            max_retries,
                            retry_count + 1,
                        )
                    elif error_result.get("fallback_action") == "regular_write":
                        logger.warning(
                            f"Falling back to regular write for {path} due to read error"
                        )
                        return await self._perform_direct_write(
                            path=path, content=new_content, encoding=encoding
                        )
                    elif error_result.get("should_abort"):
                        return {
                            "success": False,
                            "error": f"Failed to read existing content: {str(e)}",
                            "operation": "read_before_write",
                            "error_details": error_result,
                        }
                    else:
                        # Continue with warning
                        current_content = f"<Error reading file: {str(e)}>"
                        logger.warning(f"Continuing with read error for {path}: {e}")

            # Prepare review data
            review_data = {
                "file_path": str(path),
                "file_exists": file_exists,
                "current_content": current_content,
                "new_content": (
                    new_content
                    if isinstance(new_content, str)
                    else new_content.decode(encoding, errors="ignore")
                ),
                "encoding": encoding,
                "operation": operation_type,
            }

            # Generate diff if both contents exist
            try:
                if current_content and not current_content.startswith(
                    "<Error reading file:"
                ):
                    if isinstance(new_content, str):
                        review_data["diff"] = self._generate_content_diff(
                            current_content, new_content, str(path)
                        )
                    elif isinstance(new_content, bytes):
                        review_data["diff"] = self._generate_content_diff(
                            current_content,
                            new_content.decode(encoding, errors="ignore"),
                            str(path),
                        )
            except Exception as e:
                # Handle diff generation error
                diff_error = DiffGenerationError(str(path), e)
                error_result = self.error_handler.handle_error(
                    diff_error, retry_count, max_retries
                )

                if error_result.get("fallback_action") == "continue_with_warning":
                    logger.warning(f"Continuing without diff for {path}: {e}")
                    review_data["diff"] = f"<Error generating diff: {str(e)}>"
                else:
                    review_data["diff"] = None

            # Present to user for review if callback provided
            user_decision = {"approved": True, "modified_content": None}
            if user_review_callback:
                try:
                    user_decision = await user_review_callback(review_data)
                except Exception as e:
                    # Handle callback error
                    callback_error = CallbackError(str(path), e)
                    error_result = self.error_handler.handle_error(
                        callback_error, retry_count, max_retries
                    )

                    if error_result.get("should_retry"):
                        return await self._read_before_write_with_recovery(
                            path,
                            new_content,
                            encoding,
                            user_review_callback,
                            max_retries,
                            retry_count + 1,
                        )
                    elif error_result.get("fallback_action") == "regular_write":
                        logger.warning(
                            f"Falling back to regular write for {path} due to callback error"
                        )
                        return await self._perform_direct_write(
                            path=path, content=new_content, encoding=encoding
                        )
                    else:
                        return {
                            "success": False,
                            "error": f"User review failed: {str(e)}",
                            "operation": "read_before_write",
                            "error_details": error_result,
                        }

            # Handle user decision - CRITICAL FIX for cancellation
            if user_decision.get("cancelled"):
                # User pressed 'q' to quit - exit immediately without retry
                logger.info(f"🛑 User cancelled file operation for {path}")
                return {
                    "success": False,
                    "error": "User cancelled operation",
                    "operation": "read_before_write",
                    "user_decision": user_decision,
                    "cancelled": True,  # Mark as cancelled for upstream handling
                    "final": True,  # No retries should be attempted
                }
            elif not user_decision.get("approved", False):
                from .read_before_write_errors import UserRejectionError

                rejection_reason = user_decision.get("reason", "User rejected changes")
                rejection_error = UserRejectionError(str(path), rejection_reason)
                error_result = self.error_handler.handle_error(
                    rejection_error, retry_count, max_retries
                )

                return {
                    "success": False,
                    "error": "User rejected the file modification",
                    "operation": "read_before_write",
                    "user_decision": user_decision,
                    "error_details": error_result,
                }

            # Validate content if modified by user
            final_content = user_decision.get("modified_content", new_content)

            # Ensure final_content is not None
            if final_content is None:
                final_content = new_content

            if user_decision.get("modified_content") is not None:
                try:
                    self._validate_content(final_content, str(path))
                except Exception as validation_error:
                    content_error = ContentValidationError(
                        str(path), str(validation_error)
                    )
                    error_result = self.error_handler.handle_error(
                        content_error, retry_count, max_retries
                    )

                    if error_result.get("fallback_action") == "prompt_user":
                        return {
                            "success": False,
                            "error": f"Content validation failed: {str(validation_error)}",
                            "operation": "read_before_write",
                            "error_details": error_result,
                            "requires_user_action": True,
                        }

            # Proceed with write operation - CRITICAL FIX for infinite loop
            try:
                # Skip read_before_write to avoid calling the replaced write_file method again
                # This directly performs the file write without triggering approval again
                write_result = await self._perform_direct_write(
                    path=path,
                    content=final_content,
                    encoding=encoding,
                    backup=True,  # Always create backup in read-before-write mode
                    atomic=True,
                )

                # Add read-before-write specific metadata
                write_result["operation"] = "read_before_write"
                write_result["had_existing_content"] = file_exists
                write_result["user_reviewed"] = bool(user_review_callback)
                write_result["content_modified_by_user"] = (
                    user_decision.get("modified_content") is not None
                )
                write_result["retry_count"] = retry_count

                return write_result

            except Exception as e:
                # Handle write error
                write_error = FileWriteError(str(path), e)
                error_result = self.error_handler.handle_error(
                    write_error, retry_count, max_retries
                )

                if error_result.get("should_retry"):
                    return await self._read_before_write_with_recovery(
                        path,
                        new_content,
                        encoding,
                        user_review_callback,
                        max_retries,
                        retry_count + 1,
                    )
                else:
                    return {
                        "success": False,
                        "error": f"Write operation failed: {str(e)}",
                        "operation": "read_before_write",
                        "error_details": error_result,
                    }

        except Exception as e:
            # Handle any unexpected errors
            logger.error(
                f"Unexpected error in read_before_write for {path}: {e}",
                exc_info=True,
            )
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "operation": "read_before_write",
                "retry_count": retry_count,
            }

    def _validate_content(self, content: Union[str, bytes], file_path: str):
        """
        Validate file content before writing.

        Args:
            content: Content to validate
            file_path: Path to the file for context

        Raises:
            ValueError: If content validation fails
        """
        if content is None:
            raise ValueError("Content cannot be None")

        if isinstance(content, str):
            # Basic text content validation
            if len(content) > 50 * 1024 * 1024:  # 50MB limit
                raise ValueError("Content exceeds maximum size limit (50MB)")

            # Check for suspicious patterns (basic security)
            suspicious_patterns = [
                "#!/usr/bin/env",  # Executable scripts
                "eval(",  # Code evaluation
                "exec(",  # Code execution
                "__import__",  # Dynamic imports
            ]

            content_lower = content.lower()
            for pattern in suspicious_patterns:
                if pattern in content_lower:
                    logger.warning(
                        f"Potentially suspicious content pattern detected: {pattern}"
                    )

        elif isinstance(content, bytes):
            # Basic binary content validation
            if len(content) > 100 * 1024 * 1024:  # 100MB limit for binary
                raise ValueError("Binary content exceeds maximum size limit (100MB)")

        # File-specific validation based on extension
        path = Path(file_path)
        if path.suffix.lower() in [".py", ".js", ".ts", ".sh"]:
            # Basic syntax check for code files could be added here
            pass

    def _generate_content_diff(
        self, current_content: str, new_content: str, file_path: str
    ) -> str:
        """Generate unified diff between current and new content."""
        import difflib

        current_lines = current_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff_lines = list(
            difflib.unified_diff(
                current_lines,
                new_lines,
                fromfile=f"{file_path} (current)",
                tofile=f"{file_path} (new)",
                lineterm="",
            )
        )

        return "".join(diff_lines)

    async def preview_file_modification(
        self,
        path: Union[str, Path],
        new_content: Union[str, bytes],
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        """
        Preview file modification without making changes.

        Args:
            path: Path to the file
            new_content: Proposed new content
            encoding: File encoding for text files

        Returns:
            Dict with preview information including diff
        """
        path = Path(path).resolve()

        # Read current content if file exists
        current_content = None
        file_exists = await aiofiles.os.path.exists(path)

        if file_exists:
            try:
                current_content = await self.read_file(path, encoding=encoding)
            except Exception as e:
                current_content = f"<Error reading file: {str(e)}>"

        # Prepare preview data
        preview = {
            "file_path": str(path),
            "file_exists": file_exists,
            "operation_type": "create" if not file_exists else "modify",
            "current_content": current_content,
            "new_content": (
                new_content
                if isinstance(new_content, str)
                else new_content.decode(encoding, errors="ignore")
            ),
            "current_size": len(current_content) if current_content else 0,
            "new_size": (
                len(new_content)
                if isinstance(new_content, str)
                else len(new_content.decode(encoding, errors="ignore"))
            ),
            "encoding": encoding,
        }

        # Generate diff if both contents exist
        if current_content:
            new_content_str = (
                new_content
                if isinstance(new_content, str)
                else new_content.decode(encoding, errors="ignore")
            )
            preview["diff"] = self._generate_content_diff(
                current_content, new_content_str, str(path)
            )
            preview["has_changes"] = current_content != new_content_str
        else:
            preview["diff"] = None
            preview["has_changes"] = True  # New file is always a change

        return preview

    # Directory awareness methods

    def get_current_working_directory(self) -> Path:
        """Get the current working directory."""
        return Path.cwd()

    async def is_git_repository(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Check if the given path (or current directory) is a Git repository."""

        check_path = (
            Path(path).resolve() if path else self.get_current_working_directory()
        )

        # Check for .git directory
        git_dir = check_path / ".git"
        if git_dir.exists() and git_dir.is_dir():
            return True

        # Check if we're inside a git repository by looking up the directory tree
        current = check_path
        while current != current.parent:  # Stop at filesystem root
            git_dir = current / ".git"
            if git_dir.exists() and git_dir.is_dir():
                return True
            current = current.parent

        return False

    async def get_git_repository_root(
        self, path: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """Get the root directory of the Git repository, if any."""

        check_path = (
            Path(path).resolve() if path else self.get_current_working_directory()
        )

        # Look up the directory tree for .git directory
        current = check_path
        while current != current.parent:  # Stop at filesystem root
            git_dir = current / ".git"
            if git_dir.exists() and git_dir.is_dir():
                return current
            current = current.parent

        return None

    async def get_directory_context(
        self, path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive directory context including working directory and repository status."""

        target_path = (
            Path(path).resolve() if path else self.get_current_working_directory()
        )

        context = {
            "current_working_directory": str(target_path),
            "is_git_repository": await self.is_git_repository(target_path),
            "git_repository_root": None,
            "relative_to_repo_root": None,
        }

        # If it's a git repository, get additional git context
        if context["is_git_repository"]:
            repo_root = await self.get_git_repository_root(target_path)
            if repo_root:
                context["git_repository_root"] = str(repo_root)
                try:
                    context["relative_to_repo_root"] = str(
                        target_path.relative_to(repo_root)
                    )
                except ValueError:
                    # Path is not relative to repo root (shouldn't happen, but just in case)
                    context["relative_to_repo_root"] = str(target_path)

        return context

    async def cleanup(self) -> None:
        """Cleanup resources and temporary files."""

        # Clean up any remaining temporary files
        for operation in self.active_operations.values():
            temp_path = operation.get("temp_path")
            if temp_path and await aiofiles.os.path.exists(temp_path):
                try:
                    await aiofiles.os.remove(temp_path)
                except:
                    pass

        self.active_operations.clear()

    # Additional methods for test compatibility

    async def create_file(
        self,
        path: Union[str, Path],
        content: Union[str, bytes] = "",
        encoding: str = "utf-8",
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a new file with the given content (test compatibility wrapper)."""
        return await self.write_file(
            path=path, content=content, encoding=encoding, **kwargs
        )

    async def modify_file(
        self,
        path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = "utf-8",
        **kwargs,
    ) -> Dict[str, Any]:
        """Modify an existing file with new content (test compatibility wrapper)."""
        return await self.write_file(
            path=path, content=content, encoding=encoding, **kwargs
        )

    async def execute_operation(self, operation: Operation) -> OperationResult:
        """Execute a file system operation (interface compatibility)."""

        try:
            if operation.type == OperationType.FILE_READ:
                result = await self.read_file(operation.data["path"])
                return OperationResult(success=True, data=result)
            elif operation.type == OperationType.FILE_WRITE:
                result = await self.write_file(
                    path=operation.data["path"],
                    content=operation.data["content"],
                    encoding=operation.data.get("encoding", "utf-8"),
                )
                return OperationResult(success=result["success"], data=result)
            elif operation.type == OperationType.FILE_DELETE:
                result = await self.delete_file(
                    path=operation.data["path"],
                    backup=operation.data.get("backup", True),
                )
                return OperationResult(success=result["success"], data=result)
            else:
                return OperationResult(
                    success=False,
                    error=f"Unsupported operation type: {operation.type}",
                )
        except Exception as e:
            return OperationResult(success=False, error=str(e))

    async def preview_operation(self, operation: Operation) -> str:
        """Generate a preview of what the operation will do (interface compatibility)."""
        try:
            if operation.type == OperationType.FILE_READ:
                return f"Read file: {operation.data['path']}"
            elif operation.type == OperationType.FILE_WRITE:
                path = operation.data["path"]
                exists = await aiofiles.os.path.exists(path)
                action = "Update" if exists else "Create"
                content_preview = str(operation.data["content"])[:100]
                if len(content_preview) < len(str(operation.data["content"])):
                    content_preview += "..."
                return f"{action} file: {path}\nContent preview: {content_preview}"
            elif operation.type == OperationType.FILE_DELETE:
                return f"Delete file: {operation.data['path']}"
            else:
                return f"Unknown operation: {operation.type}"
        except Exception as e:
            return f"Preview error: {str(e)}"
