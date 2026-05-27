"""Claude subscription OAuth credential management.

Loads bearer tokens from ~/.claude/.credentials.json (shared with Claude Code)
and handles token refresh when expired.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

CREDENTIALS_DEFAULT_PATH = Path.home() / ".claude" / ".credentials.json"
TOKEN_REFRESH_URL = "https://console.anthropic.com/v1/oauth/token"
EXPIRY_BUFFER_SECONDS = 60


@dataclass
class ClaudeCredentials:
    access_token: str
    refresh_token: str
    expires_at_ms: int
    scopes: list
    subscription_type: str
    rate_limit_tier: str

    @classmethod
    def from_dict(cls, data: dict) -> "ClaudeCredentials":
        oauth = data["claudeAiOauth"]
        return cls(
            access_token=oauth["accessToken"],
            refresh_token=oauth["refreshToken"],
            expires_at_ms=oauth["expiresAt"],
            scopes=oauth["scopes"],
            subscription_type=oauth["subscriptionType"],
            rate_limit_tier=oauth["rateLimitTier"],
        )

    @property
    def is_expired(self) -> bool:
        now_ms = int(time.time() * 1000)
        return now_ms >= (self.expires_at_ms - EXPIRY_BUFFER_SECONDS * 1000)


def load_claude_credentials(path: Optional[str] = None) -> Optional[ClaudeCredentials]:
    creds_path = Path(path) if path else CREDENTIALS_DEFAULT_PATH
    try:
        if not creds_path.exists():
            logger.debug(f"Claude credentials file not found: {creds_path}")
            return None
        data = json.loads(creds_path.read_text())
        return ClaudeCredentials.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse Claude credentials: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error loading Claude credentials: {e}")
        return None


async def refresh_claude_token(
    creds: ClaudeCredentials,
    credentials_path: Optional[str] = None,
) -> Optional[ClaudeCredentials]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                TOKEN_REFRESH_URL,
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": creds.refresh_token,
                },
                headers={"Content-Type": "application/json"},
                timeout=15.0,
            )

        if response.status_code != 200:
            logger.warning(f"Token refresh failed: HTTP {response.status_code}")
            return None

        token_data = response.json()
        new_access = token_data["access_token"]
        new_refresh = token_data.get("refresh_token", creds.refresh_token)
        expires_in = token_data.get("expires_in", 3600)
        new_expires_ms = int((time.time() + expires_in) * 1000)

        new_creds = ClaudeCredentials(
            access_token=new_access,
            refresh_token=new_refresh,
            expires_at_ms=new_expires_ms,
            scopes=creds.scopes,
            subscription_type=creds.subscription_type,
            rate_limit_tier=creds.rate_limit_tier,
        )

        if credentials_path:
            _save_credentials(new_creds, credentials_path)

        return new_creds

    except Exception as e:
        logger.warning(f"Token refresh error: {e}")
        return None


def _save_credentials(creds: ClaudeCredentials, path: str) -> None:
    creds_path = Path(path)
    try:
        existing = json.loads(creds_path.read_text()) if creds_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        existing = {}

    existing["claudeAiOauth"] = {
        "accessToken": creds.access_token,
        "refreshToken": creds.refresh_token,
        "expiresAt": creds.expires_at_ms,
        "scopes": creds.scopes,
        "subscriptionType": creds.subscription_type,
        "rateLimitTier": creds.rate_limit_tier,
    }

    creds_path.write_text(json.dumps(existing, indent=2))
    logger.debug(f"Saved refreshed credentials to {path}")
