"""API authentication using API keys."""
from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException, status


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """Verify API key from X-API-Key header.

    Args:
        x_api_key: API key from request header

    Returns:
        The validated API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    # Get expected API key from environment
    expected_key = os.environ.get("AI_TRADER_API_KEY")

    # If no key is configured, generate a warning but allow access in development
    if expected_key is None:
        # In production, this should be a hard error
        # For now, we'll log a warning
        import logging
        logging.getLogger("ai_trader.auth").warning(
            "AI_TRADER_API_KEY not configured - authentication disabled! "
            "Set AI_TRADER_API_KEY environment variable for production."
        )
        return "dev-mode-no-auth"

    # Check if API key was provided
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return x_api_key


def generate_api_key() -> str:
    """Generate a cryptographically secure API key.

    Returns:
        A 32-byte hex string (64 characters)
    """
    return secrets.token_hex(32)


# Example usage and key generation
if __name__ == "__main__":
    print("Generated API Key:")
    print(generate_api_key())
    print("\nAdd to your .env file:")
    print(f"AI_TRADER_API_KEY={generate_api_key()}")
