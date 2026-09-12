"""
FastAPI dependencies for authentication and role-based authorization.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status, Header, Request
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import User
from auth.security import decode_access_token


def get_token_from_request(
    request: Request,
    authorization: Optional[str] = Header(None)
) -> Optional[str]:
    """Extracts Bearer token from Authorization header or cookie."""
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:].strip()
    # Also check access_token cookie if present
    cookie_token = request.cookies.get("access_token")
    if cookie_token:
        return cookie_token
    return None


def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Returns the authenticated User model if a valid token is provided, otherwise None.
    """
    token = get_token_from_request(request, authorization)
    if not token:
        return None

    payload = decode_access_token(token)
    if not payload:
        return None

    user_id_str = payload.get("sub")
    if not user_id_str or not str(user_id_str).isdigit():
        return None

    user = db.query(User).filter(User.id == int(user_id_str)).first()
    if not user or not user.is_active:
        return None

    return user


def require_authenticated_user(
    current_user: Optional[User] = Depends(get_current_user)
) -> User:
    """
    Requires any authenticated user (BANK_EMPLOYEE or BANK_MANAGER).
    Raises HTTP 401 Unauthorized if token is missing or invalid.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return current_user


def require_role(required_role: str):
    """
    Requires the authenticated user to possess a specific role.
    Raises HTTP 403 Forbidden if user role does not match.
    """
    def role_checker(current_user: User = Depends(require_authenticated_user)) -> User:
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Manager access required. This action requires the '{required_role}' role."
            )
        return current_user
    return role_checker


# Dedicated dependency for Manager Analytics
require_manager = require_role("BANK_MANAGER")
