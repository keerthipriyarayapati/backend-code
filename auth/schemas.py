"""
Pydantic schemas for authentication, users, and tokens.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    email: str = Field(..., description="User corporate email address")
    password: str = Field(..., description="User password")


class UserResponse(BaseModel):
    id: int
    full_name: str
    email: str
    role: str
    branch_id: Optional[str] = None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenPayload(BaseModel):
    sub: str
    email: str
    role: str
    branch_id: Optional[str] = None
    full_name: Optional[str] = None
    exp: int
