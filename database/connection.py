"""
SQLAlchemy Database Connection & Session Management for SQLite.
Provides thread-safe session factory, FastAPI dependency injection, and automatic directory initialization.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger("DatabaseConnection")

# Resolve project root dynamically (no hardcoded absolute paths)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "loan_agent.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# Create engine with check_same_thread=False for FastAPI compatibility
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initializes database directory, creates tables, and seeds initial data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initializing SQLite database at: {DB_PATH}")
    
    # Import models to register metadata
    import database.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")

    # Auto-migrate documents table schema if needed
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            res = conn.execute(text("PRAGMA table_info(documents)")).fetchall()
            existing_cols = {row[1] for row in res}
            
            if "is_active" not in existing_cols:
                conn.execute(text("ALTER TABLE documents ADD COLUMN is_active BOOLEAN DEFAULT 1"))
                conn.commit()
                logger.info("Migrated documents table: added is_active column.")
            if "ocr_success" not in existing_cols:
                conn.execute(text("ALTER TABLE documents ADD COLUMN ocr_success BOOLEAN DEFAULT 0"))
                conn.commit()
                logger.info("Migrated documents table: added ocr_success column.")
            if "extraction_error" not in existing_cols:
                conn.execute(text("ALTER TABLE documents ADD COLUMN extraction_error VARCHAR(100)"))
                conn.commit()
                logger.info("Migrated documents table: added extraction_error column.")

            # Migrate loan_applications table schema
            app_res = conn.execute(text("PRAGMA table_info(loan_applications)")).fetchall()
            app_cols = {row[1] for row in app_res}
            if "employee_id" not in app_cols:
                conn.execute(text("ALTER TABLE loan_applications ADD COLUMN employee_id VARCHAR(100)"))
                conn.commit()
                logger.info("Migrated loan_applications table: added employee_id column.")
            if "branch_id" not in app_cols:
                conn.execute(text("ALTER TABLE loan_applications ADD COLUMN branch_id VARCHAR(100)"))
                conn.commit()
                logger.info("Migrated loan_applications table: added branch_id column.")
            if "risk_level" not in app_cols:
                conn.execute(text("ALTER TABLE loan_applications ADD COLUMN risk_level VARCHAR(50)"))
                conn.commit()
                logger.info("Migrated loan_applications table: added risk_level column.")
            if "completed_at" not in app_cols:
                conn.execute(text("ALTER TABLE loan_applications ADD COLUMN completed_at DATETIME"))
                conn.commit()
                logger.info("Migrated loan_applications table: added completed_at column.")
            if "processing_time" not in app_cols:
                conn.execute(text("ALTER TABLE loan_applications ADD COLUMN processing_time FLOAT"))
                conn.commit()
                logger.info("Migrated loan_applications table: added processing_time column.")
    except Exception as e:
        logger.warning(f"Column migration check encountered: {e}")

    # Seed document requirements, loan policies, and default users
    try:
        from database.seed import seed_document_requirements
        from policy_kb.loader import seed_loan_policies
        from database.models import User
        from auth.security import get_password_hash

        db = SessionLocal()
        try:
            seed_document_requirements(db)
            seed_loan_policies(db)

            # Seed default BANK_EMPLOYEE and BANK_MANAGER if not present
            def_users = [
                {
                    "email": "employee@bank.com",
                    "full_name": "John Employee",
                    "role": "BANK_EMPLOYEE",
                    "branch_id": "BR-MUMBAI-01",
                    "password": "Password@123"
                },
                {
                    "email": "manager@bank.com",
                    "full_name": "Jane Manager",
                    "role": "BANK_MANAGER",
                    "branch_id": "BR-MUMBAI-01",
                    "password": "Password@123"
                }
            ]
            for u_data in def_users:
                existing_u = db.query(User).filter(User.email == u_data["email"]).first()
                if not existing_u:
                    u = User(
                        email=u_data["email"],
                        full_name=u_data["full_name"],
                        role=u_data["role"],
                        branch_id=u_data["branch_id"],
                        password_hash=get_password_hash(u_data["password"]),
                        is_active=True
                    )
                    db.add(u)
            db.commit()
            logger.info("Default bank employee and manager accounts ensured.")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error seeding database requirements, policies, or users: {e}", exc_info=True)
