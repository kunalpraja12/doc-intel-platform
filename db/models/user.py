"""User model placeholder.

TODO: Add authentication, role, and tenant metadata if multi-user access is required.
"""

from sqlalchemy import Column, Integer, String

from db.session import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    full_name = Column(String, nullable=False)
