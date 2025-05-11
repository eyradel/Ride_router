from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    optimization_results = relationship("OptimizationResult", back_populates="user")
    optimization_history_views = relationship("OptimizationHistoryView", back_populates="user")

class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    client_id = Column(String, unique=True, index=True)
    client_secret = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class OptimizationResult(Base):
    __tablename__ = "optimization_results"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    total_routes = Column(Integer)
    total_passengers = Column(Integer)
    total_distance = Column(Float)
    total_cost = Column(Float)
    average_route_distance = Column(Float)
    average_route_cost = Column(Float)
    average_passengers_per_route = Column(Float)
    cost_per_passenger = Column(Float)
    routes = Column(JSON)  # Store routes as JSON
    grid_size = Column(Integer)
    sigma = Column(Float)
    learning_rate = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="optimization_results")
    history_views = relationship("OptimizationHistoryView", back_populates="optimization_result")

class OptimizationHistoryView(Base):
    __tablename__ = "optimization_history_views"

    id = Column(Integer, primary_key=True, index=True)
    optimization_result_id = Column(Integer, ForeignKey("optimization_results.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    view_type = Column(String)  # 'list' or 'detail'
    filters_applied = Column(JSON, nullable=True)
    viewed_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    optimization_result = relationship("OptimizationResult", back_populates="history_views")
    user = relationship("User", back_populates="optimization_history_views") 