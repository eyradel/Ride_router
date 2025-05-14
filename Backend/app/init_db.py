from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
import os
from dotenv import load_dotenv
from .database import Base, engine
from .models import User, Client, OptimizationResult, OptimizationHistoryView, UserRole

def init_db():
    """Initialize the database and create all tables"""
    try:
        # Drop all existing tables
        Base.metadata.drop_all(bind=engine)
        print("Dropped all existing tables")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")
        
        # Create initial admin user if needed
        from .create_superuser import create_superuser
        create_superuser(
            email="admin@example.com",
            username="admin",
            password="admin123",  # Change this in production!
            full_name="System Administrator",
            mobile_number="+1234567890",
            country="Ghana",
            company="System"
        )
        print("Initial admin user created")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise

if __name__ == "__main__":
    init_db() 