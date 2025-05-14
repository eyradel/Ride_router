import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError, OperationalError
from dotenv import load_dotenv

def setup_database():
    """Set up the database and create all required tables"""
    # Load environment variables
    load_dotenv()
    
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Error: DATABASE_URL not found in .env file")
        print("Please create a .env file with the following content:")
        print("DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ride_router")
        sys.exit(1)
    
    try:
        # Create engine
        engine = create_engine(database_url)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("Successfully connected to database")
        
        # Import models and create tables
        from app.models import Base
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")
        
        # Create initial admin user
        from app.create_superuser import create_superuser
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
        
    except OperationalError as e:
        print("\nError: Could not connect to database")
        print("Please check that:")
        print("1. PostgreSQL is installed and running")
        print("2. The database 'ride_router' exists")
        print("3. The username and password in DATABASE_URL are correct")
        print("\nTo create the database, run:")
        print("createdb ride_router")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    print("Setting up database...")
    setup_database()
    print("\nDatabase setup complete!")
    print("\nYou can now start the server with:")
    print("uvicorn app.api:app --reload") 