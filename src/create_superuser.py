from src.database import SessionLocal
from src.models import User
from src.security import get_password_hash
import sys

def create_superuser(email: str, username: str, password: str, full_name: str):
    db = SessionLocal()
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(
            (User.email == email) | (User.username == username)
        ).first()
        
        if existing_user:
            print(f"User with email {email} or username {username} already exists")
            return
        
        # Create superuser
        hashed_password = get_password_hash(password)
        superuser = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
            full_name=full_name,
            is_active=True,
            is_superuser=True
        )
        
        db.add(superuser)
        db.commit()
        print(f"Superuser {username} created successfully!")
        
    except Exception as e:
        print(f"Error creating superuser: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python create_superuser.py <email> <username> <password> <full_name>")
        sys.exit(1)
    
    email = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]
    full_name = sys.argv[4]
    
    create_superuser(email, username, password, full_name) 