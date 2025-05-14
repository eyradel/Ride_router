from app.database import SessionLocal
from app.models import User, UserRole
from app.security import get_password_hash
import sys

def create_superuser(
    email: str,
    username: str,
    password: str,
    full_name: str,
    mobile_number: str,
    country: str,
    company: str
):
    db = SessionLocal()
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(
            (User.email == email) | (User.username == username) | (User.mobile_number == mobile_number)
        ).first()
        
        if existing_user:
            print(f"User with email {email}, username {username}, or mobile number {mobile_number} already exists")
            return
        
        # Create superuser
        hashed_password = get_password_hash(password)
        superuser = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
            full_name=full_name,
            mobile_number=mobile_number,
            country=country,
            company=company,
            role=UserRole.ADMIN,
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
    if len(sys.argv) != 8:
        print("Usage: python create_superuser.py <email> <username> <password> <full_name> <mobile_number> <country> <company>")
        sys.exit(1)
    
    create_superuser(
        email=sys.argv[1],
        username=sys.argv[2],
        password=sys.argv[3],
        full_name=sys.argv[4],
        mobile_number=sys.argv[5],
        country=sys.argv[6],
        company=sys.argv[7]
    ) 