from datetime import timedelta
from typing import List, Union, Any
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from . import models, schemas, security
from .database import get_db
from jose import jwt, JWTError
from .core.config import settings
from .core.email import send_verification_email
from .core.sms import send_verification_sms

router = APIRouter()

@router.post("/register", response_model=schemas.User)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)) -> Any:
    """Register a new user"""
    # Check if user already exists
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username is taken
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Check if mobile number is taken
    db_user = db.query(models.User).filter(models.User.mobile_number == user.mobile_number).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mobile number already registered"
        )
    
    # Create new user
    hashed_password = security.get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        mobile_number=user.mobile_number,
        country=user.country,
        company=user.company,
        role=user.role,
        hashed_password=hashed_password,
        is_active=False  # Set to False until verified
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    # Generate verification token
    verification_token = jwt.encode({"sub": str(db_user.id)}, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    # Send verification email
    send_verification_email(db_user.email, verification_token)
    # Send verification SMS
    send_verification_sms(db_user.mobile_number, verification_token)
    return db_user

@router.post("/register/client", response_model=schemas.Client)
def register_client(client: schemas.ClientCreate, db: Session = Depends(get_db)):
    """Register a new client application"""
    client_id, client_secret = security.generate_client_credentials()
    
    db_client = models.Client(
        name=client.name,
        client_id=client_id,
        client_secret=client_secret
    )
    db.add(db_client)
    db.commit()
    db.refresh(db_client)
    
    # Return client with credentials
    return {
        "id": db_client.id,
        "name": db_client.name,
        "client_id": client_id,
        "client_secret": client_secret,
        "is_active": True,
        "created_at": db_client.created_at,
        "updated_at": None
    }

@router.post("/token", response_model=schemas.Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
) -> Any:
    """Get access token for user"""
    user = security.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": security.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@router.get("/users/me", response_model=schemas.User)
def read_users_me(current_user: models.User = Depends(security.get_current_active_user)):
    if not isinstance(current_user, models.User):
        raise HTTPException(status_code=403, detail="This endpoint is for users only")
    return current_user

@router.get("/users", response_model=List[schemas.User])
def read_users(
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    if not isinstance(current_user, models.User) or not current_user.is_superuser:
        raise HTTPException(status_code=400, detail="Not enough permissions")
    users = db.query(models.User).offset(skip).limit(limit).all()
    return users

@router.put("/users/me", response_model=schemas.User)
def update_user_me(
    user_update: schemas.UserUpdate,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update current user's information"""
    if not isinstance(current_user, models.User):
        raise HTTPException(status_code=403, detail="This endpoint is for users only")
    
    # Check if email is being changed and if it's already taken
    if user_update.email is not None and user_update.email != current_user.email:
        existing_user = db.query(models.User).filter(models.User.email == user_update.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        current_user.email = user_update.email
    
    # Check if username is being changed and if it's already taken
    if user_update.username is not None and user_update.username != current_user.username:
        existing_user = db.query(models.User).filter(models.User.username == user_update.username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        current_user.username = user_update.username
    
    # Check if mobile number is being changed and if it's already taken
    if user_update.mobile_number is not None and user_update.mobile_number != current_user.mobile_number:
        existing_user = db.query(models.User).filter(models.User.mobile_number == user_update.mobile_number).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mobile number already registered"
            )
        current_user.mobile_number = user_update.mobile_number
    
    # Update other fields
    if user_update.full_name is not None:
        current_user.full_name = user_update.full_name
    if user_update.country is not None:
        current_user.country = user_update.country
    if user_update.company is not None:
        current_user.company = user_update.company
    if user_update.role is not None:
        current_user.role = user_update.role
    if user_update.password is not None:
        current_user.hashed_password = security.get_password_hash(user_update.password)
    
    db.commit()
    db.refresh(current_user)
    return current_user

@router.delete("/users/me", response_model=schemas.User)
def delete_user_me(
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    if not isinstance(current_user, models.User):
        raise HTTPException(status_code=403, detail="This endpoint is for users only")
        
    db.delete(current_user)
    db.commit()
    return current_user

@router.delete("/users/{user_id}", response_model=schemas.User)
def delete_user(
    user_id: int,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    if not isinstance(current_user, models.User) or not current_user.is_superuser:
        raise HTTPException(status_code=400, detail="Not enough permissions")
    
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own admin account")
    
    db.delete(user)
    db.commit()
    return user

@router.get("/users/{user_id}", response_model=schemas.User)
def read_user_by_id(
    user_id: int,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific user by ID (admin only)"""
    if not isinstance(current_user, models.User) or not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to view user details"
        )
    
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    return user

@router.put("/users/{user_id}/make-admin", response_model=schemas.User)
def make_user_admin(
    user_id: int,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Make a user an admin (superuser only)"""
    if not isinstance(current_user, models.User) or not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only superusers can make other users admin"
        )
    
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    user.is_superuser = True
    db.commit()
    db.refresh(user)
    return user

@router.put("/users/{user_id}", response_model=schemas.User)
def update_user(
    user_id: int,
    user_update: schemas.UserUpdate,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update a user's information (admin only)"""
    if not isinstance(current_user, models.User) or not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to update user"
        )
    
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    # Check if email is being changed and if it's already taken
    if user_update.email is not None and user_update.email != user.email:
        existing_user = db.query(models.User).filter(models.User.email == user_update.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        user.email = user_update.email
    
    # Check if username is being changed and if it's already taken
    if user_update.username is not None and user_update.username != user.username:
        existing_user = db.query(models.User).filter(models.User.username == user_update.username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        user.username = user_update.username
    
    # Check if mobile number is being changed and if it's already taken
    if user_update.mobile_number is not None and user_update.mobile_number != user.mobile_number:
        existing_user = db.query(models.User).filter(models.User.mobile_number == user_update.mobile_number).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mobile number already registered"
            )
        user.mobile_number = user_update.mobile_number
    
    # Update other fields
    if user_update.full_name is not None:
        user.full_name = user_update.full_name
    if user_update.country is not None:
        user.country = user_update.country
    if user_update.company is not None:
        user.company = user_update.company
    if user_update.role is not None:
        user.role = user_update.role
    if user_update.password is not None:
        user.hashed_password = security.get_password_hash(user_update.password)
    
    db.commit()
    db.refresh(user)
    return user

@router.get("/verify")
def verify_email(token: str, db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = int(payload.get("sub"))
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    user = db.query(models.User).get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = True
    db.commit()
    return {"msg": "Account verified!"} 