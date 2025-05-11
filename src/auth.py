from datetime import timedelta
from typing import List, Union
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from . import models, schemas, security
from .database import get_db

router = APIRouter()

@router.post("/register", response_model=schemas.User)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Check if this is the first user
    is_first_user = db.query(models.User).first() is None
    
    # Create new user
    hashed_password = security.get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        full_name=user.full_name,
        is_superuser=is_first_user  # Make first user a superuser
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    if is_first_user:
        print(f"First user {user.username} has been created as a superuser!")
    
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

@router.post("/token")
async def login_for_access_token(
    grant_type: str = Form(...),
    username: str = Form(None),
    password: str = Form(None),
    client_id: str = Form(None),
    client_secret: str = Form(None),
    db: Session = Depends(get_db)
):
    if grant_type == "password":
        # User authentication
        if not username or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username and password required for password grant type"
            )
            
        user = db.query(models.User).filter(models.User.username == username).first()
        if not user or not security.verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = security.create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        
    elif grant_type == "client_credentials":
        # Client authentication
        if not client_id or not client_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Client ID and client secret required for client_credentials grant type"
            )
            
        client = db.query(models.Client).filter(
            models.Client.client_id == client_id,
            models.Client.client_secret == client_secret
        ).first()
        
        if not client or not client.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid client credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = security.create_access_token(
            data={"client_id": client.client_id}, expires_delta=access_token_expires
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid grant type. Supported types: password, client_credentials"
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
    if not isinstance(current_user, models.User):
        raise HTTPException(status_code=403, detail="This endpoint is for users only")
        
    if user_update.email is not None:
        current_user.email = user_update.email
    if user_update.username is not None:
        current_user.username = user_update.username
    if user_update.full_name is not None:
        current_user.full_name = user_update.full_name
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