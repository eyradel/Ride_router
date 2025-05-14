from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import secrets
import string
from . import models, schemas
from .database import get_db
from .core.config import settings

# Security configuration
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def generate_client_credentials():
    """Generate a random client ID and secret"""
    client_id = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
    client_secret = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(64))
    return client_id, client_secret

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        client_id: str = payload.get("client_id")
        
        if username is None and client_id is None:
            raise credentials_exception
            
        token_data = schemas.TokenData(username=username, client_id=client_id)
    except JWTError:
        raise credentials_exception
    
    if token_data.username:
        user = db.query(models.User).filter(models.User.username == token_data.username).first()
        if user is None:
            raise credentials_exception
        return user
    elif token_data.client_id:
        client = db.query(models.Client).filter(models.Client.client_id == token_data.client_id).first()
        if client is None or not client.is_active:
            raise credentials_exception
        return client
    
    raise credentials_exception

def get_current_active_user(current_user = Depends(get_current_user)):
    if isinstance(current_user, models.User) and not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    elif isinstance(current_user, models.Client) and not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive client")
    return current_user

def authenticate_user(db: Session, username: str, password: str):
    """Authenticate a user by username and password"""
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user 