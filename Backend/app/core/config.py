import os
from typing import Optional

class Settings:
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
    ALGORITHM: str = "HS256"  # Algorithm for JWT signing
    
    # SMS/Email settings
    MNOTIFY_API_KEY: str = os.getenv("MNOTIFY_API_KEY")
    SENDER_ID: str = os.getenv("SENDER_ID", "RideRouter")

settings = Settings() 