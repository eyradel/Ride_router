import os
import requests
from dotenv import load_dotenv

load_dotenv()

MNOTIFY_API_KEY = os.getenv("MNOTIFY_API_KEY")
SENDER_ID = os.getenv("SENDER_ID", "RideRouter")

MNOTIFY_EMAIL_URL = "https://apps.mnotify.net/emailapi"

def send_verification_email(email: str, token: str):
    if not MNOTIFY_API_KEY:
        raise ValueError("MNOTIFY_API_KEY is not set in environment variables.")
    
    msg = f"""Welcome to RideRouter!

Your verification code is: {token}

Please enter this code on the verification page to activate your account.

If you didn't request this verification, please ignore this email.

Best regards,
RideRouter Team"""
    
    params = {
        "key": MNOTIFY_API_KEY,
        "to": email,
        "msg": msg,
        "sender_id": SENDER_ID,
    }
    try:
        response = requests.get(MNOTIFY_EMAIL_URL, params=params, timeout=10)
        response.raise_for_status()
        # Optionally, check response.json() for delivery status
    except Exception as e:
        # TODO: Add proper logging
        print(f"Failed to send email: {e}") 