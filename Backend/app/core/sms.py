import os
import requests
from dotenv import load_dotenv

load_dotenv()

MNOTIFY_API_KEY = os.getenv("MNOTIFY_API_KEY")
SENDER_ID = os.getenv("SENDER_ID", "RideRouter")

MNOTIFY_SMS_URL = "https://apps.mnotify.net/smsapi"

def send_verification_sms(mobile_number: str, token: str):
    if not MNOTIFY_API_KEY:
        raise ValueError("MNOTIFY_API_KEY is not set in environment variables.")
    
    msg = f"Welcome to RideRouter! Your verification code is: {token}. Enter this code on the verification page to activate your account."
    
    params = {
        "key": MNOTIFY_API_KEY,
        "to": mobile_number,
        "msg": msg,
        "sender_id": SENDER_ID,
    }
    try:
        response = requests.get(MNOTIFY_SMS_URL, params=params, timeout=10)
        response.raise_for_status()
    except Exception as e:
        # TODO: Add proper logging
        print(f"Failed to send SMS: {e}") 