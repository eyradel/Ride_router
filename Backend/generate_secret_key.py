import secrets
import base64

def generate_secret_key():
    # Generate 32 random bytes and encode them in base64
    random_bytes = secrets.token_bytes(32)
    secret_key = base64.b64encode(random_bytes).decode('utf-8')
    print("\nGenerated SECRET_KEY:")
    print("-" * 50)
    print(secret_key)
    print("-" * 50)
    print("\nAdd this to your .env file as:")
    print(f"SECRET_KEY={secret_key}")

if __name__ == "__main__":
    generate_secret_key() 