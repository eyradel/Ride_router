from app.create_superuser import create_superuser

if __name__ == "__main__":
    # Create a superuser with predefined values
    create_superuser(
        email="eyramdela14@gmail.com",
        username="eyram",
        password="admin246",
        full_name="AI Engineer",
        mobile_number="0501550969",
        country="Ghana",
        company="4th-ir"
    ) 