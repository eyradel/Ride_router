# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY main.py .
COPY app/ app/
COPY setup_database.py .
COPY init_database.py .
COPY create_superuser.py .
COPY .python-version .

# Create and set up data directory
RUN mkdir -p /app/data

# Create virtual environment and install dependencies using uv
RUN uv venv
RUN uv pip install --no-cache .
RUN uv init_db.py
# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "localhost", "--port", "8000"] 