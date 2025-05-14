FROM python:3.10

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

# Set default environment variables
ENV DATABASE_URL="postgresql://postgres:postgres@db:5432/ride"
ENV PYTHONPATH=/app

# Create virtual environment and install dependencies
RUN uv venv
RUN . .venv/bin/activate && uv pip install --no-cache .

# Create startup script
RUN echo '#!/bin/bash\n\
. .venv/bin/activate\n\
python init_database.py\n\
uvicorn main:app --host 0.0.0.0 --port 8000' > /app/start.sh && \
chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["/app/start.sh"] 