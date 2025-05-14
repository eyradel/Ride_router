# Ride Router Backend

A FastAPI-based backend service for optimizing staff transport routes. The application provides route optimization, staff management, and real-time tracking capabilities.

## Features

- Staff transport route optimization
- User authentication and authorization
- Staff management system
- Route history and analytics
- Real-time route tracking
- API documentation with Swagger UI

## Quick Start with Docker

1. Clone the repository:
```bash
git clone <repository-url>
cd Ride/Backend
```

2. Create a `.env` file:
```env
DATABASE_URL=postgresql://rider:rider@db:5432/ride
POSTGRES_USER=rider
POSTGRES_PASSWORD=rider
POSTGRES_DB=ride
SECRET_KEY=your-secret-key-here
GOOGLE_MAPS_API_KEY=your-google-maps-api-key
```

3. Start the application:
```bash
docker-compose up --build
```

4. Access the application:
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- pgAdmin: http://localhost:5050

## Database Management with pgAdmin

1. Access pgAdmin:
   - Open http://localhost:5050 in your browser
   - Login with:
     - Email: admin@admin.com
     - Password: admin

2. Add a new server:
   - Right-click on "Servers" → "Register" → "Server"
   - General tab:
     - Name: Ride DB
   - Connection tab:
     - Host: db
     - Port: 5432
     - Database: ride
     - Username: rider
     - Password: rider
   - Save the connection

3. Browse the database:
   - Expand "Servers" → "Ride DB" → "Databases" → "ride"
   - You can now view and manage tables, run queries, etc.

## Docker Commands

### Start the Application
```bash
docker-compose up --build
```

### Stop the Application
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f
```

### Rebuild and Start
```bash
docker-compose up --build --force-recreate
```

### Database Commands
```bash
# Connect to database
docker-compose exec db psql -U rider -d ride

# List databases
docker-compose exec db psql -U rider -l
```

## API Endpoints

- `/auth/*` - Authentication endpoints (login, register, etc.)
- `/optimize` - Route optimization endpoints
- `/users` - User management endpoints
- `/optimize/history` - Route history endpoints

## Development

### Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Git

### Environment Variables
Required environment variables:
- `DATABASE_URL`: PostgreSQL connection string (default: postgresql://rider:rider@db:5432/ride)
- `POSTGRES_USER`: Database user (default: rider)
- `POSTGRES_PASSWORD`: Database password (default: rider)
- `POSTGRES_DB`: Database name (default: ride)
- `SECRET_KEY`: JWT token signing key
- `GOOGLE_MAPS_API_KEY`: Google Maps API key for route optimization

Optional environment variables:
- `MNOTIFY_API_KEY`: For SMS notifications
- `SENDER_ID`: SMS sender ID

## Troubleshooting

1. Database Connection Issues
   - Check database logs: `docker-compose logs db`
   - Ensure environment variables match in .env and docker-compose.yml
   - Verify database is running: `docker-compose ps`
   - If pgAdmin can't connect, verify the host is set to "db" (not localhost)

2. Port Conflicts
   - Check if ports 8000, 5432, or 5050 are in use
   - Modify ports in docker-compose.yml if needed

3. Application Issues
   - Check logs: `docker-compose logs -f`
   - Verify environment variables
   - Ensure all required services are running
   - Try rebuilding: `docker-compose up --build --force-recreate`
