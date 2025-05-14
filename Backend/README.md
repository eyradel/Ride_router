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
DATABASE_URL=postgresql://postgres:postgres@db:5432/ride
SECRET_KEY=your-secret-key-here
GOOGLE_MAPS_API_KEY=your-google-maps-api-key
```

3. Run with Docker Compose:
```bash
docker-compose up --build
```

4. Access the application:
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

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
- `DATABASE_URL`: PostgreSQL connection string
- `SECRET_KEY`: JWT token signing key
- `GOOGLE_MAPS_API_KEY`: Google Maps API key for route optimization

Optional environment variables:
- `MNOTIFY_API_KEY`: For SMS notifications
- `SENDER_ID`: SMS sender ID

### Database Access
```bash
docker-compose exec db psql -U postgres -d ride
```

## Troubleshooting

1. Port Conflicts
   - Check if ports 8000 or 5432 are in use
   - Modify ports in docker-compose.yml if needed

2. Database Connection
   - Verify DATABASE_URL in .env
   - Check database container status: `docker-compose ps`

3. Application Issues
   - Check logs: `docker-compose logs -f`
   - Verify environment variables
   - Ensure all required services are running
