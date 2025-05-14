from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from typing import List, Optional, Dict, Any, Union, TypeVar, Generic
import pandas as pd
import json
from app.route_optimizer import StaffTransportOptimizer
import os
from dotenv import load_dotenv
import logging
import traceback
from datetime import datetime
import sys
from app import auth
from app.database import engine, Base
from sqlalchemy.orm import Session
from app import models, schemas, security
from app.database import get_db

# Create database tables
Base.metadata.create_all(bind=engine)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Verify environment variables
required_env_vars = ["GOOGLE_MAPS_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Route Optimizer API",
    description="API for optimizing staff transport routes",
    version="1.0.0"
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure this based on your deployment
)

# Add CORS middleware with security headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Add debug logging for OpenAPI schema
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup")
    logger.info(f"OpenAPI schema URL: {app.openapi_url}")
    logger.info(f"Docs URL: {app.docs_url}")
    logger.info(f"Redoc URL: {app.redoc_url}")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_msg = f"[{request_id}] Unhandled exception: {str(exc)}"
    logger.error(error_msg)
    logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id}
    )

# Initialize the optimizer
try:
    google_maps_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not google_maps_key:
        raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables")
    optimizer = StaffTransportOptimizer(google_maps_key=google_maps_key)
    logger.info("Optimizer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize optimizer: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

T = TypeVar('T')

class OptimizationParams(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "grid_size": 3,
                "sigma": 1.0,
                "learning_rate": 0.5
            }
        }
    )
    
    grid_size: int = Field(default=3, ge=2, le=5)
    sigma: float = Field(default=1.0, ge=0.5, le=2.0)
    learning_rate: float = Field(default=0.5, ge=0.1, le=1.0)

class StaffMember(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "staff_id": "1",
                "name": "John Doe",
                "latitude": 5.5826,
                "longitude": -0.1435,
                "address": "Accra"
            }
        }
    )
    
    staff_id: str
    name: str
    latitude: float = Field(ge=4.5, le=11.5)  # Ghana's latitude range
    longitude: float = Field(ge=-3.5, le=1.5)  # Ghana's longitude range
    address: str

class OptimizationRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "staff_data": [
                    {
                        "staff_id": "1",
                        "name": "John Doe",
                        "latitude": 5.5826,
                        "longitude": -0.1435,
                        "address": "Accra"
                    }
                ],
                "params": {
                    "grid_size": 3,
                    "sigma": 1.0,
                    "learning_rate": 0.5
                }
            }
        }
    )
    
    staff_data: List[StaffMember]
    params: Optional[OptimizationParams] = Field(default_factory=OptimizationParams)

# Add authentication router
app.include_router(auth.router, prefix="/auth", tags=["authentication"])

@app.get("/")
async def root():
    logger.info("Health check endpoint called")
    return {"message": "Route Optimizer API is running", "status": "healthy"}

@app.post("/optimize")
@limiter.limit("5/minute")  # Limit to 5 requests per minute
async def optimize_routes(
    request: OptimizationRequest,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"[{request_id}] Starting route optimization for {len(request.staff_data)} staff members")
    
    try:
        # Validate input data
        if not request.staff_data:
            raise HTTPException(status_code=400, detail="No staff data provided")
        
        # Convert staff data to DataFrame
        try:
            df = pd.DataFrame([member.model_dump() for member in request.staff_data])
            logger.info(f"[{request_id}] Converted staff data to DataFrame")
        except Exception as e:
            logger.error(f"[{request_id}] Failed to convert staff data to DataFrame: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid staff data format")
        
        # Validate staff data
        validated_data = optimizer.validate_staff_data(df)
        if validated_data is None:
            logger.error(f"[{request_id}] Staff data validation failed")
            raise HTTPException(status_code=400, detail="Invalid staff data")
        logger.info(f"[{request_id}] Staff data validated successfully")
        
        # Create clusters
        logger.info(f"[{request_id}] Creating clusters with params: {request.params.dict()}")
        try:
            clustered_data = optimizer.create_clusters(
                validated_data,
                grid_size=request.params.grid_size,
                sigma=request.params.sigma,
                learning_rate=request.params.learning_rate
            )
        except Exception as e:
            logger.error(f"[{request_id}] Clustering failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")
        
        if clustered_data is None:
            logger.error(f"[{request_id}] Clustering returned None")
            raise HTTPException(status_code=500, detail="Clustering failed")
        logger.info(f"[{request_id}] Clusters created successfully")
        
        # Optimize routes
        logger.info(f"[{request_id}] Starting route optimization")
        try:
            routes = optimizer.optimize_routes(clustered_data)
        except Exception as e:
            logger.error(f"[{request_id}] Route optimization failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Route optimization failed: {str(e)}")
        
        if not routes:
            logger.error(f"[{request_id}] No routes generated")
            raise HTTPException(status_code=500, detail="No routes could be generated")
        logger.info(f"[{request_id}] Routes optimized successfully: {len(routes)} routes created")
        
        # Calculate metrics
        try:
            metrics = optimizer.calculate_total_metrics(routes)
            logger.info(f"[{request_id}] Metrics calculated successfully")
        except Exception as e:
            logger.error(f"[{request_id}] Failed to calculate metrics: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to calculate metrics: {str(e)}")
        
        # Store results in database
        try:
            result = models.OptimizationResult(
                request_id=request_id,
                user_id=current_user.id,
                total_routes=metrics["total_routes"],
                total_passengers=metrics["total_passengers"],
                total_distance=metrics["total_distance"],
                total_cost=metrics["total_cost"],
                average_route_distance=metrics["average_route_distance"],
                average_route_cost=metrics["average_route_cost"],
                average_passengers_per_route=metrics["average_passengers_per_route"],
                cost_per_passenger=metrics["cost_per_passenger"],
                routes=routes,
                grid_size=request.params.grid_size,
                sigma=request.params.sigma,
                learning_rate=request.params.learning_rate
            )
            db.add(result)
            db.commit()
            db.refresh(result)
            logger.info(f"[{request_id}] Results stored in database")
        except Exception as e:
            db.rollback()
            logger.error(f"[{request_id}] Failed to store results in database: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to store results: {str(e)}")
        
        return {
            "request_id": request_id,
            "metrics": metrics,
            "routes": routes
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/csv")
@limiter.limit("5/minute")  # Limit to 5 requests per minute
async def optimize_routes_from_csv(
    request: Request,
    file: UploadFile = File(...),
    grid_size: int = 3,
    sigma: float = 1.0,
    learning_rate: float = 0.5,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"[{request_id}] Starting CSV route optimization for file: {file.filename}")
    
    try:
        # Read CSV file
        df = pd.read_csv(file.file)
        logger.info(f"[{request_id}] CSV file read successfully: {len(df)} rows")
        
        # Validate staff data
        validated_data = optimizer.validate_staff_data(df)
        if validated_data is None:
            logger.error(f"[{request_id}] Staff data validation failed")
            raise HTTPException(status_code=400, detail="Invalid staff data in CSV")
        logger.info(f"[{request_id}] Staff data validated successfully")
        
        # Create clusters
        logger.info(f"[{request_id}] Creating clusters with params: grid_size={grid_size}, sigma={sigma}, learning_rate={learning_rate}")
        clustered_data = optimizer.create_clusters(
            validated_data,
            grid_size=grid_size,
            sigma=sigma,
            learning_rate=learning_rate
        )
        
        if clustered_data is None:
            logger.error(f"[{request_id}] Clustering failed")
            raise HTTPException(status_code=500, detail="Clustering failed")
        logger.info(f"[{request_id}] Clusters created successfully")
        
        # Optimize routes
        logger.info(f"[{request_id}] Starting route optimization")
        routes = optimizer.optimize_routes(clustered_data)
        if not routes:
            logger.error(f"[{request_id}] Route optimization failed")
            raise HTTPException(status_code=500, detail="Route optimization failed")
        logger.info(f"[{request_id}] Routes optimized successfully: {len(routes)} routes created")
        
        # Calculate metrics
        metrics = optimizer.calculate_total_metrics(routes)
        logger.info(f"[{request_id}] Metrics calculated successfully")
        
        # Prepare route details
        route_details = {}
        for route_name, route in routes.items():
            try:
                distance, cost = optimizer.calculate_route_metrics(route)
                route_details[route_name] = {
                    "distance": round(distance, 2),
                    "cost": round(cost, 2),
                    "passengers": [
                        {
                            "name": passenger['name'],
                            "address": passenger['address'],
                            "staff_id": passenger['staff_id'],
                            "latitude": round(passenger['latitude'], 6),
                            "longitude": round(passenger['longitude'], 6)
                        }
                        for passenger in route
                    ]
                }
            except Exception as e:
                logger.error(f"[{request_id}] Failed to process route {route_name}: {str(e)}")
                continue
        
        # Prepare response
        response = {
            "request_id": request_id,
            "summary": {
                "total_routes": metrics['number_of_routes'],
                "total_passengers": metrics['total_passengers'],
                "total_distance": round(metrics['total_distance'], 2),
                "total_cost": round(metrics['total_cost'], 2),
                "average_route_distance": round(metrics['average_route_distance'], 2),
                "average_route_cost": round(metrics['average_route_cost'], 2),
                "average_passengers_per_route": round(metrics['average_passengers_per_route'], 1),
                "cost_per_passenger": round(metrics['cost_per_passenger'], 2)
            },
            "routes": route_details
        }
        
        # Save optimization result to database
        db_result = models.OptimizationResult(
            request_id=request_id,
            user_id=current_user.id,
            total_routes=metrics['number_of_routes'],
            total_passengers=metrics['total_passengers'],
            total_distance=round(metrics['total_distance'], 2),
            total_cost=round(metrics['total_cost'], 2),
            average_route_distance=round(metrics['average_route_distance'], 2),
            average_route_cost=round(metrics['average_route_cost'], 2),
            average_passengers_per_route=round(metrics['average_passengers_per_route'], 1),
            cost_per_passenger=round(metrics['cost_per_passenger'], 2),
            routes=route_details,
            grid_size=grid_size,
            sigma=sigma,
            learning_rate=learning_rate
        )
        db.add(db_result)
        db.commit()
        db.refresh(db_result)
        
        logger.info(f"[{request_id}] Optimization completed successfully and saved to database")
        return response
        
    except HTTPException as he:
        logger.error(f"[{request_id}] HTTP Exception: {str(he)}")
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        file.file.close()
        logger.info(f"[{request_id}] File handle closed")

@app.get("/sample-data")
async def get_sample_data():
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"[{request_id}] Fetching sample data")
    
    try:
        # Load sample data
        staff_data = optimizer.load_sample_data()
        if staff_data is None:
            logger.error(f"[{request_id}] Failed to load sample data")
            raise HTTPException(status_code=500, detail="Failed to load sample data")
        
        # Convert to list of dictionaries
        sample_data = staff_data.to_dict('records')
        logger.info(f"[{request_id}] Sample data loaded successfully: {len(sample_data)} records")
        
        return {"staff_data": sample_data}
        
    except HTTPException as he:
        logger.error(f"[{request_id}] HTTP Exception: {str(he)}")
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Add endpoint to get optimization history
@app.get("/optimize/history", response_model=List[schemas.OptimizationResult])
async def get_optimization_history(
    skip: int = 0,
    limit: int = 10,
    min_cost: Optional[float] = None,
    max_cost: Optional[float] = None,
    min_distance: Optional[float] = None,
    max_distance: Optional[float] = None,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get optimization history for the current user"""
    # Build query
    query = db.query(models.OptimizationResult)\
        .filter(models.OptimizationResult.user_id == current_user.id)
    
    # Apply filters
    filters = {}
    if min_cost is not None:
        query = query.filter(models.OptimizationResult.total_cost >= min_cost)
        filters['min_cost'] = min_cost
    if max_cost is not None:
        query = query.filter(models.OptimizationResult.total_cost <= max_cost)
        filters['max_cost'] = max_cost
    if min_distance is not None:
        query = query.filter(models.OptimizationResult.total_distance >= min_distance)
        filters['min_distance'] = min_distance
    if max_distance is not None:
        query = query.filter(models.OptimizationResult.total_distance <= max_distance)
        filters['max_distance'] = max_distance
    
    # Get results
    results = query.order_by(models.OptimizationResult.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    # Record history view
    for result in results:
        history_view = models.OptimizationHistoryView(
            optimization_result_id=result.id,
            user_id=current_user.id,
            view_type='list',
            filters_applied=filters if filters else None
        )
        db.add(history_view)
    
    db.commit()
    return results

# Add endpoint to get specific optimization result
@app.get("/optimize/{request_id}", response_model=schemas.OptimizationResult)
async def get_optimization_result(
    request_id: str,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific optimization result by request ID"""
    result = db.query(models.OptimizationResult)\
        .filter(
            models.OptimizationResult.request_id == request_id,
            models.OptimizationResult.user_id == current_user.id
        )\
        .first()
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Optimization result with request ID {request_id} not found"
        )
    
    # Record history view
    history_view = models.OptimizationHistoryView(
        optimization_result_id=result.id,
        user_id=current_user.id,
        view_type='detail'
    )
    db.add(history_view)
    db.commit()
    
    return result

# Add endpoint to get view history
@app.get("/optimize/history/views", response_model=List[schemas.OptimizationHistoryView])
async def get_optimization_view_history(
    skip: int = 0,
    limit: int = 10,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get history of optimization result views"""
    views = db.query(models.OptimizationHistoryView)\
        .filter(models.OptimizationHistoryView.user_id == current_user.id)\
        .order_by(models.OptimizationHistoryView.viewed_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    return views

# Add endpoint to update optimization parameters and rerun
@app.put("/optimize/{request_id}", response_model=schemas.OptimizationResult)
async def update_optimization(
    request_id: str,
    params: schemas.OptimizationParams,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update optimization parameters and rerun optimization"""
    # Get existing optimization result
    result = db.query(models.OptimizationResult)\
        .filter(
            models.OptimizationResult.request_id == request_id,
            models.OptimizationResult.user_id == current_user.id
        )\
        .first()
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Optimization result with request ID {request_id} not found"
        )
    
    # Extract staff data from existing routes
    staff_data = []
    for route in result.routes.values():
        for passenger in route['passengers']:
            staff_data.append({
                "staff_id": passenger['staff_id'],
                "name": passenger['name'],
                "latitude": passenger['latitude'],
                "longitude": passenger['longitude'],
                "address": passenger['address']
            })
    
    # Create new optimization request
    request = OptimizationRequest(
        staff_data=staff_data,
        params=params
    )
    
    # Run optimization with new parameters
    new_result = await optimize_routes(request, current_user, db)
    
    # Update existing result
    result.total_routes = new_result['summary']['total_routes']
    result.total_passengers = new_result['summary']['total_passengers']
    result.total_distance = new_result['summary']['total_distance']
    result.total_cost = new_result['summary']['total_cost']
    result.average_route_distance = new_result['summary']['average_route_distance']
    result.average_route_cost = new_result['summary']['average_route_cost']
    result.average_passengers_per_route = new_result['summary']['average_passengers_per_route']
    result.cost_per_passenger = new_result['summary']['cost_per_passenger']
    result.routes = new_result['routes']
    result.grid_size = params.grid_size
    result.sigma = params.sigma
    result.learning_rate = params.learning_rate
    
    db.commit()
    db.refresh(result)
    
    return result

# Add endpoint to delete optimization result
@app.delete("/optimize/{request_id}", response_model=schemas.OptimizationResult)
async def delete_optimization(
    request_id: str,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete an optimization result"""
    result = db.query(models.OptimizationResult)\
        .filter(
            models.OptimizationResult.request_id == request_id,
            models.OptimizationResult.user_id == current_user.id
        )\
        .first()
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Optimization result with request ID {request_id} not found"
        )
    
    # Delete associated history views
    db.query(models.OptimizationHistoryView)\
        .filter(models.OptimizationHistoryView.optimization_result_id == result.id)\
        .delete()
    
    # Delete the optimization result
    db.delete(result)
    db.commit()
    
    return result

# Add endpoint to update specific route
@app.patch("/optimize/{request_id}/routes/{route_id}")
async def update_route(
    request_id: str,
    route_id: str,
    route_update: dict,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update specific route details"""
    result = db.query(models.OptimizationResult)\
        .filter(
            models.OptimizationResult.request_id == request_id,
            models.OptimizationResult.user_id == current_user.id
        )\
        .first()
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Optimization result with request ID {request_id} not found"
        )
    
    # Format route_id to match the data format (e.g., "Route 1" instead of "Route_1")
    formatted_route_id = f"Route {route_id.split('_')[-1]}" if '_' in route_id else route_id
    
    if formatted_route_id not in result.routes:
        available_routes = list(result.routes.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Route {route_id} not found in optimization result. Available routes: {available_routes}"
        )
    
    # Update route details
    current_route = result.routes[formatted_route_id]
    current_route.update(route_update)
    result.routes[formatted_route_id] = current_route
    
    # Recalculate metrics for this route
    try:
        distance, cost = optimizer.calculate_route_metrics(current_route['passengers'])
        current_route['distance'] = round(distance, 2)
        current_route['cost'] = round(cost, 2)
    except Exception as e:
        logger.error(f"Failed to recalculate route metrics: {str(e)}")
    
    db.commit()
    db.refresh(result)
    
    return {"message": "Route updated successfully", "route": current_route}

# Add endpoint to delete specific route
@app.delete("/optimize/{request_id}/routes/{route_id}")
async def delete_route(
    request_id: str,
    route_id: str,
    current_user: models.User = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a specific route from optimization result"""
    try:
        # Get the optimization result
        result = db.query(models.OptimizationResult)\
            .filter(
                models.OptimizationResult.request_id == request_id,
                models.OptimizationResult.user_id == current_user.id
            )\
            .first()
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Optimization result with request ID {request_id} not found"
            )
        
        # Format route_id to match the data format
        formatted_route_id = f"Route {route_id.split('_')[-1]}" if '_' in route_id else route_id
        
        # Get current routes
        current_routes = result.routes or {}
        
        if formatted_route_id not in current_routes:
            available_routes = list(current_routes.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Route {route_id} not found in optimization result. Available routes: {available_routes}"
            )
        
        # Store route data before deletion for response
        deleted_route = current_routes[formatted_route_id]
        
        # Create a new routes dictionary without the deleted route
        updated_routes = {k: v for k, v in current_routes.items() if k != formatted_route_id}
        
        # Renumber the remaining routes
        renumbered_routes = {}
        for i, (_, route_data) in enumerate(updated_routes.items(), 1):
            renumbered_routes[f"Route {i}"] = route_data
        
        # Update the routes in the result
        result.routes = renumbered_routes
        
        # Update metrics
        result.total_routes = len(renumbered_routes)
        result.total_passengers -= len(deleted_route['passengers'])
        result.total_distance -= deleted_route['distance']
        result.total_cost -= deleted_route['cost']
        
        # Recalculate averages
        if result.total_routes > 0:
            result.average_route_distance = result.total_distance / result.total_routes
            result.average_route_cost = result.total_cost / result.total_routes
            result.average_passengers_per_route = result.total_passengers / result.total_routes
            result.cost_per_passenger = result.total_cost / result.total_passengers
        else:
            # If no routes left, reset metrics
            result.average_route_distance = 0
            result.average_route_cost = 0
            result.average_passengers_per_route = 0
            result.cost_per_passenger = 0
        
        # Commit changes to database
        db.commit()
        db.refresh(result)
        
        return {
            "message": "Route deleted successfully",
            "deleted_route": deleted_route,
            "updated_routes": renumbered_routes,
            "updated_metrics": {
                "total_routes": result.total_routes,
                "total_passengers": result.total_passengers,
                "total_distance": result.total_distance,
                "total_cost": result.total_cost,
                "average_route_distance": result.average_route_distance,
                "average_route_cost": result.average_route_cost,
                "average_passengers_per_route": result.average_passengers_per_route,
                "cost_per_passenger": result.cost_per_passenger
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error deleting route: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete route: {str(e)}"
        )

@app.get("/test-openapi")
async def test_openapi():
    """Test endpoint to verify OpenAPI schema generation"""
    schema = app.openapi()
    return {"message": "OpenAPI schema generated successfully", "schema": schema}

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {"message": "API is working"}

@app.post("/test-model")
async def test_model_endpoint(data: OptimizationParams):
    """Test endpoint that uses a Pydantic model"""
    return {"message": "Model validation successful", "data": data}

