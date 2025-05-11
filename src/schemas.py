from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_superuser: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None

class UserInDBBase(UserBase):
    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserInDB(UserInDBBase):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None
    client_id: Optional[str] = None

class ClientBase(BaseModel):
    name: str

class ClientCreate(ClientBase):
    pass

class Client(ClientBase):
    id: int
    client_id: str
    client_secret: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class ClientCredentials(BaseModel):
    client_id: str
    client_secret: str

class OptimizationParams(BaseModel):
    grid_size: int = Field(default=3, ge=2, le=5)
    sigma: float = Field(default=1.0, ge=0.5, le=2.0)
    learning_rate: float = Field(default=0.5, ge=0.1, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "grid_size": 3,
                "sigma": 1.0,
                "learning_rate": 0.5
            }
        }

class Passenger(BaseModel):
    name: str
    address: str
    staff_id: str
    latitude: float
    longitude: float

class Route(BaseModel):
    distance: float
    cost: float
    passengers: List[Passenger]

class RouteUpdate(BaseModel):
    distance: Optional[float] = None
    cost: Optional[float] = None
    passengers: Optional[List[Passenger]] = None

class OptimizationResultBase(BaseModel):
    request_id: str
    total_routes: int
    total_passengers: int
    total_distance: float
    total_cost: float
    average_route_distance: float
    average_route_cost: float
    average_passengers_per_route: float
    cost_per_passenger: float
    routes: Dict[str, Route]
    grid_size: int
    sigma: float
    learning_rate: float

class OptimizationResultCreate(OptimizationResultBase):
    pass

class OptimizationResult(OptimizationResultBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class OptimizationHistoryViewBase(BaseModel):
    optimization_result_id: int
    view_type: str
    filters_applied: Optional[Dict[str, Any]] = None

class OptimizationHistoryViewCreate(OptimizationHistoryViewBase):
    pass

class OptimizationHistoryView(OptimizationHistoryViewBase):
    id: int
    user_id: int
    viewed_at: datetime

    class Config:
        from_attributes = True 