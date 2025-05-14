from pydantic import BaseModel, EmailStr, Field, constr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from .models import UserRole

class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: str = Field(..., min_length=2, max_length=100)
    mobile_number: constr(min_length=10, max_length=15)  # Basic phone number validation
    country: str = Field(..., min_length=2, max_length=100)
    company: str = Field(..., min_length=2, max_length=100)
    role: UserRole = UserRole.STAFF

    @validator('mobile_number')
    def validate_mobile_number(cls, v):
        # Remove any non-digit characters for validation
        digits = ''.join(filter(str.isdigit, v))
        if len(digits) < 10:
            raise ValueError('Mobile number must have at least 10 digits')
        return v

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class User(UserBase):
    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    mobile_number: Optional[constr(min_length=10, max_length=15)] = None
    country: Optional[str] = Field(None, min_length=2, max_length=100)
    company: Optional[str] = Field(None, min_length=2, max_length=100)
    role: Optional[UserRole] = None
    password: Optional[str] = Field(None, min_length=8)

    @validator('mobile_number')
    def validate_mobile_number(cls, v):
        if v is None:
            return v
        # Remove any non-digit characters for validation
        digits = ''.join(filter(str.isdigit, v))
        if len(digits) < 10:
            raise ValueError('Mobile number must have at least 10 digits')
        return v

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