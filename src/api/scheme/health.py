from pydantic import BaseModel


class HealthApp(BaseModel):
    status: str 
    timestamp: str