from fastapi import APIRouter
from datetime import datetime
from src.api.scheme.health import HealthApp

router = APIRouter(prefix="Health", tags=['Health'])

@router.get('/health')
def health_api(HealthApp) -> HealthApp:
    return HealthApp(
        status="Ok",
        datetime=datetime.now()
        )
    
@router.get('/health')
def health_model(HealthApp):
    return HealthApp(
        status="Ok",
        datetime=datetime.now()
        )
    
