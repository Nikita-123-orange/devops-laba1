from fastapi import FastAPI
import uvicorn
from src.api.router.health import router as health_router

app = FastAPI(title="My FastAPI Application", description="A simple FastAPI application", version="1.0.0")

app.include_router(health_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)