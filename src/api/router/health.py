
from typing import Any

from fastapi import APIRouter

PREFIX = '/health'

router = APIRouter(prefix=PREFIX, tags=['health'])

@router.get(PREFIX)
async def health() -> dict[str, Any]:
    return {"status": "alive"}

