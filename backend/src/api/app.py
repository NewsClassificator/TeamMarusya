from fastapi import FastAPI

from src.api.routes import router as analyze_router
from src.api.routes_clickbait import router as clickbait_router
from src.api.routes_water import router as water_router


def create_app() -> FastAPI:
    """
    Application factory for the News Analysis API.
    """
    app = FastAPI(
        title="News Analysis API",
        version="0.1.0",
    )

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        return {"status": "ok"}

    app.include_router(analyze_router)
    app.include_router(clickbait_router)
    app.include_router(water_router)

    return app


app = create_app()
