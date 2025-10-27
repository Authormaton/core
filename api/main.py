import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.indexing_router import router as indexing_router
from services.logging_config import setup_logging
from services.web_fetch_service import WebFetchService
from api.endpoints.internal import router as internal_router
from api.endpoints.upload import router as upload_router
from api.endpoints.web_answering import router as web_answering_router


logger = logging.getLogger(__name__)

# Parse CORS_ALLOW_ORIGINS from env (comma-separated)
def get_cors_origins():
    origins = os.environ.get("CORS_ALLOW_ORIGINS", "")
    if origins.strip() == "*":
        return ["*"]
    # Split by comma, strip whitespace, filter empty
    return [o.strip() for o in origins.split(",") if o.strip()]


ALLOWED_ORIGINS = get_cors_origins()
ALLOW_CREDENTIALS = False if ALLOWED_ORIGINS == ["*"] else True

# Global service instances
web_fetch_service: Optional[WebFetchService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global web_fetch_service
    web_fetch_service = WebFetchService()
    yield
    # Shutdown logic
    if web_fetch_service:
        await web_fetch_service.close()


setup_logging()
app = FastAPI(title="Authormaton Core AI Engine", version="1.0", lifespan=lifespan)


# Middleware to capture X-Request-Id and inject into logs
@app.middleware("http")
async def add_request_id_to_log(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id")
    response = await call_next(request)
    if request_id:
        response.headers["X-Request-Id"] = request_id
    return response


# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Authormaton API!"}


# CORS config: allow credentials only if not wildcard
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(upload_router, prefix="/upload")
app.include_router(internal_router)
app.include_router(web_answering_router, prefix="/internal", tags=["websearch"])
app.include_router(indexing_router)


@app.get("/health")
def health():
    logger.info("Health check requested")
    return {"status": "ok"}
