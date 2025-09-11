from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from services.logging_config import setup_logging
from api.indexing_router import router as indexing_router

# Parse CORS_ALLOW_ORIGINS from env (comma-separated)
def get_cors_origins():
    origins = os.environ.get("CORS_ALLOW_ORIGINS", "")
    if origins.strip() == "*":
        return ["*"]
    # Split by comma, strip whitespace, filter empty
    return [o.strip() for o in origins.split(",") if o.strip()]

ALLOWED_ORIGINS = get_cors_origins()
ALLOW_CREDENTIALS = False if ALLOWED_ORIGINS == ["*"] else True



setup_logging()
app = FastAPI(title="Authormaton Core AI Engine", version="1.0")

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
from api.endpoints.upload import router as upload_router
from api.endpoints.internal import router as internal_router
app.include_router(upload_router, prefix="/upload")
app.include_router(internal_router)
app.include_router(indexing_router)

@app.get("/health")
def health():
    return {"status": "ok"}
