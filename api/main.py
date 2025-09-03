from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Parse CORS_ALLOW_ORIGINS from env (comma-separated)
def get_cors_origins():
    origins = os.environ.get("CORS_ALLOW_ORIGINS", "")
    if origins.strip() == "*":
        return ["*"]
    # Split by comma, strip whitespace, filter empty
    return [o.strip() for o in origins.split(",") if o.strip()]

ALLOWED_ORIGINS = get_cors_origins()
ALLOW_CREDENTIALS = False if ALLOWED_ORIGINS == ["*"] else True

app = FastAPI(title="Authormaton Core AI Engine", version="1.0")
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

@app.get("/health")
def health():
    return {"status": "ok"}
