from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Authormaton Core AI Engine", version="1.0")
# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Authormaton API!"}

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
@app.get("/health")
def health():
    return {"status": "ok"}
