from fastapi import FastAPI
from src.routers import make_datasets
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


# Enable CORS to allow your React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include ML-related routers
app.include_router(make_datasets.router)

@app.get("/")
def root():
    return {"message": "Welcome to the ML visualization app!"}