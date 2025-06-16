import pytest
import uvicorn

# Skip this test due to pydantic v1/FastAPI compatibility issues
pytest.skip("Skipping FastAPI tests due to pydantic v1 compatibility issues", allow_module_level=True)

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
