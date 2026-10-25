from fastapi import APIRouter, File, HTTPException, UploadFile
from services.predict import analyze
import tempfile
import os
import shutil

router = APIRouter(
     prefix="/api/predict"
)

@router.post("/")
def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    try:
        return analyze(path)
    finally:
        Path(path).unlink(missing_ok=True)
