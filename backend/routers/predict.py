import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile

from services.predict import analyze

router = APIRouter(prefix="/api/predict")

@router.post("/")
def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    try:
        return analyze(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process audio: {e}")
    finally:
        Path(path).unlink(missing_ok=True)