import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.predict import analyze

router = APIRouter(
     prefix="/api/predict"
)


@router.get("/")
async def predict(file: UploadFile = File(...)):
        try:
            return await analyze(file)
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=503,
                detail="Prediction Model unavailable.",
            )
