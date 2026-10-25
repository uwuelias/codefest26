from fastapi import APIRouter, HTTPException, UploadFile, File
from services.extract import get_embedding


router = APIRouter(
    prefix="/api/extract"
)

@router.post("/")
async def get_embedding(file: UploadFile = File()):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="Invalid file input.")
        embedding = await get_embedding(file)
        return embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {e!s}.")
