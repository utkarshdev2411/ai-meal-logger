from typing import Any
import asyncio
from datetime import datetime, timezone
from app.db.models import Image
from sqlalchemy import select
from app.vision.extract import extract_vision

async def process_image_by_id(session: Any, image_id: str) -> dict:
    """Gets image from DB, extracts if needed, and returns the observation dictionary."""
    stmt = select(Image).where(Image.id == image_id)
    image = (await session.execute(stmt)).scalar_one_or_none()
    
    if not image:
        return {"error": "Image not found"}
        
    if image.status == "ready" and image.observation:
        return image.observation
        
    if image.status == "failed":
        return {"error": image.error}
        
    # If it's pending but we're here, it means the background task hasn't finished,
    # or it's being done synchronously for testing. We'll extract it now.
    try:
        with open(image.path, "rb") as f:
            image_bytes = f.read()
        
        mime_type = image.mime or "image/jpeg"
        obs = await extract_vision(image_bytes, mime_type)
        obs_dict = obs.model_dump()
        
        image.status = "ready"
        image.observation = obs_dict
        image.completed_at = datetime.now(timezone.utc)
        await session.commit()
        return obs_dict
        
    except Exception as e:
        image.status = "failed"
        image.error = str(e)
        image.completed_at = datetime.now(timezone.utc)
        await session.commit()
        return {"error": str(e)}
