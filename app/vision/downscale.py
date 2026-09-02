import io
from PIL import Image

def downscale_image(image_bytes: bytes, max_edge: int) -> tuple[bytes, int, int, str]:
    """Downscales image so its longest edge is at most `max_edge`, preserving aspect ratio.
    Returns (downscaled_bytes, width, height, mime_type).
    """
    img = Image.open(io.BytesIO(image_bytes))
    
    width, height = img.size
    if max(width, height) > max_edge:
        ratio = max_edge / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    format_to_save = img.format or "JPEG"
    # Ensure RGB if it has alpha channel and we're saving as JPEG
    if img.mode in ("RGBA", "P") and format_to_save.upper() == "JPEG":
        img = img.convert("RGB")
        
    out = io.BytesIO()
    img.save(out, format=format_to_save)
    return out.getvalue(), img.width, img.height, f"image/{format_to_save.lower()}"
