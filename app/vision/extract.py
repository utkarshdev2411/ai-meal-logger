import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.config import get_settings
from app.vision.schema import VisionObservation

async def extract_vision(image_bytes: bytes, mime_type: str) -> VisionObservation:
    """Calls the vision model to extract structured food items from an image."""
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.vision_model,
        api_key=settings.require_api_key(),
        base_url=settings.llm_base_url,
        max_tokens=1024,
        temperature=0.1,
    ).with_structured_output(VisionObservation)
    
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Identify all the food items in this image. Estimate portions and overall confidence. Keep alternatives concise."
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"}
            }
        ]
    )
    
    result = await llm.ainvoke([message])
    return result
