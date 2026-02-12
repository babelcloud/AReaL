"""
Utilities for converting OpenAI Chat Completions API format to Tinker ModelInput format.

GBoxAgent uses standard OpenAI Chat Completions API format with text and image_url,
which needs to be converted to Tinker's Message format (with ImagePart) and then
to ModelInput (with ImageChunk) for training.
"""

import base64
import io
import logging
from typing import Any, Dict, List

import tinker
from PIL import Image

from tinker_cookbook import renderers

logger = logging.getLogger(__name__)


def convert_openai_responses_to_message(
    openai_message: Dict[str, Any]
) -> renderers.Message:
    """
    Convert OpenAI Chat Completions API format message to Tinker Message format.
    
    Supports both standard OpenAI format and legacy Responses API format:
    
    Standard OpenAI Chat Completions API format:
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    }
    
    Legacy Responses API format (for backward compatibility):
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "..."},
            {"type": "input_image", "image_url": "data:image/png;base64,..."}
        ]
    }
    
    Tinker Message format:
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "..."},
            {"type": "image", "image": "data:image/png;base64,..."}
        ]
    }
    
    Args:
        openai_message: Message in OpenAI Chat Completions API format
        
    Returns:
        Message in Tinker format
    """
    role = openai_message.get("role", "user")
    content = openai_message.get("content", [])
    
    # If content is a string, convert to list
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    
    # Convert content parts
    converted_parts: List[renderers.TextPart | renderers.ImagePart] = []
    for part in content:
        part_type = part.get("type")
        
        # Standard OpenAI format: "text"
        if part_type == "text":
            converted_parts.append(renderers.TextPart(type="text", text=part.get("text", "")))
        # Standard OpenAI format: "image_url"
        elif part_type == "image_url":
            image_url_obj = part.get("image_url", {})
            # Handle both dict format {"url": "..."} and string format
            if isinstance(image_url_obj, dict):
                image_url = image_url_obj.get("url", "")
            else:
                image_url = image_url_obj
            if isinstance(image_url, str):
                converted_parts.append(renderers.ImagePart(type="image", image=image_url))
            else:
                logger.warning(f"Unexpected image_url type: {type(image_url)}")
        # Gbox format: "image" with url
        elif part_type == "image":
            image_url = part.get("url", "")
            if isinstance(image_url, str):
                converted_parts.append(renderers.ImagePart(type="image", image=image_url))
            else:
                logger.warning(f"Unexpected image url type: {type(image_url)}")
        # Legacy Responses API format: "input_text" (for backward compatibility)
        elif part_type == "input_text":
            converted_parts.append(renderers.TextPart(type="text", text=part.get("text", "")))
        # Legacy Responses API format: "input_image" (for backward compatibility)
        elif part_type == "input_image":
            image_url = part.get("image_url", "")
            if isinstance(image_url, str):
                converted_parts.append(renderers.ImagePart(type="image", image=image_url))
            else:
                logger.warning(f"Unexpected image_url type: {type(image_url)}")
        else:
            logger.warning(f"Unknown content part type: {part_type}")
    
    # Return message with appropriate content format
    if len(converted_parts) == 0:
        content: renderers.Content = ""
    elif len(converted_parts) == 1 and converted_parts[0]["type"] == "text":
        content = converted_parts[0]["text"]
    else:
        content = converted_parts
    
    return renderers.Message(role=role, content=content)


def load_image_from_data_uri(data_uri: str) -> Image.Image:
    """
    Load PIL Image from data URI.
    
    Args:
        data_uri: Data URI string (e.g., "data:image/png;base64,...")
        
    Returns:
        PIL Image
    """
    if data_uri.startswith("<<gbox:image:"):
        raise ValueError(f"Gbox image placeholder cannot be decoded: {data_uri}")
    if not data_uri.startswith("data:"):
        raise ValueError(f"Expected data URI, got: {data_uri[:50]}...")
    
    # Parse data URI
    header, encoded = data_uri.split(",", 1)
    image_data = base64.b64decode(encoded)
    
    return Image.open(io.BytesIO(image_data))


def convert_messages_to_model_input(
    messages: List[renderers.Message],
    renderer: renderers.Renderer,
) -> tinker.ModelInput:
    """
    Convert a list of messages to ModelInput using the renderer.
    
    This is a convenience function that uses the renderer's build_generation_prompt
    to convert messages to ModelInput, which handles vision inputs correctly.
    
    Args:
        messages: List of messages in Tinker format
        renderer: Renderer instance (should support vision, e.g., Qwen3VLRenderer)
        
    Returns:
        ModelInput ready for training
    """
    return renderer.build_generation_prompt(messages)

