"""Prompt templates for CUA Agent."""

from datetime import datetime
from pathlib import Path
from typing import Optional


def create_system_prompt(
    task_description: str,
    max_turns: int = 20,
    app_name: Optional[str] = None,  # Deprecated, kept for backward compatibility
    cua_guide: Optional[str] = None,  # New: guide text from config
    box_type: str = "android",  # "android", "linux", "windows", etc.
    coordinate_mode: str = "gbox",  # "gbox" or "direct"
    coordinate_scale: bool = False,  # Whether to apply coordinate scaling (Direct mode only)
    screen_width: Optional[int] = None,  # Screen width in pixels (for Direct mode without scaling)
    screen_height: Optional[int] = None,  # Screen height in pixels (for Direct mode without scaling)
) -> str:
    """Create the system prompt for the CUA agent.
    
    Args:
        task_description: Description of the task to complete
        max_turns: Maximum number of turns allowed
        app_name: [Deprecated] Name of the app to operate (use cua_guide instead)
        cua_guide: Guide text for what to operate (e.g., "You need to operate the Airbnb app to complete the task.")
        box_type: Type of environment (android, linux, windows, etc.)
        coordinate_mode: Coordinate generation mode
            - "gbox": Use GBox external model (agent describes elements, GBox generates coords)
            - "direct": VLM directly outputs coordinates in tool calls
        coordinate_scale: Whether to apply coordinate scaling (Direct mode only)
            - False: Include screen dimensions in prompt, model outputs actual pixels
            - True: Don't include screen dimensions, model outputs normalized coords (e.g., 0-1000)
        screen_width: Screen width in pixels (for Direct mode without scaling, extracted from screenshot)
        screen_height: Screen height in pixels (for Direct mode without scaling, extracted from screenshot)
        
    Returns:
        Formatted system prompt
    """
    # Get the directory containing this file (utils), then go up to cua_rl
    current_dir = Path(__file__).parent.parent  # cua_rl directory
    
    # Default prompt selection based on box_type + coordinate_mode.
    if box_type.lower() in ["android"]:
        prompt_prefix = "android"
    elif box_type.lower() in ["linux", "windows", "pc", "desktop"]:
        prompt_prefix = "pc"
    else:
        # Default to android for unknown types
        prompt_prefix = "android"

    if coordinate_mode == "gbox":
        prompt_file = current_dir / "prompts" / f"{prompt_prefix}-system-prompt-gbox.txt"
    elif coordinate_mode == "direct":
        prompt_file = current_dir / "prompts" / f"{prompt_prefix}-system-prompt-direct.txt"
    else:
        raise ValueError(
            f"Unknown coordinate_mode: {coordinate_mode}. "
            f"Must be 'gbox' or 'direct'"
        )

    # Read the template from file
    with open(prompt_file, "r", encoding="utf-8") as f:
        template = f.read()
    
    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Determine CUA guide text
    if cua_guide:
        # Use provided guide text from config
        guide_text = cua_guide
    elif app_name:
        # Fallback to old app_name format for backward compatibility
        app_info = app_name.capitalize() if app_name else "Unknown"
        guide_text = f"You need to operate the {app_info} app to complete the task."
    else:
        # Default fallback
        guide_text = "You need to operate the system to complete the task."
    
    # Replace placeholders
    prompt = template.replace("TASK_DESCRIPTION", task_description)
    prompt = prompt.replace("MAX_TURNS", str(max_turns))
    prompt = prompt.replace("CURRENT_TIME", current_time)
    prompt = prompt.replace("CUA_GUIDE", guide_text)
    
    # For Direct mode, replace screen dimension placeholders
    if coordinate_mode == "direct":
        if coordinate_scale:
            # With scaling: Don't include screen dimensions in prompt
            # Replace placeholders with generic text
            prompt = prompt.replace("SCREEN_WIDTH", "[normalized]")
            prompt = prompt.replace("SCREEN_HEIGHT", "[normalized]")
            prompt = prompt.replace("SCREEN_CENTER_X", "[500]")  # Center of 1000x1000
            prompt = prompt.replace("SCREEN_CENTER_Y", "[500]")
        else:
            # Without scaling: Include actual screen dimensions
            if screen_width is None or screen_height is None:
                # Initial call before first screenshot - leave placeholders
                prompt = prompt.replace("SCREEN_WIDTH", "[TO_BE_DETERMINED]")
                prompt = prompt.replace("SCREEN_HEIGHT", "[TO_BE_DETERMINED]")
                prompt = prompt.replace("SCREEN_CENTER_X", "[TO_BE_DETERMINED]")
                prompt = prompt.replace("SCREEN_CENTER_Y", "[TO_BE_DETERMINED]")
            else:
                # Screen dimensions provided - replace with actual values
                center_x = screen_width // 2
                center_y = screen_height // 2
                
                prompt = prompt.replace("SCREEN_WIDTH", str(screen_width))
                prompt = prompt.replace("SCREEN_HEIGHT", str(screen_height))
                prompt = prompt.replace("SCREEN_CENTER_X", str(center_x))
                prompt = prompt.replace("SCREEN_CENTER_Y", str(center_y))
    
    return prompt

