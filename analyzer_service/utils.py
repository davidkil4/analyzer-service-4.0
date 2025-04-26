import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def count_words(text: str) -> int:
    """Counts words in a string, splitting by whitespace."""
    if not text:
        return 0
    return len(text.split())

def load_prompt(prompt_path: str | Path) -> str:
    """Loads a prompt from a file relative to this utils.py file."""
    # Get the directory containing this utils.py file
    utils_dir = Path(__file__).parent
    # Construct the full path relative to the utils_dir
    full_prompt_path = utils_dir / prompt_path
    try:
        with open(full_prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {full_prompt_path}")
        # Return a default/error message or raise an exception?
        return "Error: Prompt file not found."
    except Exception as e:
        logger.error(f"Error reading prompt file {full_prompt_path}: {e}")
        return "Error: Could not read prompt file."

# Add any other utility functions needed, e.g.:
# - Functions to generate unique IDs (for AS units, clauses)
# - More sophisticated text cleaning/normalization functions
