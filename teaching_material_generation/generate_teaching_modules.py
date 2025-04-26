import concurrent.futures
import json
import logging
import os
import re
import time
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import google.generativeai as genai
from google.generativeai import GenerativeModel
from dotenv import load_dotenv

# --- Add project root to sys.path ---
# Get the absolute path of the current script
current_script_path = Path(__file__).resolve()
# Assuming the script is in 'teaching_material_generation', go up two levels to the project root
project_root = current_script_path.parent.parent
sys.path.append(str(project_root))
logger = logging.getLogger(__name__)
logger.info(f"Added project root to sys.path: {project_root}")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Restore Gemini Client Setup --- 
load_dotenv() # Ensure env vars are loaded

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables.")
    sys.exit(1)

genai.configure(api_key=API_KEY)

# --- Constants --- 
MODEL_NAME = "gemini-2.0-flash" # USER SPECIFIED MODEL
MAX_WORKERS = 5 # Max concurrent API calls
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 2 # seconds

# Set generation config if needed (e.g., temperature, safety settings)
generation_config = genai.GenerationConfig(
    temperature=0.7, # Example: adjust as needed
    # response_mime_type="application/json" # Request JSON directly if model supports
)

# Configure safety settings (adjust levels as needed)
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

# Initialize the Generative Model
try:
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    logger.info(f"Successfully initialized Gemini model: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize Gemini model: {e}")
    sys.exit(1)

# --- Helper Functions ---

def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Loads a JSON file, returning a dictionary or None on error."""
    if not file_path.exists():
        logger.error(f"Input file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
             logger.error(f"JSON data in {file_path} is not a dictionary.")
             return None # Or handle as appropriate
        return data
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {file_path}")
        return None
    except Exception as e:
         logger.error(f"Unexpected error loading file {file_path}: {e}")
         return None

def save_output(data: List[Dict[str, Any]], output_path: Path):
    """Saves the generated modules to a JSON file."""
    if not data:
        logger.warning("No modules generated, skipping save.")
        return
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Attempting to save {len(data)} modules to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully saved output to: {output_path}")
    except IOError as e:
        logger.error(f"Error saving output file {output_path}: {e}")
    except Exception as e:
         logger.error(f"Unexpected error during saving to {output_path}: {e}")

def determine_priority(utt_data: Dict[str, Any]) -> str:
    """Determines the priority level from utterance metadata."""
    # Navigate safely through potentially missing keys
    filtering_metadata = utt_data.get('filtering_metadata', {})
    if not isinstance(filtering_metadata, dict):
        logger.warning(f"'filtering_metadata' is not a dictionary for utterance. Defaulting to LOW priority.")
        return 'LOW'
    priority = filtering_metadata.get('priority_level', 'LOW')
    # Validate the priority level
    if priority not in ['LOW', 'MEDIUM', 'HIGH']:
        logger.warning(f"Invalid priority level '{priority}' found. Defaulting to LOW.")
        priority = 'LOW'
    return priority

def normalize_for_id(text: str) -> str:
    """Simple normalization for creating IDs."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\W+', '_', text)
    return text.strip('_').lower()[:30] # Limit length for sanity

def format_utterance_for_prompt(utt_data: Dict[str, Any]) -> str:
    """Formats a single utterance's data for inclusion in a batch prompt."""
    original = utt_data.get('original', 'N/A')
    corrected = utt_data.get('corrected', 'N/A')
    context_str = "\\n".join([f"{turn.get('speaker', '?')}: {turn.get('text', '')}" for turn in utt_data.get('preceding_context', [])])
    if not context_str: context_str = "(No preceding conversation context available)"

    errors_str = "No specific errors listed."
    if utt_data.get('error_details'):
        errors_str = "\\n".join([
            f"- Clause: '{err.get('clause_text', 'N/A')}' -> Error ({err.get('severity', 'N/A')} {err.get('category', 'N/A')}): {err.get('error_description', 'N/A')} (Suggestion: {err.get('correction_suggestion', 'N/A')})"
            for err in utt_data['error_details']
        ])

    target_pattern = utt_data.get('target_pattern', 'N/A')
    approach_reason = utt_data.get('approach_reason', 'N/A')
    priority = utt_data.get('priority_level', 'N/A') # Used for Med/High
    utterance_index = utt_data.get('utterance_index', 'N/A') # Crucial for mapping results

    # Basic structure common to all types
    prompt_part = f"""
    ---
    Utterance Identifier: {utterance_index}
    Conversation Context Leading Up To This Utterance:
    {context_str}

    Learner's Utterance:
    Original: \"{original}\"
    Corrected: \"{corrected}\"
    """
    # Add type-specific details
    if priority != 'N/A' and priority != 'DRILL': # Conversational Low/Med/High
         prompt_part += f"""
    Identified Issue (Priority: {priority}):
    Topic/Error: {target_pattern}
    Reasoning: {approach_reason}
    Specific Errors Found:
    {errors_str}
        """
    elif priority == 'DRILL': # Drill
        prompt_part += f"""
    Identified Target for Drill Practice:
    Pattern/Topic: {target_pattern}
    Reason: {approach_reason}
        """

    formatted_string = prompt_part.strip()
    logger.debug(f"Formatted prompt data for utterance {utterance_index}:\n{formatted_string}") # Added DEBUG logging
    return formatted_string

def parse_batch_response(response_text: Optional[str], batch_type: str, expected_count: int) -> List[Dict[str, Any]]:
    """Parses the JSON response from a batch AI call."""
    if not response_text:
        logger.error(f"Batch API call for {batch_type} returned empty response.")
        return []

    try:
        # Clean the response
        cleaned_response = clean_json_response(response_text)
        if not cleaned_response:
            logger.error(f"Cleaned response is empty for {batch_type} batch.")
            logger.debug(f"Original response for {batch_type}: {response_text}")
            return []

        # Parse the JSON structure (expecting {"generated_modules": [...]})
        data = json.loads(cleaned_response)

        if not isinstance(data, dict) or "generated_modules" not in data:
            logger.error(f"Batch response for {batch_type} is not a dict with 'generated_modules' key.")
            logger.debug(f"Parsed data for {batch_type}: {data}")
            return []

        modules = data["generated_modules"]

        if not isinstance(modules, list):
            logger.error(f"'generated_modules' in {batch_type} batch response is not a list.")
            logger.debug(f"Parsed modules for {batch_type}: {modules}")
            return []

        # Optional: Check if count matches expected (can be tricky if AI skips some)
        if len(modules) != expected_count:
             logger.warning(f"Expected {expected_count} modules for {batch_type}, but received {len(modules)}. AI might have skipped some inputs.")

        # Basic validation of module structure (can be expanded)
        valid_modules = []
        for i, module in enumerate(modules):
            if isinstance(module, dict) and 'utterance_index' in module and 'explanations' in module and 'problems' in module:
                 valid_modules.append(module)
            else:
                logger.warning(f"Invalid module structure received in {batch_type} batch at index {i}. Skipping. Module: {module}")

        return valid_modules

    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error for {batch_type} batch: {e}")
        logger.error(f"Raw AI Response for {batch_type}: {response_text}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error parsing {batch_type} batch response: {e}")
        return []

# Simple wrapper to mimic the removed util function
def call_gemini_api(prompt: str, model: GenerativeModel) -> Optional[str]:
    """Calls the configured Gemini API and returns the text response."""
    try:
        logger.debug(f"Sending prompt to Gemini (length: {len(prompt)} chars): {prompt[:200]}...")
        # Request JSON output directly if model/API supports it
        # response = model.generate_content(prompt, generation_config=genai.GenerationConfig(response_mime_type="application/json"))
        response = model.generate_content(prompt) # Assuming text output for now

        # Basic check for blocked content
        if not response.candidates:
             logger.error("AI response blocked or empty. Prompt Feedback: %s", response.prompt_feedback)
             return None
             
        # Log safety ratings if available
        if hasattr(response.candidates[0], 'safety_ratings'):
             logger.debug(f"Safety Ratings: {response.candidates[0].safety_ratings}")

        response_text = response.text
        logger.debug(f"Raw AI Response Text (length: {len(response_text)} chars): {response_text[:200]}...")
        return response_text
    except Exception as e:
        logger.error(f"Error during Gemini API call: {e}")
        # Log response parts if available for debugging
        if 'response' in locals() and hasattr(response, 'prompt_feedback'):
            logger.error(f"Prompt Feedback: {response.prompt_feedback}")
        return None

# Simple wrapper to mimic the removed util function
def clean_json_response(response_text: Optional[str]) -> Optional[str]:
    """Cleans the AI response to extract the JSON part."""
    if not response_text:
        return None
    # Attempt to find JSON block within markdown
    match = re.search(r'```(json)?(.*)```', response_text, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(2).strip()
        logger.debug("Extracted JSON block from markdown.")
        return cleaned
    else:
        # If no markdown, assume the whole text might be JSON (or close)
        cleaned = response_text.strip()
        # Basic cleanup: remove potential leading/trailing non-JSON chars if necessary
        # Be careful not to remove valid JSON start/end chars
        # cleaned = cleaned.lstrip(...).rstrip(...)
        logger.debug("No markdown found, using stripped response text.")
        return cleaned

# --- Batch AI Call Functions ---

def call_ai_batch_drill(
    drill_inputs: List[Dict[str, Any]],
    model: GenerativeModel
) -> List[Dict[str, Any]]:
    """Processes a batch of drill utterances via a single API call."""
    if not drill_inputs:
        return []

    logger.info(f"Starting BATCH API call for {len(drill_inputs)} drill items.")
    formatted_inputs = "\n".join([format_utterance_for_prompt({**item, 'priority_level': 'DRILL'}) for item in drill_inputs]) # Add marker

    prompt = f"""
Context:
You are an AI tutor creating focused language drills based on a batch of learner utterances and corrections. For each utterance provided, generate a teaching module focused on reinforcing the specified grammatical pattern or vocabulary item.

Input Batch:
Below is a list of utterances needing drill practice. Each item includes conversation context, the original utterance, the correction, the target pattern/topic, the reason for the drill, pattern analysis with usage frequency and context, and a unique 'Utterance Identifier'.

{formatted_inputs}

Your Task:
Generate a single JSON object containing a key "generated_modules". The value should be a list, where each element corresponds to one input utterance and follows the structure below. Include the 'Utterance Identifier' from the input in your output for mapping.

Output JSON Structure for EACH module in the list:
{{
  "utterance_index": "string (Copy the Utterance Identifier from the corresponding input item)",
  "explanations": {{
    "introduction": "string (1-2 sentences referencing the Conversation Context and pattern frequency data from pattern_analysis to establish relevance)",
    "main_explanation": "string (3-4 sentences explaining the target pattern/topic, mentioning its frequency, typical contexts of use from pattern_analysis, and common misconceptions)",
    "recap": "string (1 sentence summarizing the key takeaway for this drill)"
  }},
  "problems": [
    // 5-8 Dictation problems focusing ONLY on the target pattern/topic
    {{
      "problem_id": "string (e.g., 'dictation_1')",
      "type": "Dictation",
      "target_text": "string (Sentence for dictation, reinforcing the target pattern with natural language similar to conversation context)",
      "prompt": "string (Brief hint focusing on the specific intention from pattern_analysis)",
      "difficulty": "string (Either 'Basic', 'Intermediate', or 'Advanced' based on pattern frequency - higher frequency = more basic)"
    }},
    // ... more Dictation problems ...
  ],
  "help_context": "string (1 sentence describing the specific focus of THIS drill set, mentioning both the pattern and its typical usage context)"
}}

Instructions:
1. Process EACH utterance in the input batch.
2. For each, generate one corresponding JSON module according to the structure above.
3. **Introduction:** Reference the `Conversation Context` and frequency data from `pattern_analysis` to establish relevance.
4. **Main Explanation:** Explain not just what the pattern is, but when and how it's used (context from pattern_analysis).
5. **Dictation Problems:** 
   - Create 5-8 unique sentences per module that directly practice the specific `Pattern/Topic`
   - Vary the difficulty based on the frequency data (higher frequency = more basic sentences)
   - Include everyday situations similar to those in the conversation context
   - Ensure at least 2 problems incorporate words/phrases from the original context
6. **Accuracy:** Ensure `utterance_index` is copied correctly.
7. **Output Format:** Return a single JSON object: `{{"generated_modules": [module1, module2, ...]}}`.

Example Input (excerpt):
---
Utterance Identifier: 12
Conversation Context Leading Up To This Utterance:
interviewer: I see. Tennis elbow?
student: Tennis elbow?
interviewer: Uhu.

Learner's Utterance:
Original: "Umm. But the reason is not tennis"
Corrected: "But it's not from tennis"

Identified Target for Drill Practice:
Pattern/Topic: it's not from [source]
Reason: The learner used a less natural expression for negating the source of something
---

Example Output Module (excerpt):
{{
  "utterance_index": "12",
  "explanations": {{
    "introduction": "In your conversation about tennis elbow, you used 'the reason is not tennis' instead of the more natural expression 'it's not from tennis'. This is a very common pattern in English for indicating something doesn't originate from a source.",
    "main_explanation": "When denying the source or origin of something, English speakers typically use the structure 'it's not from X' rather than 'the reason is not X'. The pattern 'from [source]' is extremely common in everyday conversation when talking about origins. The negated form 'it's not from [source]' follows the same high-frequency pattern and sounds more natural to native speakers.",
    "recap": "Use 'it's not from [source]' when denying the origin of something rather than 'the reason is not [source]'."
  }},
  "problems": [
    {{
      "problem_id": "dictation_1",
      "type": "Dictation",
      "target_text": "The pain is not from playing tennis, it's from typing too much.",
      "prompt": "Stating the origin of pain (negation + source)",
      "difficulty": "Basic"
    }},
    ...
  ],
  "help_context": "This drill focuses on using 'it's not from [source]' to naturally express negation of origins in everyday conversation."
}}
"""
    # logger.debug(f"--- BATCH DRILL PROMPT ---\n{prompt}\n--- END BATCH DRILL PROMPT ---")
    response_text = call_gemini_api(prompt, model)
    return parse_batch_response(response_text, "Drill", len(drill_inputs))


def call_ai_batch_low_priority(
    low_priority_inputs: List[Dict[str, Any]],
    model: GenerativeModel
) -> List[Dict[str, Any]]:
    """Processes a batch of low-priority utterances via a single API call."""
    if not low_priority_inputs:
        return []

    logger.info(f"Starting BATCH API call for {len(low_priority_inputs)} LOW priority items.")
    formatted_inputs = "\n".join([format_utterance_for_prompt(item) for item in low_priority_inputs])

    prompt = f"""
Context:
You are an AI tutor creating focused explanations based on a batch of learner utterances identified as low priority issues. For each utterance provided, generate a concise explanation module *without* interactive problems. Pay attention to the specific errors identified and pattern analysis data.

Input Batch:
Below is a list of utterances needing brief explanations. Each item includes conversation context, original/corrected text, the identified issue with detailed error analysis, pattern analysis showing frequency and usage context, and a unique 'Utterance Identifier'.

{formatted_inputs}

Your Task:
Generate a single JSON object containing a key "generated_modules". The value should be a list, where each element corresponds to one input utterance and follows the structure below. Include the 'Utterance Identifier' from the input in your output for mapping.

Output JSON Structure for EACH module in the list:
{{
  "utterance_index": "string (Copy the Utterance Identifier from the corresponding input item)",
  "explanations": {{
    "introduction": "string (1-2 sentences directly referencing the specific conversation context and the learner's actual utterance)",
    "main_explanation": "string (2-4 sentences explaining the specific error mentioned in clauses[].errors and its correction, using pattern frequency data to emphasize importance)",
    "recap": "string (1 sentence with actionable advice the learner can apply immediately)"
  }},
  "problems": [], // MUST be an empty list
  "help_context": "string (1 sentence describing the specific pattern or error from pattern_analysis, mentioning its frequency of use)"
}}

Instructions:
1. Process EACH utterance in the input batch.
2. For each, generate one corresponding JSON module according to the structure above.
3. **Use Specific Context:** Reference the exact words from the conversation and the learner's actual error.
4. **Explanations:** 
   - Focus on WHY the correction is better, not just WHAT the correction is
   - Use the detailed error information from the clauses[].errors field
   - Reference frequency and context data from pattern_analysis to explain importance
   - Keep explanations short but concrete with at least one clear example
5. **Problems Array:** Ensure `problems` is always `[]`.
6. **Accuracy:** Ensure `utterance_index` is copied correctly.
7. **Output Format:** Return a single JSON object: `{{"generated_modules": [module1, module2, ...]}}`.

Example Input (excerpt):
---
Utterance Identifier: 8
Conversation Context Leading Up To This Utterance:
interviewer: I see. Tennis elbow?
student: Tennis elbow?
interviewer: Uhu.

Learner's Utterance:
Original: "Umm. But the reason is not tennis"
Corrected: "But it's not from tennis"

Identified Issue (Priority: LOW):
Topic/Error: Sentence Structure
Reasoning: This is a relatively minor awkwardness. A quick explanation of 'it's from' vs 'the reason is' should suffice.
Specific Errors Found:
- Clause: 'the reason is not tennis' -> Error (moderate Sentence Structure): The sentence structure is awkward. The phrase 'the reason is not tennis' is grammatically correct but less natural than 'it's not from tennis'. (Suggestion: it's not from tennis)
---

Example Output Module (excerpt):
{{
  "utterance_index": "8",
  "explanations": {{
    "introduction": "In your conversation about tennis elbow, you said 'the reason is not tennis' which is grammatically correct but sounds slightly unnatural in English.",
    "main_explanation": "While 'the reason is not tennis' is technically correct, native speakers typically use 'it's not from tennis' when denying the source or origin of something. The expression 'from [source]' is very commonly used (frequency 4.1/5) in everyday conversation when talking about where something comes from or originates.",
    "recap": "When indicating something doesn't originate from a particular source, use 'it's not from [source]' rather than 'the reason is not [source]' for more natural-sounding English."
  }},
  "problems": [],
  "help_context": "This explanation focuses on the high-frequency pattern 'from [source]' (4.1/5) used for stating origins in natural English conversation."
}}
"""
    # logger.debug(f"--- BATCH LOW PRIORITY PROMPT ---\n{prompt}\n--- END BATCH LOW PRIORITY PROMPT ---")
    response_text = call_gemini_api(prompt, model)
    return parse_batch_response(response_text, "Low Priority", len(low_priority_inputs))

def call_ai_batch_med_high_priority(
    med_high_priority_inputs: List[Dict[str, Any]],
    model: GenerativeModel
) -> List[Dict[str, Any]]:
    """Processes a batch of medium/high-priority utterances via a single API call."""
    if not med_high_priority_inputs:
        return []

    logger.info(f"Starting BATCH API call for {len(med_high_priority_inputs)} MED/HIGH priority items.")
    formatted_inputs = "\n".join([format_utterance_for_prompt(item) for item in med_high_priority_inputs])

    prompt = f"""
Context:
You are an AI tutor creating comprehensive teaching modules based on a batch of learner utterances identified as medium or high priority issues. For each utterance, generate detailed explanations AND 3 practice problems. Base the content on the identified errors, pattern analysis, and conversation context.

Input Batch:
Below is a list of utterances needing teaching modules. Each item includes conversation context, original/corrected text, detailed error analysis in clauses, pattern analysis with frequency and usage context, and a unique 'Utterance Identifier'.

{formatted_inputs}

Your Task:
Generate a single JSON object containing a key "generated_modules". The value should be a list, where each element corresponds to one input utterance and follows the structure below. Include the 'Utterance Identifier'.

Output JSON Structure for EACH module in the list:
{{
  "utterance_index": "string (Copy the Utterance Identifier from the corresponding input item)",
  "explanations": {{
    "introduction": "string (1-2 sentences directly referencing the specific conversation context and highlighting the importance of the pattern based on frequency data)",
    "main_explanation": "string (3-5 sentences explaining the error specifically mentioned in clauses[].errors, WHY it matters, WHEN to use the correct form, and providing 2 contrasting examples)",
    "recap": "string (1-2 sentences summarizing the key takeaway with an actionable rule the learner can apply)"
  }},
  "problems": [
    // EXACTLY 3 problems: Mix of MultipleChoiceQuestion (MCQ) and FillBlankChoice
    {{
      "problem_id": "string (e.g., 'mcq_1')",
      "type": "MultipleChoiceQuestion",
      "question": "string (Question directly related to the same error pattern found in the original utterance)",
      "context": "string (Brief situational context similar to the conversation context)",
      "options": [
        {{"option_id": "A", "text": "string", "is_correct": boolean, "feedback": "string (Explain why right/wrong, referencing the specific error pattern)"}},
        // 3-4 options total, with at least one option containing the SAME ERROR TYPE as the original utterance
      ]
    }},
    {{
      "problem_id": "string (e.g., 'fill_blank_1')",
      "type": "FillBlankChoice",
      "question_template": "string (Sentence with '{{{{blank}}}}' placeholder, within a context similar to the original conversation)",
      "options": [
         {{"option_id": "A", "text": "string", "is_correct": boolean, "feedback": "string (Explain fit based on the pattern analysis intention and frequency)"}},
         // 3-4 options total
      ]
    }},
    // ... 1 more problem (MCQ or FillBlankChoice) ...
  ],
  "help_context": "string (1 sentence describing the specific error pattern and its frequency/importance from pattern_analysis)"
}}

Instructions:
1. Process EACH utterance in the input batch.
2. For each, generate one corresponding JSON module according to the structure above.
3. **Content:** 
   - Link explanations directly to the specific error in clauses[].errors
   - Use pattern_analysis data to explain WHY this language pattern matters (frequency, context)
   - For HIGH priority items, provide more detailed explanations and slightly more challenging problems
4. **Problems:** 
   - Exactly 3 problems per module (mix of MCQ/FillBlankChoice)
   - Each problem should test the SAME ERROR PATTERN that appeared in the original utterance
   - Create situations similar to the original conversation context
   - Include at least one question that tests understanding of WHEN to use this pattern
   - Provide detailed feedback for *every* option explaining why it's right/wrong
5. **Accuracy:** Ensure `utterance_index` is copied correctly.
6. **Output Format:** Return a single JSON object: `{{"generated_modules": [module1, module2, ...]}}`.

Example Input (excerpt):
---
Utterance Identifier: 15
Conversation Context Leading Up To This Utterance:
interviewer: Do you know what causes this?
student: I think maybe sleeping position.

Learner's Utterance:
Original: "Maybe I sleep on bad position in night."
Corrected: "Maybe I sleep in a bad position at night."

Identified Issue (Priority: MEDIUM):
Topic/Error: Preposition Usage
Reasoning: The error involves multiple preposition mistakes which affect comprehension.
Specific Errors Found:
- Clause: 'I sleep on bad position in night' -> Error (significant Preposition Usage): Incorrect prepositions. 'Sleep on bad position' should be 'sleep in a bad position' and 'in night' should be 'at night'. (Suggestion: I sleep in a bad position at night)
---

Example Output Module (excerpt):
{{
  "utterance_index": "15",
  "explanations": {{
    "introduction": "In your conversation about causes of pain, you said 'Maybe I sleep on bad position in night' which contains two common preposition errors that English learners often make.",
    "main_explanation": "There are two preposition errors to correct. First, when talking about body positions during activities like sleeping, English uses 'in a position' rather than 'on position'. The preposition 'in' is used because we're talking about how the body is arranged or configured. Second, we say 'at night' (not 'in night') as a standard time expression. Compare: 'I sleep on my back' (specific body part) vs. 'I sleep in a comfortable position' (configuration of the body).",
    "recap": "Remember to use 'in a position' when describing body configuration and 'at night' when referring to nighttime as a general period."
  }},
  "problems": [
    {{
      "problem_id": "mcq_1",
      "type": "MultipleChoiceQuestion",
      "question": "Which sentence uses prepositions correctly?",
      "context": "Talking about sleeping habits",
      "options": [
        {{"option_id": "A", "text": "I usually sleep on bad position in night.", "is_correct": false, "feedback": "This contains both errors from your original sentence: 'on bad position' should be 'in a bad position' and 'in night' should be 'at night'."}},
        {{"option_id": "B", "text": "I usually sleep in a bad position at night.", "is_correct": true, "feedback": "Correct! We use 'in a position' to describe configuration and 'at night' for the nighttime period."}},
        {{"option_id": "C", "text": "I usually sleep in bad position on night.", "is_correct": false, "feedback": "This is partially correct with 'in position' but still has an error with 'on night' which should be 'at night'."}}
      ]
    }},
    ...
  ],
  "help_context": "This module focuses on correct preposition usage with positions ('in a position') and time expressions ('at night'), both high-frequency patterns in English."
}}
"""
    # logger.debug(f"--- BATCH MED/HIGH PRIORITY PROMPT ---\n{prompt}\n--- END BATCH MED/HIGH PRIORITY PROMPT ---")
    response_text = call_gemini_api(prompt, model)
    return parse_batch_response(response_text, "Med/High Priority", len(med_high_priority_inputs))


# --- Main Processing Logic ---

def process_files(
    conv_file: Path,
    drill_file: Path,
    output_file: Path,
    model: GenerativeModel
) -> None:
    """Loads data, processes utterances concurrently, generates modules, and saves the output."""
    start_time = time.time()
    all_modules = []
    low_priority_inputs = []
    med_high_priority_inputs = []
    drill_inputs = []
    total_utterances_loaded = 0
    utterance_counter = 0 # Unique index across both files

    # --- Load and Process Conversational Data --- 
    logger.info(f"Loading conversational file: {conv_file}")
    conversational_data = load_json_file(conv_file)
    if conversational_data and isinstance(conversational_data, dict) and 'utterances' in conversational_data:
        conversational_utterances = conversational_data['utterances'] # Access the list
        if isinstance(conversational_utterances, list):
            logger.info(f"Loaded {len(conversational_utterances)} conversational utterances.")
            for utt_data in conversational_utterances: # Iterate over the list
                if not isinstance(utt_data, dict):
                    logger.warning(f"Skipping non-dictionary item in conversational utterances: {type(utt_data)} - {str(utt_data)[:100]}")
                    continue

                total_utterances_loaded += 1
                utterance_counter += 1
                utt_data['utterance_index'] = utterance_counter # Add unique index

                # Determine priority
                priority = determine_priority(utt_data)

                # Assign to appropriate batch list
                if priority == 'LOW':
                    low_priority_inputs.append(utt_data)
                elif priority in ['MEDIUM', 'HIGH']:
                    med_high_priority_inputs.append(utt_data)
                else:
                    logger.warning(f"Unknown or invalid priority '{priority}' for conv utterance index {utterance_counter}. Skipping.")
        else:
            logger.warning(f"'utterances' key in {conv_file} is not a list.")
    elif conversational_data:
         logger.warning(f"Conversational file {conv_file} loaded but is not a dictionary or lacks 'utterances' key.")
    else:
         logger.warning(f"Conversational input file not found or failed to load: {conv_file}") # Already logged in load_json_file

    # --- Load and Process Drill Data ---
    logger.info(f"Loading drill file: {drill_file}")
    drill_data = load_json_file(drill_file)
    if drill_data and isinstance(drill_data, dict) and 'utterances' in drill_data:
        drill_utterances = drill_data['utterances'] # Access the list
        if isinstance(drill_utterances, list):
            logger.info(f"Loaded {len(drill_utterances)} drill utterances.")
            for utt_data in drill_utterances: # Iterate over the list
                if not isinstance(utt_data, dict):
                    logger.warning(f"Skipping non-dictionary item in drill utterances: {type(utt_data)} - {str(utt_data)[:100]}")
                    continue

                total_utterances_loaded += 1
                utterance_counter += 1
                utt_data['utterance_index'] = utterance_counter # Add unique index

                # Add to drill batch list
                drill_inputs.append(utt_data)
        else:
            logger.warning(f"'utterances' key in {drill_file} is not a list.")
    elif drill_data:
         logger.warning(f"Drill file {drill_file} loaded but is not a dictionary or lacks 'utterances' key.")
    else:
        logger.warning(f"Drill input file not found or failed to load: {drill_file}") # Already logged in load_json_file

    logger.info(f"Total utterances prepared for processing: {len(low_priority_inputs)} Low, {len(med_high_priority_inputs)} Med/High, {len(drill_inputs)} Drill.")

    # --- Concurrent Batch API Calls ---
    all_generated_modules_from_ai = []
    logger.info("Starting concurrent batch processing...")
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_batch_type = {}
        if drill_inputs:
            future_to_batch_type[executor.submit(call_ai_batch_drill, drill_inputs, model)] = "Drill"
        if low_priority_inputs:
             future_to_batch_type[executor.submit(call_ai_batch_low_priority, low_priority_inputs, model)] = "Low Priority"
        if med_high_priority_inputs:
            future_to_batch_type[executor.submit(call_ai_batch_med_high_priority, med_high_priority_inputs, model)] = "Med/High Priority"

        if not future_to_batch_type:
            logger.warning("No inputs found for any batch type. Exiting.")
            return

        for future in concurrent.futures.as_completed(future_to_batch_type):
            batch_type = future_to_batch_type[future]
            try:
                # Result is expected to be a list of modules, each with 'utterance_index'
                batch_results = future.result()
                logger.info(f"Successfully completed batch: {batch_type}. Received {len(batch_results)} modules.")
                all_generated_modules_from_ai.extend(batch_results)
            except Exception as exc:
                logger.error(f'{batch_type} batch generated an exception: {exc}')
                # Consider more detailed logging or re-raising depending on desired behavior

    end_time = time.time()
    logger.info(f"Concurrent API calls finished in {end_time - start_time:.2f} seconds.")

    # --- Combine AI output with Source Info ---
    final_output_modules = []
    # --- Ensure we have the lists of utterances for indexing ---
    # Use the lists populated earlier in the function
    # conversational_utterances_list = conversational_utterances if conversational_utterances else []
    # drill_utterances_list = drill_utterances if drill_utterances else []
    # Correction: Need the original full lists loaded at the start, not the filtered ones
    conversational_utterances_list = conversational_data.get('utterances', []) if conversational_data and isinstance(conversational_data.get('utterances'), list) else []
    drill_utterances_list = drill_data.get('utterances', []) if drill_data and isinstance(drill_data.get('utterances'), list) else []
    # ----------------------------------------------------------

    for module_data in all_generated_modules_from_ai:
        utterance_index_raw = module_data.get("utterance_index")
        utterance_index = None

        # --- Try converting index to positive integer --- 
        if utterance_index_raw is not None:
            try:
                utterance_index_int = int(utterance_index_raw)
                if utterance_index_int > 0:
                    utterance_index = utterance_index_int
                else:
                    logger.warning(f"Generated module has non-positive utterance_index '{utterance_index_raw}'. Skipping.")
            except (ValueError, TypeError):
                logger.warning(f"Generated module has invalid utterance_index '{utterance_index_raw}'. Skipping.")
        else:
             logger.warning(f"Generated module missing utterance_index. Skipping.")

        if utterance_index is None: # Skip if conversion failed or index was missing/invalid
            continue
        # -------------------------------------------------

        # --- Retrieve original source info based on the unified index --- 
        source_info = None
        conv_count = len(conversational_utterances_list) # Use the loaded list length
        drill_count = len(drill_utterances_list) # Use the loaded list length

        if 1 <= utterance_index <= conv_count:
             # It's a conversational utterance
            source_info = conversational_utterances_list[utterance_index - 1]
            priority = determine_priority(source_info) # Re-determine priority
            if priority == 'LOW':
                module_type = "CONVERSATIONAL_LOW"
            elif priority == 'MEDIUM':
                module_type = "CONVERSATIONAL_MEDIUM"
            elif priority == 'HIGH':
                module_type = "CONVERSATIONAL_HIGH"
            else:
                module_type = "CONVERSATIONAL_UNKNOWN"
        elif conv_count < utterance_index <= (conv_count + drill_count):
            # It's a drill utterance
            drill_list_index = utterance_index - conv_count - 1
            if 0 <= drill_list_index < drill_count:
                 source_info = drill_utterances_list[drill_list_index]
                 module_type = "DRILL_PRACTICE"
            else:
                 logger.error(f"Calculated drill index {drill_list_index} out of bounds for utterance index {utterance_index}. Skipping.")
        else:
            logger.error(f"Utterance index {utterance_index} out of bounds for both conversational and drill lists. Skipping.")
            continue

        final_module = {
            "module_id": utterance_index, # Use the index as the base ID
            "module_type": module_type,
            "source_utterance_info": {
                "original": source_info.get('original'),
                "corrected": source_info.get('corrected'),
                "priority_level": source_info.get('filtering_metadata', {}).get('priority_level'), # Add priority if available
                "target_pattern": source_info.get('filtering_metadata', {}).get('target_pattern')
            },
            # Get generated content, ensure keys exist
            "explanations": module_data.get("explanations", {"introduction": "Error: Missing explanation", "main_explanation": "", "recap": ""}),
            "problems": module_data.get("problems", []),
            "help_context": module_data.get("help_context", "Error: Missing help context")
        }
        final_output_modules.append(final_module)

    logger.info(f"Processing finished. Generated {len(final_output_modules)} total modules from {total_utterances_loaded} loaded utterances.")
    save_output(final_output_modules, output_file)

# --- Main Execution Logic ---
if __name__ == "__main__":
    # --- Reinstate Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Generate teaching modules from conversational and drill practice JSON files using batch processing."
    )
    parser.add_argument(
        "--conversational-file",
        type=Path,
        required=True,
        help="Path to the input JSON file containing conversational utterances."
    )
    parser.add_argument(
        "--drill-file",
        type=Path,
        required=True,
        help="Path to the input JSON file containing drill practice utterances."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to save the combined output JSON file."
    )
    args = parser.parse_args()

    # Use parsed arguments
    conversational_file = args.conversational_file
    drill_file = args.drill_file
    output_file = args.output_file

    logger.info("--- Teaching Module Generation Script Started (Batch Mode) ---")
    logger.info(f"Conversational Input: {conversational_file}")
    logger.info(f"Drill Input: {drill_file}")
    logger.info(f"Output File: {output_file}")

    script_start_time = time.time()
    process_files(conversational_file, drill_file, output_file, model)
    script_end_time = time.time()

    logger.info("--- Teaching Module Generation Script Finished ---")
    logger.info(f"Total execution time: {script_end_time - script_start_time:.2f} seconds")
    logger.info("----------------------------------------")
