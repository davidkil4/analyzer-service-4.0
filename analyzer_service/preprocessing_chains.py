import os
import sys
import logging
from dotenv import load_dotenv
from pathlib import Path
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from functools import partial
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser, PydanticOutputParser, JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI

# Import schemas and utils
from .schemas import (
    InputUtterance,
    TranslatedUtterance, 
    SegmentedASUnit,
    Clause, 
    ClauseListOutput, 
    ASUnitWithClauses,
    AlignedASUnitWithClauses,
    ClauseAlignmentOutput, 
    AlignedClause,
    PreprocessingOutput,
    PreprocessedASUnit # Added PreprocessedASUnit
)

logger = logging.getLogger(__name__)
load_dotenv()

# --- Constants ---
# Define project root relative to this file's location
PROJECT_ROOT = Path(__file__).parent.parent
PROMPT_DIR = PROJECT_ROOT / "analyzer_service" / "prompts"

# --- Helper Functions ---
def load_prompt_from_file(file_path: Path) -> str:
    """Loads text content from a file.

    Args:
        file_path: The Path object representing the file to load.

    Returns:
        The content of the file as a string, or an error message if loading fails.
    """
    try:
        return file_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {file_path}")
        return f"Error: Prompt file not found at {file_path}"
    except Exception as e:
        logger.error(f"Error loading prompt from {file_path}: {e}")
        return ""

class NewlineListOutputParser(BaseOutputParser[List[str]]):
    """Parses the LLM output assuming items are separated by newlines."""
    def parse(self, text: str) -> List[str]:
        """Parses the output of an LLM call."""
        # Simple split by newline, removing empty strings and stripping whitespace
        items = [line.strip() for line in text.strip().split('\n') if line.strip()]
        if not items:
             logger.warning(f"NewlineListOutputParser got empty or whitespace-only text: '{text}'. Returning empty list.")
             return []
        logger.debug(f"Parsed items: {items}")
        return items

def count_words(text: str) -> int:
    """Counts the number of words in a text string (splits by whitespace)."""
    if not text:
        return 0
    return len(text.split())

def filter_short_translated_utterance(translated_utterance: TranslatedUtterance | None, min_words: int = 3) -> TranslatedUtterance | None:
    """
    Filters out TranslatedUtterance objects if their translated_text has fewer than min_words.
    Returns the object if it passes the filter, otherwise None.
    Handles potential None input gracefully.
    """
    if translated_utterance is None:
        logger.debug("Input to filter_short_translated_utterance was None, skipping.")
        return None # Pass None through

    word_count = count_words(translated_utterance.translated_text)
    if word_count >= min_words:
        logger.debug(f"Utterance passed filter (ID: {translated_utterance.original_utterance.id}, Words: {word_count}): '{translated_utterance.translated_text[:50]}...'" )
        return translated_utterance
    else:
        logger.info(f"Filtering out short utterance (ID: {translated_utterance.original_utterance.id}, Words: {word_count}): '{translated_utterance.translated_text}'")
        return None # Signal to filter this item

# Segmentation formatting function
def format_segmented_units(segmentation_result: Dict[str, Any]) -> List[SegmentedASUnit]:
    """
    Takes the output dictionary containing the original utterance and segmented text list,
    and formats it into a list of SegmentedASUnit objects.
    Handles cases where segmentation output is None or empty.
    """
    original_utterance = segmentation_result.get('original_utterance')
    segmented_texts = segmentation_result.get('segmented_texts')
    original_input_text = segmentation_result.get('original_input_text', '') # Get original text

    # Use original_utterance.text if original_input_text wasn't explicitly passed (fallback)
    if not original_input_text and original_utterance:
         original_input_text = original_utterance.text
         logger.debug("Using original_utterance.text as fallback for original_input_text in formatting.")

    if not original_utterance or not segmented_texts:
        logger.warning(f"Input missing for formatting segmented units. Utterance ID: {original_utterance.id if original_utterance else 'N/A'}. Segments found: {bool(segmented_texts)}. Returning empty list.")
        return [] # Return empty list if input is incomplete or segmentation failed

    segmented_units = []
    for i, unit_text in enumerate(segmented_texts):
        as_unit_id = f"u{original_utterance.id}-as{i+1}" # Generate ID
        unit = SegmentedASUnit(
            as_unit_id=as_unit_id,
            original_utterance_id=original_utterance.id, # Store original ID for reference
            original_input_text=original_input_text, # Assign original text
            as_unit_text=unit_text.strip() # Ensure schema field name matches 'as_unit_text'
        )
        segmented_units.append(unit)
        logger.debug(f"Formatted AS Unit: {unit}")

    return segmented_units

def analyze_and_format_clauses(as_unit: SegmentedASUnit, analyzer: Runnable) -> ASUnitWithClauses:
    """Analyzes clauses in an AS unit and formats the output."""
    logger.debug(f"Analyzing clauses for AS Unit: {as_unit.as_unit_id}")
    try:
        # Provide all variables expected by the clause_analysis_prompt
        input_dict = {
            "input_as_units": as_unit.as_unit_text,
            "input_format": "Single AS Unit text", # Describe the input format
            "output_format": "JSON object with a single key 'clauses' containing a list of strings" # Describe the desired output format
        }
        clause_list_output = analyzer.invoke(input_dict)

        # Ensure clauses is a list, even if parsing returns None or incorrect type
        clauses = clause_list_output.clauses if isinstance(clause_list_output, ClauseListOutput) and clause_list_output.clauses else []

        logger.debug(f"Successfully parsed {len(clauses)} clauses for AS Unit: {as_unit.as_unit_id}")
    except Exception as e: # Catch potential parsing or LLM errors
        logger.error(f"Error analyzing clauses for AS Unit {as_unit.as_unit_id}: {e}", exc_info=True)
        clauses = [] # Return empty list on error

    # Combine original AS unit data with the new clauses
    return ASUnitWithClauses(**as_unit.model_dump(), clauses=clauses)

def parse_alignment_output(text: str) -> Optional[str]:
    """Parses the raw output from the alignment LLM call."""
    # Remove potential prefix and strip whitespace
    cleaned_text = re.sub(r"^Aligned Original Segment:\s*", "", text.strip()).strip()
    if cleaned_text == "NO_ALIGNMENT_FOUND" or not cleaned_text:
        return None
    return cleaned_text

def align_as_unit_and_format(as_unit_with_clauses: ASUnitWithClauses, aligner: Runnable) -> AlignedASUnitWithClauses:
    """Aligns the AS unit text with original Korean and updates the object."""
    logger.debug(f"Aligning AS Unit text for: {as_unit_with_clauses.as_unit_id}")
    try:
        # Provide all variables expected by the as_unit_alignment_prompt
        input_dict = {
            "original_input_text": as_unit_with_clauses.original_input_text,
            "target_as_unit_text": as_unit_with_clauses.as_unit_text # Map field to prompt variable
        }
        aligned_text = aligner.invoke(input_dict)

        # Update the object with the aligned text
        # Use dict() and ** to create a new instance with the updated field
        return AlignedASUnitWithClauses(**as_unit_with_clauses.dict(), aligned_original_text=aligned_text)
    except Exception as e:
        logger.error(f"Error aligning AS Unit {as_unit_with_clauses.as_unit_id}: {e}", exc_info=True)
        return AlignedASUnitWithClauses(
            **as_unit_with_clauses.dict(),
            aligned_original_text=None
        )

def parse_clause_alignment_json_or_none(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to parse the LLM output as JSON.
    Returns None if the input is 'NO_ALIGNMENT_FOUND' or parsing fails.
    """
    cleaned_text = text.strip()
    if cleaned_text == "NO_ALIGNMENT_FOUND":
        return None
    try:
        # Find JSON block if wrapped in markdown ```json ... ```
        match = re.search(r"```json\s*([\s\S]*?)\s*```", cleaned_text)
        if match:
            json_str = match.group(1)
        else:
            json_str = cleaned_text # Assume it's raw JSON

        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse clause alignment JSON: {e}. Raw text: '{text}'")
        return None

def perform_clause_alignment_and_final_format(
    aligned_as_unit_wc: AlignedASUnitWithClauses, # Output from AS Unit Alignment
    clause_aligner: Runnable
) -> List[PreprocessedASUnit]:
    """
    Performs clause alignment for each AS unit and formats the final output.
    """
    aligned_clauses = []
    for clause in aligned_as_unit_wc.clauses:
        try:
            logger.debug(f"Aligning clause: '{clause.clause_text}' for AS Unit: {aligned_as_unit_wc.as_unit_id}")
            # Prepare input for clause aligner, matching the prompt variables
            input_dict = {
                "aligned_original_text": aligned_as_unit_wc.aligned_original_text or aligned_as_unit_wc.original_input_text, # Use aligned if available, else original
                "target_clause_text": clause.clause_text,
                "input_format": "Original AS Unit text and a target clause from it.", # Describe input
                "output_format": "JSON object with keys: aligned_original_clause_segment (string), is_korean (boolean), original_clause_type ('word' or 'phrase')" # Describe output
            }
            alignment_result: ClauseAlignmentOutput = clause_aligner.invoke(input_dict) # Use the full input dict
            logger.debug(f"[DEBUG] Clause Aligner Output for '{clause.clause_text}': {alignment_result}") # <-- DEBUG LOG ADDED HERE

            # Update the original clause with alignment info
            # Ensure alignment_result is not None and is the correct type before accessing attributes
            if isinstance(alignment_result, ClauseAlignmentOutput):
                aligned_clause = AlignedClause(
                    **clause.dict(), # Copy original clause fields
                    aligned_original_clause_segment=alignment_result.aligned_original_clause_segment,
                    is_korean=alignment_result.is_korean,
                    original_clause_type=alignment_result.original_clause_type
                )
                aligned_clauses.append(aligned_clause)
            else:
                 logger.warning(f"Clause alignment for '{clause.clause_text}' in {aligned_as_unit_wc.as_unit_id} did not return expected ClauseAlignmentOutput. Got: {type(alignment_result)}")
                 # Append original clause without alignment info if parsing fails
                 aligned_clauses.append(AlignedClause(**clause.dict()))

        except Exception as e:
            logger.error(f"Error aligning clause '{clause.clause_text}' for AS Unit {aligned_as_unit_wc.as_unit_id}: {e}", exc_info=True)
        
    preprocessed_as_unit = PreprocessedASUnit(
        **aligned_as_unit_wc.model_dump(exclude={'clauses'}), # Exclude original clauses
        clauses=aligned_clauses # Set the new aligned clauses
    )
    return [preprocessed_as_unit] # Return the list containing the single unit

def get_preprocessing_chain() -> Runnable:
    """
    Constructs and returns the full LangChain preprocessing chain.
    Outputs a PreprocessingOutput object.

    Steps:
    1. Translation/Normalization
    2. Filtering
    3. Segmentation
    4. Clause Analysis
    5. AS Unit Alignment
    6. Clause Alignment & Final Formatting

    Returns:
        A LangChain Runnable that takes an InputUtterance and returns a PreprocessingOutput.
    """
    # Initialize LLM within the function
    LLM_MODEL_NAME = "gemini-2.0-flash" 
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", # Explicitly set the model
            temperature=0.1, # Low temp for more deterministic translation
            # convert_system_message_to_human=True # Might be needed depending on Gemini version
        )
        logger.info(f"Initialized LLM within get_preprocessing_chain: {LLM_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM {LLM_MODEL_NAME} within get_preprocessing_chain: {e}")
        # If LLM fails here, the chain cannot proceed.
        # Return a lambda that immediately returns a failure state.
        # Note: This assumes PreprocessingOutput schema exists and can be instantiated.
        def _failed_chain_init(input_utterance: InputUtterance) -> PreprocessingOutput:
            return PreprocessingOutput(processed_utterances=[], skipped=True, skip_reason=f"LLM Initialization failed: {e}")
        return RunnableLambda(_failed_chain_init)

    # --- 1. Translation and Normalization Chain ---
    translation_prompt_text = load_prompt_from_file(PROMPT_DIR / "translation_prompt.txt")
    translation_prompt = ChatPromptTemplate.from_template(translation_prompt_text)
    translation_chain = (
        {"original_utterance": RunnablePassthrough()} # Pass the whole InputUtterance
        | RunnablePassthrough.assign( # Keep original_input_text separate
            original_input_text=lambda x: x['original_utterance'].text
          )
        # Map the utterance text to the key expected by the prompt
        | RunnablePassthrough.assign(
            korean_english_text=lambda x: x['original_utterance'].text
          )
        | {"original_utterance": lambda x: x['original_utterance'],
           "original_input_text": lambda x: x['original_input_text'],
           # Pass the correct key to the prompt and use the local llm instance
           "translated_text": (lambda x: {"korean_english_text": x['korean_english_text']}) | translation_prompt | llm | StrOutputParser()
           }
        | RunnableLambda(lambda x: TranslatedUtterance(**x)) # Convert to Pydantic
    )
    logger.info("Translation chain created.")

    # --- 2. Filter Short Utterances ---
    filter_chain = RunnableLambda(filter_short_translated_utterance)
    logger.info("Filter chain created.")

    # --- 3. AS Unit Segmentation Chain ---
    segmentation_prompt_text = load_prompt_from_file(PROMPT_DIR / "segmentation_prompt.txt")
    segmentation_prompt = ChatPromptTemplate.from_template(segmentation_prompt_text)
    segmentation_runnable = (
        RunnablePassthrough.assign(
            # Assign keys needed by the prompt
            input_utterances=itemgetter("translated_text"),
            input_format=RunnableLambda(lambda x: "Single long utterance"), # Constant value
            output_format=RunnableLambda(lambda x: "One clause or phrase per line") # Constant value
        )
        # Select ONLY the keys the prompt expects
        | RunnableParallel(
            input_utterances=itemgetter("input_utterances"),
            input_format=itemgetter("input_format"),
            output_format=itemgetter("output_format")
        )
        | segmentation_prompt
        | llm
        | NewlineListOutputParser()
    )
    logger.info("Segmentation runnable created.")

    # --- 4. Clause Analysis Chain ---
    clause_analysis_prompt_text = load_prompt_from_file(PROMPT_DIR / "clause_analysis_prompt.txt")
    clause_analysis_prompt = ChatPromptTemplate.from_template(clause_analysis_prompt_text)
    clause_analyzer = (
        clause_analysis_prompt
        | llm
        | PydanticOutputParser(pydantic_object=ClauseListOutput)
    )
    logger.info("Clause analyzer created.")

    # --- 5. AS Unit Alignment Chain ---
    as_unit_alignment_prompt_text = load_prompt_from_file(PROMPT_DIR / "as_unit_alignment_prompt.txt")
    as_unit_alignment_prompt = ChatPromptTemplate.from_template(as_unit_alignment_prompt_text)
    as_unit_aligner = (
        as_unit_alignment_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(parse_alignment_output) # Parses "Aligned Text: ..."
    )
    logger.info("AS unit aligner created.")

    # --- 6. Clause Alignment Chain ---
    clause_alignment_prompt_text = load_prompt_from_file(PROMPT_DIR / "clause_alignment_prompt.txt")
    clause_alignment_prompt = ChatPromptTemplate.from_template(clause_alignment_prompt_text)
    # Expects JSON output, parses it, validates with Pydantic if needed (or just returns dict)
    clause_aligner = (
         clause_alignment_prompt
         | llm
         | PydanticOutputParser(pydantic_object=ClauseAlignmentOutput)
    )
    logger.info("Clause aligner created.")


    # --- Bundle Chains --- 
    all_chains = {
        "translation": translation_chain,
        "filter": filter_chain,
        "segmentation": segmentation_runnable,
        "clause_analysis": clause_analyzer,
        "as_unit_alignment": as_unit_aligner,
        "clause_alignment": clause_aligner,
    }

    # --- Create Full Chain with Orchestration Function --- 
    # Use partial to pass the chains dict to the helper function
    full_preprocessing_pipeline = RunnableLambda(
        partial(_run_full_preprocessing_step, chains=all_chains),
        name="FullPreprocessingPipeline" # Add a name for easier debugging/tracing
    )

    logger.info("Full preprocessing pipeline (RunnableLambda wrapping placeholder) created.")
    return full_preprocessing_pipeline

# --- Define create_segmented_as_unit helper ---
def create_segmented_as_unit(segmented_text: str, original_utterance: TranslatedUtterance, index: int) -> SegmentedASUnit:
    """Creates a SegmentedASUnit object from segmented text and original utterance info."""
    # Create a unique ID for the segmented unit (e.g., original_id-segN)
    # Access ID via original_utterance.original_utterance.id
    original_id = original_utterance.original_utterance.id
    as_unit_id = f"{original_id}-as{index+1}" # Use asX convention, 1-based index
    return SegmentedASUnit(
        as_unit_id=as_unit_id,
        original_utterance_id=original_id, # Pass the correct ID
        original_input_text=original_utterance.original_input_text, # Pass original text
        as_unit_text=segmented_text # Pass the segmented text itself
        # Removed fields not in SegmentedASUnit schema (speaker, timestamp, etc.)
    )
# --- End create_segmented_as_unit helper ---


# --- Orchestration Function (Placeholder) ---
def _run_full_preprocessing_step(
    input_utterance: InputUtterance,
    *,
    chains: Dict[str, Runnable]
) -> Optional[PreprocessingOutput]:
    """
    Orchestrates the entire preprocessing sequence for a single utterance.
    (Placeholder - Logic to be added incrementally)
    Handles filtering, segmentation, and mapping over sub-units/clauses.
    Returns PreprocessingOutput if successful and not filtered, else None.
    """
    logger.debug(f"Running full preprocessing for utterance: {input_utterance.id}")

    # 1. Translation
    try:
        translation_chain = chains['translation']
        translated: TranslatedUtterance = translation_chain.invoke(input_utterance)
        logger.debug(f"Translated {input_utterance.id}: '{translated.translated_text}'")
    except Exception as e:
        logger.error(f"Error during translation for utterance {input_utterance.id}: {e}", exc_info=True)
        return PreprocessingOutput(processed_utterances=[], skipped=True, skip_reason=f"Translation error: {e}")

    # 2. Filtering
    try:
        filter_chain = chains['filter']
        filtered_utterance: Optional[TranslatedUtterance] = filter_chain.invoke(translated)
        if filtered_utterance is None:
            logger.info(f"Utterance {input_utterance.id} filtered out due to length.")
            return PreprocessingOutput(processed_utterances=[], skipped=True, skip_reason="Filtered short")
        logger.debug(f"Utterance {input_utterance.id} passed filtering.")
    except Exception as e:
        logger.error(f"Error during filtering for utterance {input_utterance.id}: {e}", exc_info=True)
        # If filtering itself fails, we probably can't proceed
        # Return a failure state.
        # Note: This assumes PreprocessingOutput schema exists and can be instantiated.
        return PreprocessingOutput(processed_utterances=[], skipped=True, skip_reason=f"Filtering error: {e}")

    # 3. Segmentation
    try:
        segmentation_runnable = chains['segmentation']
        # Pass the translated text to the segmentation runnable
        segmented_texts: List[str] = segmentation_runnable.invoke({
            "translated_text": filtered_utterance.translated_text
        })

        segmented_as_units: List[SegmentedASUnit] = [
            create_segmented_as_unit(text, filtered_utterance, i)
            for i, text in enumerate(segmented_texts)
            if text # Ensure empty strings from parser are ignored
        ]

        logger.debug(f"Segmented {input_utterance.id} into {len(segmented_as_units)} AS units.")

        if not segmented_as_units:
            logger.warning(f"No AS units segmented for {input_utterance.id} (Text: '{filtered_utterance.translated_text}').")
            return PreprocessingOutput(processed_utterances=[], skipped=True, skip_reason="Segmentation resulted in zero units")

    except Exception as e:
        logger.error(f"Error during segmentation for utterance {input_utterance.id}: {e}", exc_info=True)
        return PreprocessingOutput(processed_utterances=[], skipped=True, skip_reason=f"Segmentation error: {e}")

    # --- Steps 4, 5, 6: Process each Segmented AS Unit ---
    processed_units_output: List[PreprocessedASUnit] = []
    clause_analyzer = chains['clause_analysis']
    as_unit_aligner = chains['as_unit_alignment']
    clause_aligner = chains['clause_alignment']

    for segment in segmented_as_units:
        try:
            logger.debug(f"Processing segment: {segment.as_unit_id}")
            # 4. Clause Analysis (Applied per segment)
            as_unit_with_clauses = analyze_and_format_clauses(segment, clause_analyzer)
            logger.debug(f" > Clauses analyzed for {segment.as_unit_id}")

            # 5. AS Unit Alignment (Applied per segment)
            aligned_as_unit_wc = align_as_unit_and_format(as_unit_with_clauses, as_unit_aligner)
            logger.debug(f" > AS Unit aligned for {segment.as_unit_id}")

            # 6. Clause Alignment (Applied per segment, iterates internally over clauses)
            aligned_result_list: List[PreprocessedASUnit] = perform_clause_alignment_and_final_format(
                aligned_as_unit_wc, # Input is the unit with clauses
                clause_aligner
            )

            if aligned_result_list: # Check if the list is not empty
                final_preprocessed_unit = aligned_result_list[0] # Get the single PreprocessedASUnit
                processed_units_output.append(final_preprocessed_unit) # Append the unit to our list for the utterance
                logger.debug(f" > Clauses aligned and unit finalized for {segment.as_unit_id}")
            else:
                logger.warning(f"Clause alignment and final formatting returned empty list for {segment.as_unit_id}")

        except Exception as e:
            logger.error(f"Error processing segment {segment.as_unit_id} from utterance {input_utterance.id}: {e}", exc_info=True)
            # Optionally create a 'failed' PreprocessedASUnit structure here
            # or just skip adding it to the output.
            # Skipping for now to avoid partial/failed data in the final list.
            continue # Skip to the next segment

    # After processing all segments for the utterance, return the aggregated result
    logger.info(f"Finished processing all {len(processed_units_output)} segments for utterance {input_utterance.id}")
    return PreprocessingOutput(processed_utterances=processed_units_output)
