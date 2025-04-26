import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple, Generator
from operator import itemgetter

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.output_parsers.list import NumberedListOutputParser, CommaSeparatedListOutputParser
from langchain_core.exceptions import OutputParserException

from .schemas import (
    InputUtterance,
    PreprocessingOutput,
    PreprocessedASUnit,
    AlignedClause,
    AnalyzedClause,
    AlignedASUnit,
    AlignedASUnitWithClauses,
    ClauseAnalysisOutput,
    ClauseAlignmentOutput,
    ErrorSeverity, # Added for Main Chain
    ErrorDetail,   # Added for Main Chain
    ClauseAnalysis, # Added for Main Chain
    AlignmentOnlyOutput # Added import
)
from .utils import load_prompt # Corrected import

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions --- 

def filter_short_translated_utterance(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Filters out utterances that are too short after translation."""
    # Placeholder - Logic to be added
    translated_text = data.get("translated_text")
    utterance_id = data.get("utterance_id", "UNKNOWN_ID")
    if translated_text and len(translated_text.split()) >= 3:
        logger.debug(f"Utterance passed filter (ID: {utterance_id}, Words: {len(translated_text.split())}): '{translated_text[:50]}...'" )
        return data
    else:
        logger.info(f"Filtering out short utterance (ID: {utterance_id}, Words: {len(translated_text.split()) if translated_text else 0}): '{translated_text}'")
        return None # Indicate filtering

def perform_clause_alignment_and_final_format(
    as_unit_id: str,
    original_utterance_id: str,
    original_input_text: str,
    as_unit_text: str,
    aligned_original_as_unit_text: Optional[str],
    analyzed_clauses: List[AnalyzedClause], # Expect List[AnalyzedClause]
    clause_aligner: Runnable,
    clause_alignment_format_instructions: str # Added format instructions
) -> List[PreprocessedASUnit]:
    """
    Performs clause alignment for each AnalyzedClause and formats the final PreprocessedASUnit.
    """
    final_aligned_clauses: List[AlignedClause] = []

    # Determine the reference text for clause alignment
    # Prefer the specifically aligned AS unit segment if available, otherwise use the original input text
    # This assumes clauses are found within the span of the AS unit's aligned segment if it exists.
    alignment_reference_text = aligned_original_as_unit_text if aligned_original_as_unit_text is not None else original_input_text

    for clause in analyzed_clauses:
        try:
            logger.debug(f"Aligning clause: '{clause.clause_text}' for AS Unit: {as_unit_id}")
            # Prepare input for clause aligner, matching the prompt variables
            input_dict = {
                "aligned_original_text": alignment_reference_text,
                "target_clause_text": clause.clause_text,
                # Add format instructions expected by the prompt
                "input_format": clause_alignment_format_instructions,
                "output_format": clause_alignment_format_instructions
            }
            alignment_result: Optional[ClauseAlignmentOutput] = None
            try:
                # Clause aligner now uses PydanticOutputParser directly
                alignment_result = clause_aligner.invoke(input_dict)
                logger.debug(f"Clause Aligner Output for '{clause.clause_text}': {alignment_result}")
            except OutputParserException as pe:
                logger.warning(f"Clause aligner Pydantic parsing failed for '{clause.clause_text}' in {as_unit_id}: {pe}. Raw output: {getattr(pe, 'llm_output', 'N/A')}")
            except Exception as invoke_err:
                logger.error(f"Error invoking clause_aligner for '{clause.clause_text}' in {as_unit_id}: {invoke_err}", exc_info=True)

            # Construct AlignedClause
            aligned_clause = AlignedClause(
                clause_text=clause.clause_text, # Use text from the AnalyzedClause
                aligned_original_clause_segment=getattr(alignment_result, 'aligned_original_clause_segment', None) if alignment_result else None,
                is_korean=getattr(alignment_result, 'is_korean', None) if alignment_result else None,
                original_clause_type=getattr(alignment_result, 'original_clause_type', None) if alignment_result else None
            )
            final_aligned_clauses.append(aligned_clause)

        except Exception as e:
            logger.error(f"General error during clause alignment for clause '{clause.clause_text}' in AS Unit {as_unit_id}: {e}", exc_info=True)
            # Append clause with no alignment info on error
            final_aligned_clauses.append(AlignedClause(clause_text=clause.clause_text))

    # Construct the final PreprocessedASUnit object
    preprocessed_as_unit = PreprocessedASUnit(
        as_unit_id=as_unit_id,
        original_utterance_id=original_utterance_id,
        original_input_text=original_input_text,
        as_unit_text=as_unit_text,
        aligned_original_text=aligned_original_as_unit_text,
        clauses=final_aligned_clauses # Use the list of AlignedClause
    )
    return [preprocessed_as_unit]

def segment_and_analyze_utterance(
    data: Dict[str, Any],
    segmentation_runnable: Runnable,
    clause_analyzer: Runnable,
    as_unit_aligner: Runnable,
    clause_aligner: Runnable,
    clause_alignment_format_instructions: str # Added format instructions
) -> List[PreprocessedASUnit]:
    """
    Processes a single filtered utterance through segmentation, clause analysis, and alignment.
    """
    utterance: InputUtterance = data['utterance']
    original_text: str = data['original_text']
    utterance_id = utterance.id # Use 'id' from InputUtterance
    logger.debug(f"Starting segmentation and analysis for utterance: {utterance_id}")

    # --- 3. Segmentation ---
    try:
        logger.debug(f"Segmenting text for {utterance_id}: '{original_text[:100]}...'" ) # Log snippet
        # Pass the translated text (original_text) to the segmentation runnable
        segmented_texts: List[str] = segmentation_runnable.invoke({
            # Key expected by segmentation_runnable (based on get_preprocessing_chain)
            "input_utterances": original_text
        })

        if not segmented_texts or all(not s.strip() for s in segmented_texts):
            logger.warning(f"Segmentation returned no valid segments for utterance {utterance_id}. Text: '{original_text}'")
            return []

        logger.debug(f"Segmented {utterance_id} into {len(segmented_texts)} potential AS units.")

    except Exception as e:
        logger.error(f"Error during segmentation for utterance {utterance_id}: {e}", exc_info=True)
        return []

    all_processed_as_units_for_utterance: List[PreprocessedASUnit] = []

    # Process each segmented text chunk
    for i, as_unit_segment_text in enumerate(segmented_texts):
        if not as_unit_segment_text.strip():
            logger.debug(f"Skipping empty segment {i+1} for utterance {utterance_id}")
            continue

        as_unit_id = f"{utterance_id}-as{i+1}"
        logger.debug(f"--- Processing segment {as_unit_id} ---")

        try:
            # --- 4. Clause Analysis (Applied per segment) ---
            logger.debug(f"Analyzing clauses for AS Unit: {as_unit_id}")
            # Invoke clause analyzer, expecting ClauseAnalysisOutput
            clause_analysis_result: ClauseAnalysisOutput = clause_analyzer.invoke({
                # Key expected by clause_analyzer (based on get_preprocessing_chain)
                "input_as_units": as_unit_segment_text
            })
            analyzed_clauses: List[AnalyzedClause] = clause_analysis_result.clauses
            logger.debug(f" > Analyzed {len(analyzed_clauses)} clauses for {as_unit_id}")

            # --- 5. Align AS Unit with Original Text ---
            logger.debug(f"Aligning AS Unit: {as_unit_id}")
            alignment_input = {
                "original_input_text": original_text,
                "target_as_unit_text": as_unit_segment_text
            }
            # Invoke aligner, expecting AlignmentOnlyOutput
            alignment_result: AlignmentOnlyOutput = as_unit_aligner.invoke(alignment_input)
            aligned_original_as_unit_text = alignment_result.aligned_original_text
            logger.debug(f" > Aligned AS Unit {as_unit_id} to original: '{aligned_original_as_unit_text}'")

            # --- 6. Clause Alignment & Final Formatting (Applied per AS unit) ---
            logger.debug(f"Preparing for clause alignment & final formatting for AS Unit: {as_unit_id}")
            # Pass all necessary components to the updated function
            final_aligned_units: List[PreprocessedASUnit] = perform_clause_alignment_and_final_format(
                as_unit_id=as_unit_id,
                original_utterance_id=utterance_id,
                original_input_text=original_text,
                as_unit_text=as_unit_segment_text,
                aligned_original_as_unit_text=aligned_original_as_unit_text,
                analyzed_clauses=analyzed_clauses,
                clause_aligner=clause_aligner,
                clause_alignment_format_instructions=clause_alignment_format_instructions # Pass instructions
            )

            all_processed_as_units_for_utterance.extend(final_aligned_units)
            logger.info(f"Successfully processed segment {as_unit_id}")

        except OutputParserException as ope:
             logger.error(f"Output parsing error processing segment {as_unit_id} for utterance {utterance_id}: {ope}", exc_info=False) # Log less verbosely
             logger.error(f" >> Parser Error Detail: {ope.args[0]}") # Log the specific error message
             logger.error(f" >> LLM Output causing error: {getattr(ope, 'llm_output', 'N/A')}") # Log raw output if available
        except Exception as e:
            logger.error(f"General error processing segment {as_unit_id} for utterance {utterance_id}: {e}", exc_info=True)
            # Continue to next segment if one fails
            continue

    logger.info(f"Finished processing segments for utterance {utterance_id}. Produced {len(all_processed_as_units_for_utterance)} PreprocessedASUnit(s).")
    return all_processed_as_units_for_utterance

def preprocess_single_utterance(
    utterance: InputUtterance, 
    translation_chain: Runnable, 
    filter_runnable: Runnable, 
    segmentation_runnable: Runnable, 
    clause_analyzer: Runnable, 
    as_unit_aligner: Runnable, 
    clause_aligner: Runnable,
    clause_alignment_format_instructions: str # Added format instructions
) -> Optional[PreprocessingOutput]:
    """Full preprocessing logic for a single utterance, including translation, filtering, and segmentation/analysis."""
    logger.debug(f"Running full preprocessing for utterance: {utterance.id}")
    try:
        # Prepare initial data for translation and potential filtering
        initial_data = {
            "utterance_id": utterance.id,
            "input_text": utterance.text, # Used by translation prompt
            "original_text": utterance.text # Keep original text for later steps
        }

        # 1. Translation
        translated_text = translation_chain.invoke({"input_text": utterance.text})
        logger.debug(f"Translated {utterance.id}: '{translated_text[:50]}...'") 
        
        # Add translated text to the data dict for filtering
        data_for_filtering = {**initial_data, "translated_text": translated_text}

        # 2. Filtering
        filtered_data = filter_runnable.invoke(data_for_filtering)

        # 3. Check if filtered out
        if filtered_data is None:
            logger.info(f"Utterance {utterance.id} filtered out due to length.")
            return None # Utterance was filtered, return None
        
        logger.debug(f"Utterance {utterance.id} passed filtering.")
        # filtered_data now contains: utterance_id, input_text, original_text, translated_text

        # 4. Segmentation, Analysis, and Alignment (using the next helper)
        # Pass all necessary runnables to the next stage
        # Construct the data dict expected by segment_and_analyze_utterance
        data_for_segmentation = {
            **filtered_data, # Include keys like original_text, utterance_id
            'utterance': utterance # Add the full InputUtterance object
        }
        preprocessing_output = segment_and_analyze_utterance(
            data_for_segmentation, # Pass the correctly structured dict
            segmentation_runnable,
            clause_analyzer,
            as_unit_aligner,
            clause_aligner,
            clause_alignment_format_instructions=clause_alignment_format_instructions # Pass instructions
        )

        # Wrap the list of PreprocessedASUnit objects into the final PreprocessingOutput
        if preprocessing_output is not None:
             return PreprocessingOutput(processed_utterances=preprocessing_output)
        else:
             # If segment_and_analyze_utterance returns None (e.g., segmentation failed)
             return PreprocessingOutput(processed_utterances=[]) # Return empty list as per schema

    except Exception as e:
        logger.error(f"Error during full preprocessing for utterance {utterance.id}: {e}", exc_info=True)
        return None # Return None on any top-level error for this utterance

# --- Preprocessing Chain --- 

def get_preprocessing_chain() -> Runnable:
    """Constructs and returns the complete preprocessing LangChain."""
    logger.info("Initializing LLM within get_preprocessing_chain...")
    # Configure the LLM (Gemini Flash in this case)
    # Note: Adjust model_name if using a different Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0, convert_system_message_to_human=True) # Corrected model name

    # --- 1. Translation Sub-Chain ---
    logger.info("Creating Translation chain...")
    translation_prompt_template = load_prompt("./prompts/translation_prompt.txt")
    translation_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=translation_prompt_template),
        HumanMessage(content="{input_text}")
    ])
    translation_chain = (
        translation_prompt 
        | llm 
        | StrOutputParser()
    )
    logger.info("Translation chain created.")

    # --- 2. Filtering Step (using RunnableLambda) ---
    logger.info("Creating Filter chain...")
    # Filter runs after translation. Input is expected to be the dict from the initial passthrough
    # containing 'utterance_id' and 'translated_text'.
    filter_runnable = RunnableLambda(filter_short_translated_utterance)
    logger.info("Filter chain created.")

    # --- 3. Segmentation Sub-Chain ---
    logger.info("Creating Segmentation runnable...")
    segmentation_prompt_template = load_prompt("./prompts/segmentation_prompt.txt")
    segmentation_prompt = ChatPromptTemplate.from_template(segmentation_prompt_template)

    def split_into_list(text: str) -> List[str]:
        """Splits the text by newlines and cleans up entries."""
        if not text:
            return []
        # Split by one or more newline characters
        items = re.split(r'\n+', text.strip())
        # Remove empty strings that might result from multiple newlines or trailing newlines
        cleaned_items = [item.strip() for item in items if item.strip()]
        return cleaned_items
        
    # Expects 'translated_text' as input
    segmentation_runnable = (
        segmentation_prompt
        | llm
        | StrOutputParser() # Get the raw string output
        | RunnableLambda(split_into_list) # Split the string into a list
    )
    logger.info("Segmentation runnable created.")

    # --- 4. Clause Analysis Sub-Chain ---
    logger.info("Creating Clause Analyzer...")
    clause_analysis_prompt_template = load_prompt("./prompts/clause_analysis_prompt.txt")
    # Instantiate the parser first to get format instructions
    clause_parser = PydanticOutputParser(pydantic_object=ClauseAnalysisOutput)
    format_instructions = clause_parser.get_format_instructions()
    
    # Create the prompt template, explicitly defining variables
    clause_analysis_prompt = PromptTemplate(
        template=clause_analysis_prompt_template,
        input_variables=["input_as_units"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    # Define the chain
    clause_analyzer = (
        clause_analysis_prompt
        | llm
        | clause_parser # Use the instantiated parser
    )
    logger.info("Clause analyzer created.")

    # --- 5. AS Unit Alignment Sub-Chain ---
    logger.info("Creating AS Unit Aligner...")
    as_unit_alignment_prompt_template = load_prompt("./prompts/as_unit_alignment_prompt.txt")
    # Instantiate parser first for the specific AlignmentOnlyOutput schema
    alignment_output_parser = PydanticOutputParser(pydantic_object=AlignmentOnlyOutput)
    alignment_format_instructions = alignment_output_parser.get_format_instructions()
    
    # Create prompt template using the new format instructions
    as_unit_alignment_prompt = PromptTemplate(
        template=as_unit_alignment_prompt_template,
        input_variables=["original_input_text", "target_as_unit_text"],
        partial_variables={"format_instructions": alignment_format_instructions}
    )
    
    # Define the chain
    as_unit_aligner = (
        as_unit_alignment_prompt
        | llm
        | alignment_output_parser # Use the new parser
    )
    logger.info("AS unit aligner created.")

    # --- 6. Clause Alignment Sub-Chain ---
    logger.info("Creating Clause Aligner...")
    clause_alignment_prompt_template = load_prompt("./prompts/clause_alignment_prompt.txt")
    clause_alignment_parser = PydanticOutputParser(pydantic_object=ClauseAlignmentOutput)
    # Get format instructions to be passed down
    clause_alignment_format_instructions = clause_alignment_parser.get_format_instructions()

    # Note: Unlike AS Unit aligner, we assume the prompt file itself has {input_format} and {output_format}
    # and doesn't use partial_variables here. The instructions are passed at invocation time.
    clause_alignment_prompt = PromptTemplate(
        template=clause_alignment_prompt_template,
        input_variables=[
            "aligned_original_text",
            "target_clause_text",
            "input_format",
            "output_format"
        ]
        # No partial_variables for format_instructions here
    )

    clause_aligner = (
        clause_alignment_prompt
        | llm
        | clause_alignment_parser
    )
    logger.info("Clause aligner created.")

    # --- 7. Combine Chains into the Full Preprocessing Pipeline --- 
    # This will use RunnableLambda to wrap the main orchestration logic
    # Placeholder for the final chain structure

    # Combine the steps using helper functions
    # This will involve calling preprocess_single_utterance within a RunnableLambda
    full_preprocessing_pipeline = RunnableLambda(lambda utterance: preprocess_single_utterance(
        utterance,
        translation_chain,
        filter_runnable,
        segmentation_runnable,
        clause_analyzer,
        as_unit_aligner,
        clause_aligner,
        clause_alignment_format_instructions=clause_alignment_format_instructions # Pass instructions to bind
    ))

    logger.info("Full preprocessing pipeline (RunnableLambda wrapping placeholder) created.")
    # TODO: Define the actual chain structure combining all steps correctly
    # For now, returning a placeholder to allow testing script to load it.
    # return full_preprocessing_pipeline
    return full_preprocessing_pipeline # Return the actual full preprocessing pipeline

# --- Main Analysis Chain (Placeholder) ---

def get_main_analysis_chain() -> Runnable:
    """Constructs and returns the main analysis LangChain (Placeholder)."""
    # Placeholder: Define analysis steps (correction, accuracy, patterns, scoring)
    return RunnableLambda(lambda x: {"analysis_output": "not implemented"})