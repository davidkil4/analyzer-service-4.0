# This file will contain the LangChain chains for the main analysis steps:
# 1. Correction
# 2. Accuracy Analysis
# 3. Pattern Analysis
# 4. Scoring (Complexity + Accuracy)

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from operator import itemgetter
import time
import math

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import PydanticOutputParser # Import PydanticOutputParser

# Assuming AnalysisInputItem and AlignedClause are defined in schemas.py
from .schemas import AnalysisInputItem, AlignedClause, ContextUtterance, ErrorDetail, ErrorList, PatternDetail
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# Define project root relative to this file's location
PROJECT_ROOT = Path(__file__).parent.parent
PROMPT_DIR = PROJECT_ROOT / "analyzer_service" / "prompts"

# --- LLM Initialization (Shared) ---
# Initialize the LLM once here, as it's likely shared across analysis chains
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1, convert_system_message_to_human=True)

# --- Helper Functions ---

def format_context(context: Optional[List[ContextUtterance]]) -> str:
    """Formats the context list into a string for the prompt."""
    if not context:
        return "No prior context provided."
    return "\n".join([f"{u.speaker}: {u.text}" for u in context])

def _parse_correction_output(raw_output: str) -> str:
    """Parses the raw LLM string to extract only the corrected utterance."""
    try:
        lines = [line.strip() for line in raw_output.strip().split('\n') if line.strip()]
        # Find the line starting with "Corrected utterance:" (case-insensitive)
        for i, line in enumerate(lines):
            if line.lower().startswith("corrected utterance:"):
                # Extract text after the prefix
                corrected_text = line[len("Corrected utterance:"):].strip()
                # Sometimes the LLM might add extra blank lines or parts below
                # If the next line is clearly not part of the sentence, stop here
                if i + 1 < len(lines) and lines[i+1].lower().startswith("key changes:"):
                    return corrected_text
                # Otherwise, assume it might be a multi-line correction (less likely)
                # and try to rejoin, though this might need refinement.
                # For now, just return the first line found.
                return corrected_text
        
        # Fallback: If the specific prefix isn't found, maybe the LLM just returned the text?
        # Return the first non-empty line as a guess.
        if lines:
            logger.warning(f"Could not find 'Corrected utterance:' prefix in LLM output. Using first line: {lines[0]}")
            return lines[0]
        else:
            logger.error(f"Could not parse correction output: '{raw_output}'")
            return "[PARSING ERROR]"
    except Exception as e:
        logger.error(f"Error parsing correction output: '{raw_output}'. Error: {e}")
        return "[PARSING ERROR]"

# --- Chain Definitions ---

def get_correction_chain() -> Runnable:
    """Constructs and returns the correction chain.

    This chain iterates through clauses of an AnalysisInputItem, corrects each
    clause using the provided context, and updates the 'corrected_clause_text'
    field within the original item.

    Input: AnalysisInputItem
    Output: Modified AnalysisInputItem (with corrected_clause_text populated)
    """
    logger.info("Initializing Clause-Level Correction Chain...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)

    # --- Define Prompt using System/Human Messages (Updated with new instructions) --- 
    system_instruction = ( "# INSTRUCTIONS:\n\n"
                          "## STEP 1: UNDERSTAND LEARNER INTENT\n"
                          "- Analyze the utterance in the context of the prior conversation\n"
                          "- Identify the communication goal (answering a question, making a request, etc.)\n"
                          "- Consider Korean-English specific patterns that might be present:\n"
                          "  * Subject omission (common in Korean)\n"
                          "  * Direct translation of Korean sentence structures\n"
                          "  * Topic-prominent structures instead of subject-prominent\n"
                          "  * Verb positioning at the end of clauses\n"
                          "  * Article/preposition challenges specific to Korean speakers\n\n"
                          "## STEP 2: MAKE GRAMMATICAL CORRECTIONS\n"
                          "After understanding the intent, perform the following steps:\n\n"
                          "1. Grammatical Corrections\n"
                          "- Correct **all clear grammatical errors** in:\n"
                          "  • Verb tense/aspect • Articles • Prepositions\n"
                          "  • Noun number • Pronoun reference\n"
                          "  • Quantifiers (\"minimum three\" → \"a minimum of three\")\n"
                          "  • Parallel structure (\"knowledge and skill of...\" → \"knowledge of... and skills in\")\n"
                          "  • Subject-verb agreement\n"
                          "  • Word order in questions and statements\n\n"
                          "2. Natural Expression Transformations\n"
                          "- Transform awkward but grammatically correct phrases:\n"
                          "  • \"My problem is only one\" → \"I have only one problem\"\n"
                          "  • \"This is tennis elbow by bad golf\" → \"This tennis elbow is from playing golf badly\"\n"
                          "  • \"So, now, I use the many times\" → \"So now, I use it frequently\"\n"
                          "- When the original structure is fundamentally unnatural but grammatical, reconstruct the sentence\n"
                          "- Pay special attention to sentence structures that follow Korean patterns rather than English ones\n\n"
                          "## STEP 3: APPLY FORMULAIC LANGUAGE PATTERNS\n"
                          "If the utterance remains unclear or non-native-like after grammatical corrections:\n\n"
                          "1. Identify Appropriate Formulaic Sequences\n"
                          "- Apply common formulaic sequences including:\n"
                          "  • Polywords (e.g., \"by the way\", \"of course\", \"as well as\")\n"
                          "  • Collocations (e.g., \"make a decision\", \"strong opinion\", \"take responsibility\")\n"
                          "  • Institutionalized phrases (e.g., \"if you don't mind\", \"in other words\", \"to be honest\")\n"
                          "  • Lexical bundles (e.g., \"as a result of\", \"on the other hand\", \"in terms of\")\n"
                          "  • Greetings and discourse markers: \"Well\", \"You know\", \"I mean\"\n"
                          "  • Fixed expressions: \"as a matter of fact\", \"to tell you the truth\"\n\n"
                          "2. Apply Pattern-Based Corrections\n"
                          "- Replace non-native expressions with equivalent formulaic sequences\n"
                          "- Maintain the learner's intended meaning and vocabulary level\n"
                          "- Only add sequences that are contextually appropriate\n"
                          "- Consider what a native speaker would naturally say in the same context\n\n"
                          "## CORRECTION PRIORITIES:\n"
                          "1. Errors that impede understanding\n"
                          "2. Systematic grammatical errors\n"
                          "3. Unnatural expressions that are technically correct\n"
                          "4. Minor stylistic improvements (only if other issues are addressed)\n\n"
                          "## IMPORTANT NOTE ON APPROPRIATE CORRECTION:\n"
                          "- The goal is to create natural, fluent English that preserves the learner's intended meaning\n"
                          "- Use conversation context to resolve ambiguities and determine implied meanings\n"
                          "- When the literal translation is awkward but grammatically correct, prioritize creating a response that fits naturally in the conversation flow"
                         )

    human_template = ( "# Task: Correct the learner utterance based on context and grammatical rules.\n\n"
                       "# PRIOR CONVERSATION CONTEXT:\n{prior_context}\n\n"
                       "# CLAUSE TO CORRECT:\n{utterance_to_correct}\n\n"
                       "# CORRECTED CLAUSE:" # Simplified output request
                     )

    correction_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_instruction),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

    # --- Explicit Chain Definition with Itemgetter --- 
    # Updated Output Parser needed for the new multi-part format
    # For now, we still use StrOutputParser(), but this will likely need changing
    # to a Pydantic or JSON parser once we process the structured output.
    single_clause_correction_chain = (
        {
            "prior_context": itemgetter("prior_context"),
            "utterance_to_correct": itemgetter("utterance_to_correct")
        }
        | correction_prompt # Use the new ChatPromptTemplate
        | llm 
        | StrOutputParser()
    )

    # Define the main function to be wrapped in RunnableLambda
    def correct_clauses_in_item(input_item: AnalysisInputItem) -> AnalysisInputItem:
        logger.debug(f"Starting clause correction for AS Unit: {input_item.as_unit_id}")
        if not input_item.clauses:
            logger.debug(f"No clauses found for AS Unit: {input_item.as_unit_id}, skipping correction.")
            return input_item

        # Format context once for the entire AS unit
        formatted_context = format_context(input_item.context)

        # Prepare batch input for the LLM
        batch_inputs = [
            {
                "prior_context": formatted_context,
                "utterance_to_correct": clause.clause_text
            }
            for clause in input_item.clauses
        ]

        try:
            logger.debug(f"Invoking LLM batch for {input_item.as_unit_id} with inputs: {json.dumps(batch_inputs, indent=2)}")
            # Invoke the LLM in batch for efficiency
            correction_results = single_clause_correction_chain.batch(batch_inputs, {"max_concurrency": 5})
            logger.debug(f"Raw LLM batch results for {input_item.as_unit_id}: {correction_results}")
            logger.debug(f"Received {len(correction_results)} correction results for {input_item.as_unit_id}.")

            # Update the original clause objects with the corrections
            if len(correction_results) == len(input_item.clauses):
                for i, clause in enumerate(input_item.clauses):
                    # Only update if the correction is different and not empty
                    parsed_correction = _parse_correction_output(correction_results[i])
                    if parsed_correction and parsed_correction != clause.clause_text:
                         clause.corrected_clause_text = parsed_correction
                         logger.debug(f"  Clause {i+1} corrected for {input_item.as_unit_id}")
                    else:
                        logger.debug(f"  Clause {i+1} unchanged or empty correction for {input_item.as_unit_id}")
            else:
                logger.error(f"Mismatch between clause count ({len(input_item.clauses)}) and correction results ({len(correction_results)}) for {input_item.as_unit_id}.")
                # Decide how to handle mismatch - skip update? add error flag?

        except Exception as e:
            logger.error(f"Error during batch correction for {input_item.as_unit_id}: {e}", exc_info=True)
            # Handle exception - skip update? add error flag?

        return input_item # Return the modified input item

    # Wrap the processing function in RunnableLambda
    full_chain = RunnableLambda(correct_clauses_in_item)

    logger.info("Clause-Level Correction Chain Initialized Successfully.")
    return full_chain

# --- Accuracy Analysis Chain ---

PROMPT_DIR = Path(__file__).parent / "prompts"

# Map for determining severity of using Korean based on original type
KOREAN_SEVERITY_MAP = {
    "word": "minor",
    "collocation": "moderate",
    "phrase": "critical",
    None: "moderate" # Fallback if type is missing or unexpected
}

def get_accuracy_analysis_chain() -> Runnable:
    """Initializes and returns the accuracy analysis chain.

    This chain takes an original clause and a corrected clause,
    analyzes the errors in the original based on the correction,
    and outputs a structured list of errors.
    
    Input keys: "original_clause", "corrected_clause"
    Output: ErrorList (Pydantic object)
    """
    logger.info("Initializing Accuracy Analysis Chain...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0) # Low temp for deterministic analysis

    # Load the prompt from file
    try:
        accuracy_prompt_text = (PROMPT_DIR / "accuracy_analysis_prompt.txt").read_text()
    except FileNotFoundError:
        logger.error("Accuracy analysis prompt file not found!")
        raise

    # Use JsonOutputParser which should handle the list output directly
    json_parser = JsonOutputParser(pydantic_object=List[ErrorDetail]) # Expect a list of ErrorDetail dicts

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", accuracy_prompt_text), # The whole prompt acts as system instructions
        ("human", "# Perform Analysis:\nOriginal Clause: {original_clause}\nCorrected Clause: {corrected_clause}\n\nOutput:\n```json") # Guide LLM to start JSON output
    ])

    # Define the chain
    accuracy_chain = (
        prompt_template
        | llm
        | StrOutputParser() # Get the raw string output first
        | RunnableLambda(lambda text: text.strip().strip('```json').strip('```').strip()) # Clean potential markdown
        | json_parser # Parse the cleaned string into ErrorList
    )

    logger.info("Accuracy Analysis Chain Initialized Successfully.")
    return accuracy_chain


async def process_accuracy_analysis_for_unit(analysis_input_item: AnalysisInputItem, accuracy_chain: Runnable) -> AnalysisInputItem:
    """Processes all clauses within a single AnalysisInputItem for accuracy analysis."""
    logger.debug(f"Starting accuracy analysis for AS Unit: {analysis_input_item.as_unit_id}")
    start_time = time.time()

    batch_input = []
    clauses_to_process_indices = [] # Keep track of which clauses we are processing

    for i, clause in enumerate(analysis_input_item.clauses):
        # Only analyze if we have both original and a non-empty corrected clause
        if clause.clause_text and clause.corrected_clause_text and clause.corrected_clause_text not in ["[LLM INVOCATION ERROR]", "[RESULT COUNT MISMATCH]", "[PARSING ERROR]"]:
            batch_input.append({
                "original_clause": clause.clause_text,
                "corrected_clause": clause.corrected_clause_text
            })
            clauses_to_process_indices.append(i)
        else:
             logger.debug(f"Skipping accuracy analysis for clause {i+1} in {analysis_input_item.as_unit_id} due to missing/invalid original or corrected text.")
             # Ensure errors_found is empty for skipped clauses
             clause.errors_found = [] 

    if not batch_input:
        logger.warning(f"No valid clauses found for accuracy analysis in AS Unit: {analysis_input_item.as_unit_id}")
        # Ensure all clauses have empty errors_found if none were processed
        for clause in analysis_input_item.clauses:
             clause.errors_found = []
        return analysis_input_item

    # Invoke LLM in batch
    logger.debug(f"Invoking accuracy LLM batch for {analysis_input_item.as_unit_id} with {len(batch_input)} clauses.")
    try:
        # The output should be a list of dictionaries (parsed ErrorList.errors)
        results = accuracy_chain.batch(batch_input, config={"max_concurrency": 5})
        logger.debug(f"Raw accuracy results for {analysis_input_item.as_unit_id}: {results}")
    except Exception as e:
        logger.error(f"Error invoking accuracy analysis chain for AS Unit {analysis_input_item.as_unit_id}: {e}", exc_info=True)
        # Assign error message to all processed clauses in this unit
        for idx in clauses_to_process_indices:
             analysis_input_item.clauses[idx].errors_found = [
                 ErrorDetail(category="Analysis Error", severity="critical", error="LLM invocation failed", correction=str(e))
             ]
        return analysis_input_item

    # Check result count
    if len(results) != len(clauses_to_process_indices):
        logger.error(f"Mismatch between number of processed clauses ({len(clauses_to_process_indices)}) and accuracy results ({len(results)}) for AS Unit: {analysis_input_item.as_unit_id}")
        for idx in clauses_to_process_indices:
             analysis_input_item.clauses[idx].errors_found = [
                 ErrorDetail(category="Analysis Error", severity="critical", error="Result count mismatch", correction="")
             ]
        return analysis_input_item

    # Assign errors back to the corresponding clauses
    logger.debug(f"Received {len(results)} accuracy analysis results for {analysis_input_item.as_unit_id}.")
    for i, result_list in enumerate(results):
        clause_index = clauses_to_process_indices[i]
        target_clause = analysis_input_item.clauses[clause_index] # Get the clause object
        try:
            # Result should be the list of error dicts directly from JsonOutputParser
            # Iterate directly over the result_list to parse LLM errors
            error_details = [ErrorDetail(**err) for err in result_list]

            # Check if the original was Korean and add the rule-based error
            if target_clause.is_korean:
                clause_type = target_clause.original_clause_type
                severity = KOREAN_SEVERITY_MAP.get(clause_type, "moderate") # Use map with fallback
                korean_error = ErrorDetail(
                    category="Korean Vocabulary",
                    severity=severity,
                    error=f"Original segment was Korean ({clause_type or 'unknown type'})", # Be descriptive
                    correction="Replaced with English clause" # Standard correction text
                )
                error_details.append(korean_error) # Append to the list of errors

            target_clause.errors_found = error_details # Assign the potentially augmented list
            logger.debug(f"  Assigned {len(error_details)} errors (incl. potential Korean flag) to clause {clause_index + 1} for {analysis_input_item.as_unit_id}")
        except Exception as e:
             logger.error(f"Error parsing/assigning accuracy result for clause {clause_index + 1} in {analysis_input_item.as_unit_id}: {e}. Result: {result_list}", exc_info=True)
             # Assign error marker, potentially overwriting LLM errors if parsing those failed
             target_clause.errors_found = [
                 ErrorDetail(category="Analysis Error", severity="critical", error="Result processing failed", correction=str(e))
             ]

    end_time = time.time()
    logger.debug(f"Finished accuracy analysis for {analysis_input_item.as_unit_id} in {end_time - start_time:.2f} seconds.")
    return analysis_input_item

# --- Pattern Analysis Chain ---

def get_pattern_analysis_chain() -> Runnable:
    """Initializes and returns the pattern analysis chain."""
    logger.info("Initializing Pattern Analysis Chain...")
    try:
        pattern_prompt_path = PROMPT_DIR / "pattern_analysis_prompt.txt"
        pattern_prompt_text = pattern_prompt_path.read_text(encoding='utf-8')
        logger.debug(f"Loaded pattern analysis prompt from {pattern_prompt_path}")
    except FileNotFoundError:
        logger.error("Pattern analysis prompt file not found!")
        raise

    # Expecting a JSON list of PatternDetail objects
    json_parser = JsonOutputParser(pydantic_object=List[PatternDetail])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", pattern_prompt_text), # System instructions from the file
        ("human", "Corrected Clause: {corrected_clause}") # Input variable for the clause text
    ])

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0)

    chain = prompt_template | llm | json_parser
    logger.info("Pattern Analysis Chain Initialized Successfully.")
    return chain

async def process_pattern_analysis_for_unit(analysis_input_item: AnalysisInputItem, pattern_chain: Runnable) -> AnalysisInputItem:
    """Processes pattern analysis for all clauses in a single AnalysisInputItem."""
    start_time = time.time()
    logger.info(f"Starting pattern analysis for {analysis_input_item.as_unit_id}...")

    batch_input = []
    clauses_to_process_indices = []

    # Prepare batch input for clauses that have corrected text
    for i, clause in enumerate(analysis_input_item.clauses):
        # Only analyze if we have a non-empty corrected clause
        # and it's not one of the error placeholders from the correction step
        if clause.corrected_clause_text and clause.corrected_clause_text not in ["[LLM INVOCATION ERROR]", "[RESULT COUNT MISMATCH]", "[PARSING ERROR]"]:
            batch_input.append({"corrected_clause": clause.corrected_clause_text})
            clauses_to_process_indices.append(i)
        else:
            logger.debug(f"  Skipping pattern analysis for clause {i+1} in {analysis_input_item.as_unit_id} (no valid corrected text).")
            # Ensure pattern_analysis is initialized or remains None/empty
            clause.clause_pattern_analysis = clause.clause_pattern_analysis or [] # Or None, depending on desired default

    if not batch_input:
        logger.info(f"No clauses eligible for pattern analysis in {analysis_input_item.as_unit_id}.")
        proc_time = time.time() - start_time
        logger.info(f"Finished pattern analysis for {analysis_input_item.as_unit_id} in {proc_time:.2f} seconds (no analysis performed).")
        return analysis_input_item

    # Invoke the pattern chain in batch
    try:
        logger.debug(f"Invoking pattern analysis chain for {len(batch_input)} clauses...")
        results = await pattern_chain.abatch(batch_input)
        logger.debug(f"Received {len(results)} pattern analysis results.")
    except Exception as e:
        logger.error(f"Error invoking pattern analysis chain for {analysis_input_item.as_unit_id}: {e}", exc_info=True)
        # Mark all processed clauses with an error pattern
        error_pattern = PatternDetail(
            intention="Analysis Error", category="System", component="Chain Invocation Failed",
            frequency_level=0.0, usage_context="Error during LLM call", relative_note=str(e)
        )
        for i in clauses_to_process_indices:
            analysis_input_item.clauses[i].clause_pattern_analysis = [error_pattern]
        proc_time = time.time() - start_time
        logger.info(f"Finished pattern analysis for {analysis_input_item.as_unit_id} in {proc_time:.2f} seconds (with errors).")
        return analysis_input_item

    # Assign results back to the corresponding clauses
    if len(results) != len(clauses_to_process_indices):
        logger.error(f"Mismatch between pattern result count ({len(results)}) and processed clause count ({len(clauses_to_process_indices)}) for {analysis_input_item.as_unit_id}.")
        # Handle mismatch - potentially assign error patterns
        error_pattern = PatternDetail(
            intention="Analysis Error", category="System", component="Result Count Mismatch",
            frequency_level=0.0, usage_context="Mismatch in batch processing results", relative_note=None
        )
        for i in clauses_to_process_indices:
             analysis_input_item.clauses[i].clause_pattern_analysis = [error_pattern]
    else:
        for i, result_list in enumerate(results):
            clause_index = clauses_to_process_indices[i]
            target_clause = analysis_input_item.clauses[clause_index]
            try:
                # Result should be the list of PatternDetail dicts from JsonOutputParser
                pattern_details = [PatternDetail(**pattern) for pattern in result_list]
                target_clause.clause_pattern_analysis = pattern_details
                logger.debug(f"  Assigned {len(pattern_details)} patterns to clause {clause_index + 1} for {analysis_input_item.as_unit_id}")
            except Exception as e:
                logger.error(f"Error parsing/assigning pattern result for clause {clause_index + 1} in {analysis_input_item.as_unit_id}: {e}. Result: {result_list}", exc_info=True)
                target_clause.clause_pattern_analysis = [
                    PatternDetail(
                        intention="Analysis Error", category="System", component="Result Parsing Failed",
                        frequency_level=0.0, usage_context="Error parsing LLM JSON output", relative_note=str(e)
                    )
                ]

    proc_time = time.time() - start_time
    logger.info(f"Finished pattern analysis for {analysis_input_item.as_unit_id} in {proc_time:.2f} seconds.")
    return analysis_input_item

# --- Complexity Analysis Chain (Placeholder) ---
def get_complexity_analysis_chain():
    logger.warning("Complexity analysis chain is not implemented yet.")
    pass

# --- Scoring Constants (derived from score_analysis.py and complexity_analyzer.py) ---
MAX_CLAUSE_DENSITY = 2.5
MAX_MLC = 12.0
CLAUSE_WEIGHT = 0.7
MLC_WEIGHT = 0.3
ACCURACY_LAMBDA_FACTOR = 1.2
SEVERITY_WEIGHTS = {"critical": 0.4, "moderate": 0.2, "minor": 0.1}

# --- Scoring Function ---

def calculate_scores_for_unit(item: AnalysisInputItem) -> AnalysisInputItem:
    """Calculates complexity and accuracy scores for a single AnalysisInputItem.

    This function should be called *after* all LLM-based analyses (correction,
    accuracy analysis, pattern analysis) are complete.
    """
    logger.debug(f"Calculating scores for {item.as_unit_id}...")

    # --- Complexity Calculation ---
    try:
        total_clauses = len(item.clauses)
        total_as_units = 1 # Assuming one AS Unit per AnalysisInputItem
        clause_density = total_clauses / total_as_units if total_as_units else 0

        # Calculate MLC using clause_text (the original, non-corrected text might be better for complexity?)
        # Using clause_text as per original complexity_analyzer.py logic
        total_words = 0
        valid_clauses_for_mlc = 0
        if item.clauses:
            for clause in item.clauses:
                # Need to handle potential None clause_text if pre-processing failed?
                # Assuming clause_text is always present based on upstream logic
                if clause.clause_text:
                    total_words += len(clause.clause_text.split())
                    valid_clauses_for_mlc += 1
            mlc = total_words / valid_clauses_for_mlc if valid_clauses_for_mlc > 0 else 0
        else:
            mlc = 0

        # Normalize
        normalized_clause_density = min(clause_density / MAX_CLAUSE_DENSITY, 1.0)
        normalized_mlc = min(mlc / MAX_MLC, 1.0)

        # Weighted score
        complexity_score = (CLAUSE_WEIGHT * normalized_clause_density) + (MLC_WEIGHT * normalized_mlc)
        item.complexity_score = round(complexity_score, 4) # Use more precision internally?
        logger.debug(f"  Complexity score for {item.as_unit_id}: {item.complexity_score} (Density: {clause_density:.2f}, MLC: {mlc:.2f})")

    except Exception as e:
        logger.error(f"Error calculating complexity score for {item.as_unit_id}: {e}", exc_info=True)
        item.complexity_score = None # Indicate error

    # --- Accuracy Calculation ---
    try:
        all_errors: List[ErrorDetail] = []
        for clause in item.clauses:
            if clause.errors_found:
                all_errors.extend(clause.errors_found)

        # Calculate impact
        total_impact = 0.0
        for error in all_errors:
            # Use .get() for safety, default to 'minor' weight if severity is missing/invalid
            weight = SEVERITY_WEIGHTS.get(error.severity.lower(), SEVERITY_WEIGHTS['minor'])
            total_impact += weight

        # Apply formula
        accuracy_score = math.exp(-ACCURACY_LAMBDA_FACTOR * total_impact)
        item.accuracy_score = round(accuracy_score, 4) # Use more precision internally?
        logger.debug(f"  Accuracy score for {item.as_unit_id}: {item.accuracy_score} (Total Impact: {total_impact:.2f}, Errors: {len(all_errors)})")

    except Exception as e:
        logger.error(f"Error calculating accuracy score for {item.as_unit_id}: {e}", exc_info=True)
        item.accuracy_score = None # Indicate error

    logger.debug(f"Finished calculating scores for {item.as_unit_id}.")
    return item

def get_scoring_chain() -> Runnable:
    logger.warning("Scoring chain is not implemented yet.")
    pass
