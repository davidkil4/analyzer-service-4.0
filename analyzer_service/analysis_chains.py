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

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Import necessary schemas
from .schemas import AnalysisInputItem, MainAnalysisOutput, ContextUtterance, AlignedClause
from .preprocessing_chains import load_prompt_from_file # Import helper

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

# Placeholders for subsequent chains
def get_accuracy_analysis_chain() -> Runnable:
    logger.warning("Accuracy analysis chain is not implemented yet.")
    pass

def get_pattern_analysis_chain() -> Runnable:
    logger.warning("Pattern analysis chain is not implemented yet.")
    pass

def get_scoring_chain() -> Runnable:
    logger.warning("Scoring chain is not implemented yet.")
    pass
