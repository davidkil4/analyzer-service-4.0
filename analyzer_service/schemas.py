from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum

# --- Raw Input Schema --- 
class InputUtterance(BaseModel):
    """Represents a single utterance from the input JSON file."""
    id: str
    speaker: str
    text: str # Original text, potentially mixed Korean/English
    timestamp: Optional[int] = None # Make timestamp optional if not always present

# --- Pre-processing Schemas --- 

class TranslatedUtterance(BaseModel):
    """Output after the translation/normalization step."""
    original_utterance: InputUtterance
    original_input_text: str # Holds the original text from InputUtterance
    translated_text: str
    # Keep track of original Korean parts if needed for alignment later?
    # original_korean_segments: Optional[List[str]] = None 

class SegmentedASUnit(BaseModel):
    """Represents a single Analysis of Speech (AS) unit segmented from an utterance."""
    # Need a way to link back to the original utterance ID
    original_utterance_id: str
    as_unit_id: str = Field(..., description="Unique ID for the AS unit (e.g., u1-as1)")
    original_input_text: str # Holds the original text from InputUtterance
    as_unit_text: str # The text of the segmented AS unit

class AlignedASUnit(SegmentedASUnit): 
    """AS Unit after aligning with original Korean text."""
    aligned_original_text: Optional[str] = None # Specific segment of original aligned

class Clause(BaseModel):
    """Represents a single clause identified within an AS unit."""
    clause_text: str = Field(..., description="The text of the clause.")

class ClauseListOutput(BaseModel):
    """Output schema for the Pydantic parser in the clause analysis step."""
    clauses: List[Clause] = Field(..., description="List of clauses found in the AS unit.")

class ASUnitWithClauses(SegmentedASUnit):
    """Represents an AS unit after clause analysis."""
    # Inherits fields from SegmentedASUnit:
    # as_unit_id: str
    # original_utterance_id: str
    # original_input_text: str
    # as_unit_text: str
    clauses: List[Clause] = Field(..., description="List of clauses derived from as_unit_text.")

class AlignedASUnitWithClauses(ASUnitWithClauses):
    """Represents an AS unit with clauses after aligning the AS unit text
       to the original input utterance text."""
    # Inherits fields from ASUnitWithClauses:
    # as_unit_id: str
    # original_utterance_id: str
    # original_input_text: str # The full original utterance text
    # as_unit_text: str        # The English text of this AS unit
    # clauses: List[Clause]    # English clauses for this AS unit
    aligned_original_text: Optional[str] = Field(
        None, description="The specific segment from original_input_text that corresponds to as_unit_text. None if alignment failed."
    )

class ClauseAlignmentOutput(BaseModel):
    """Represents the direct JSON output of the clause alignment LLM call."""
    aligned_original_clause_segment: Optional[str] # Allow None if alignment fails
    is_korean: Optional[bool] # Make Optional
    original_clause_type: Optional[Literal['word', 'phrase', 'collocation']] # Make Optional

class ErrorDetail(BaseModel):
    category: str = Field(description="The category of the identified error (e.g., 'Verb Tense/Aspect', 'Article').")
    severity: str = Field(description="The severity of the error ('critical', 'moderate', 'minor').")
    error: str = Field(description="A brief description of the specific error in the original clause.")
    correction: str = Field(description="The specific correction applied in the corrected clause.")

class ErrorList(BaseModel):
    errors: List[ErrorDetail] = Field(description="A list of all errors identified between the original and corrected clauses.")

class PatternDetail(BaseModel):
    """Represents a single identified formulaic pattern within a clause."""
    intention: str = Field(..., description="The communicative goal of the pattern (e.g., 'Expressing opinion', 'Requesting').")
    category: str = Field(..., description="The type of formula (e.g., 'Polyword', 'Frame', 'Sentence_Stem', 'Pattern').")
    component: str = Field(..., description="The formulaic sequence itself, potentially with placeholders (e.g., 'I think that', 'the ___ thing is').")
    frequency_level: float = Field(..., description="Frequency rating (1-5) in natural spoken English, possibly adjusted relatively.")
    usage_context: str = Field(..., description="Brief description of typical usage context.")
    relative_note: Optional[str] = Field(None, description="Optional note comparing frequency to other patterns in the same clause.")

class AlignedClause(Clause):
    """Represents a clause after alignment with original text and analysis."""
    # Inherits fields from Clause:
    # clause_text: str
    # ... (other potential fields from Clause if added later)

    # Fields populated by Clause Alignment step:
    aligned_original_clause_segment: Optional[str] = Field(
        None, description="The specific segment from the original AS unit text corresponding to this clause."
    )
    is_korean: Optional[bool] = Field(
        None, description="Indicates if the aligned_original_clause_segment is predominantly Korean."
    )
    original_clause_type: Optional[Literal['word', 'phrase', 'collocation']] = Field(
        None, description="Classification of the aligned_original_clause_segment ('word', 'phrase', or 'collocation')."
    )
    corrected_clause_text: Optional[str] = Field(None, description="Corrected version of the clause text.")
    errors_found: List[ErrorDetail] = Field([], description="List of errors found in the clause.")
    clause_pattern_analysis: Optional[List[PatternDetail]] = Field([], description="List of patterns found in the clause.") # Populated by pattern analysis

class PreprocessedASUnit(BaseModel): 
    """Final structure for a single AS unit after all pre-processing.
       Should contain AlignedClauses."""
    # --- From SegmentedASUnit --- 
    as_unit_id: str
    original_utterance_id: str # ID of the original Utterance
    original_input_text: str # The full original input text of the utterance
    as_unit_text: str # The text of this specific AS unit
    # --- From AlignedASUnitWithClauses ---
    aligned_original_text: Optional[str] # The aligned segment for the *whole* AS unit
    # --- Updated with AlignedClauses --- 
    clauses: List[AlignedClause] # Now contains fully aligned clauses

class PreprocessingOutput(BaseModel):
    """Represents the final output of the entire pre-processing pipeline."""
    processed_utterances: List[PreprocessedASUnit]

# --- Main Analysis Schemas (Placeholders - Define based on analysis tasks) --- 

class AnalysisInput(BaseModel):
    """Input to the main analysis chain (likely a list of preprocessed units)."""
    batch_preprocessed_data: List[PreprocessingOutput]

class AnalysisResult(BaseModel):
    """Represents the final output of one analysis step/component."""
    # Example fields - replace with actual analysis outputs
    component_name: str
    results: Dict[str, Any]
    recommendations: Optional[List[str]] = None

class FinalOutput(BaseModel):
    """Overall final output structure for a batch."""
    batch_id: int
    analysis_results: List[AnalysisResult]
    # Include original or preprocessed data for reference?
    # source_data: Optional[AnalysisInput] = None

# ==========================
# Main Analysis Output Schemas
# ==========================

class Severity(str, Enum):
    CRITICAL = "critical"
    MODERATE = "moderate"
    MINOR = "minor"

class ClauseAnalysis(BaseModel):
    clause_text: str
    corrected_clause_text: Optional[str] = None
    errors_found: List[ErrorDetail]
    clause_pattern_analysis: Optional[Dict[str, Any]] = None # Structure TBD

class MainAnalysisOutput(BaseModel):
    as_unit_id: str
    original_text: str # This should be the as_unit_text from PreprocessedASUnit
    corrected_text: Optional[str] = None
    complexity_score: float
    accuracy_score: float
    clauses: List[ClauseAnalysis]
    as_unit_pattern_analysis: Optional[Dict[str, Any]] = None # Structure TBD

# ====================================
# Intermediate Analysis Chain Schemas
# ====================================

# Removing the CorrectionOutput schema as correction happens per clause
# class CorrectionOutput(BaseModel):
#     """Intermediate output from the correction chain."""
#     as_unit_id: str
#     original_as_unit_text: str # Pass through the original text for downstream use
#     corrected_text: Optional[str] = None # Allow None if correction fails or is identical
#     # Include context used for correction for potential debugging/logging? Optional.
#     # context_used: Optional[List[ContextUtterance]] = None

# ================================
# Schemas for Adding Context
# ================================

class ContextUtterance(BaseModel):
    speaker: str
    text: str

class AnalysisInputItem(PreprocessedASUnit):
    """Input schema for the main analysis chain, including prior utterance context."""
    context: Optional[List[ContextUtterance]] = None
    complexity_score: Optional[float] = Field(None, description="Calculated syntactic complexity score (structure score based on density and MLC).")
    accuracy_score: Optional[float] = Field(None, description="Calculated accuracy score based on severity and count of errors.")

# =============================
# Placeholder for Analysis Results (If needed later)
# =============================
# class AnalysisResult(BaseModel):
#     # Combine MainAnalysisOutput with teaching content?
#     analysis_output: MainAnalysisOutput
#     teaching_content: Optional[str] = None # Placeholder
#
# class FullPipelineOutput(BaseModel):
#     # Represents the final output after all chains (Preproc -> Analysis -> Teaching)
#     final_results: List[AnalysisResult]
