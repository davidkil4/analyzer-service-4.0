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

# New schema specifically for the output of the AS Unit Aligner LLM call
class AlignmentOnlyOutput(BaseModel):
    """Schema for the direct output of the AS Unit alignment step."""
    aligned_original_text: Optional[str] = Field(None, description="The aligned segment from the original text, or null if no alignment found.")

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

class AnalyzedClause(BaseModel):
    """Represents a single clause identified during initial analysis."""
    clause_text: str = Field(..., description="The text of the clause.")

class ClauseAnalysisOutput(BaseModel):
    """Output schema for the clause analysis LLM call."""
    clauses: List[AnalyzedClause] = Field(..., description="A list of clauses extracted from the AS unit.")

class ClauseAlignmentOutput(BaseModel):
    """Represents the direct JSON output of the clause alignment LLM call."""
    aligned_original_clause_segment: Optional[str] # Allow None if alignment fails
    is_korean: Optional[bool] # Make Optional
    original_clause_type: Optional[Literal['word', 'phrase', 'collocation']] # Make Optional

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

# --- Main Analysis Schemas ---

class ErrorSeverity(str, Enum):
    """Enumeration for error severity levels."""
    CRITICAL = "critical"
    MODERATE = "moderate"
    MINOR = "minor"

class ErrorDetail(BaseModel):
    """Schema for a single error identified within a clause."""
    category: str = Field(..., description="The category of the error (e.g., 'Sentence Structure', 'Vocabulary').")
    severity: ErrorSeverity = Field(..., description="The severity of the error.")
    error: str = Field(..., description="A description of the specific error found.")
    correction: Optional[str] = Field(None, description="The suggested correction for the error.")

class ClauseAnalysis(BaseModel):
    """Schema for the analysis results of a single clause."""
    clause_text: str = Field(..., description="The original text of the clause.")
    corrected_clause_text: Optional[str] = Field(None, description="The corrected version of the clause text, if applicable.")
    errors_found: List[ErrorDetail] = Field(default_factory=list, description="A list of errors found within this clause.")
    clause_pattern_analysis: Optional[Any] = Field(None, description="Pattern analysis results specifically for this clause (Structure TBD).") # Using Any until structure is defined

class MainAnalysisOutput(BaseModel):
    """Schema for the final output of the main analysis chain for a single AS Unit."""
    as_unit_id: str = Field(..., description="Unique identifier for the AS unit.")
    original_text: str = Field(..., description="The original text of the AS unit (post-preprocessing).")
    corrected_text: Optional[str] = Field(None, description="The corrected version of the AS unit text, if applicable.")
    complexity_score: Optional[float] = Field(None, description="The calculated complexity score for the AS unit.")
    accuracy_score: Optional[float] = Field(None, description="The calculated accuracy score for the AS unit.")
    clauses: List[ClauseAnalysis] = Field(default_factory=list, description="A list containing the analysis results for each clause in the AS unit.")
    as_unit_pattern_analysis: Optional[Any] = Field(None, description="Pattern analysis results for the entire AS unit (Structure TBD).") # Using Any until structure is defined

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
