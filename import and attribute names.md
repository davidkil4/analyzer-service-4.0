# Important Imports and Attribute Names

This document summarizes key import paths, schema attribute names, and input data keys used in the Analyzer Service project, particularly focusing on `run_preprocessing_test.py` and `analyzer_service/schemas.py`.

## 1. Imports (`run_preprocessing_test.py`)

- `json`: For loading/dumping JSON data.
- `os`: For path manipulation (`os.path.join`, `os.path.dirname`).
- `logging`: For setting up and using logging.
- `dotenv`: `load_dotenv` to load environment variables.
- `langchain_core.runnables`: `Runnable` for type hinting.
- `typing`: `List`, `Dict`, `Any`, `Optional`.
- `analyzer_service.chains`: `get_preprocessing_chain`.
- `analyzer_service.schemas`:
    - `InputUtterance`
    - `ContextUtterance`
    - `PreprocessedASUnit`
    - `AnalysisInputItem`

## 2. Schema Definitions (`analyzer_service/schemas.py`)

- **`InputUtterance`**:
    - `id`: `str` (Expected to match `id` from input JSON)
    - `speaker`: `Optional[str]`
    - `text`: `str` (Expected to match `text` from input JSON)
- **`ContextUtterance`**:
    - `speaker`: `str`
    - `text`: `str`
- **`PreprocessedASUnit`**:
    - `as_unit_id`: `str` (Generated, e.g., "u48-as1")
    - `original_utterance_id`: `str` (Should match the `id` from `InputUtterance`)
    - `original_input_text`: `str` (The *full* original text of the utterance)
    - `as_unit_text`: `str` (The text of *this specific* AS unit after segmentation/processing)
    - `aligned_original_text`: `Optional[str]` (Aligned text for the whole AS unit)
    - `clauses`: `List[AlignedClause]`
- **`AlignedClause`** (Inherits from `Clause`):
    - `clause_text`: `str` (Processed clause text)
    - `aligned_original_clause_segment`: `Optional[str]`
    - `is_korean`: `Optional[bool]`
    - `original_clause_type`: `Optional[Literal['word', 'phrase', 'collocation']]`
- **`AnalysisInputItem`** (Inherits from `PreprocessedASUnit`):
    - *All fields from `PreprocessedASUnit`*
    - `context`: `List[ContextUtterance]`

## 3. Input Data Keys (`input_files/600_son_converted.json`)

- Root level: Expected key is `"utterances"` containing a list of dictionaries.
- Within each utterance dictionary:
    - `id`: `str` (e.g., "u1", "u2") - **Crucial for context lookup**
    - `speaker`: `str` (e.g., "student", "interviewer")
    - `text`: `str` (The raw utterance text) - **Used for processing**
    - (Other fields like `start`, `end` exist but are not currently used in the script).

## 4. Key Variables/Functions (`run_preprocessing_test.py`)

- `INPUT_FILE`: Path to the input JSON.
- `OUTPUT_FILE`: Path for the output JSON.
- `BATCH_SIZE`: Integer defining batch size.
- `TARGET_SPEAKER`: String, the speaker to filter for (e.g., "student").
- `utterance_map`: `Dict[str, Dict]` mapping original utterance `id` to the utterance dictionary.
- `all_utterances_list`: `List[Dict]` holding all utterance dictionaries loaded from JSON.
- `find_context(target_utterance_id: str, utterance_map: Dict, all_utterances: List) -> Optional[List[ContextUtterance]]`: Function to retrieve preceding utterances. Looks up `target_utterance_id` using the `id` key within `all_utterances`.