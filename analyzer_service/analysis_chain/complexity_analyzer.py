from typing import Dict, Any
import math

class ComplexityAnalyzer:
    """
    Calculates syntactic complexity using:
    - Clause density (clauses per AS-unit)
    - Mean length of clauses (words per clause)
    """
    #: Maximum clause density based on Foster & Skehan's upper bound.
    MAX_CLAUSE_DENSITY = 2.5  
    #: Empirical maximum mean length of clauses from corpus analysis.
    MAX_MLC = 12.0  
    #: Weight assigned to clause density in structure score calculation.
    CLAUSE_WEIGHT = 0.7
    #: Weight assigned to mean length of clauses in structure score calculation.
    MLC_WEIGHT = 0.3

    @staticmethod
    def analyze(utterance_data: Dict) -> Dict[str, Any]:
        """Returns complexity score with simplified structure"""
        syntactic = ComplexityAnalyzer._analyze_syntactic_complexity(utterance_data)
        return {
            "score": syntactic['structure_score'],
            "clause_density": syntactic['clause_density'],
            "mean_clause_length": syntactic['mlc']
        }

    @staticmethod
    def _analyze_syntactic_complexity(utterance_data: Dict) -> Dict[str, Any]:
        """Analyzes syntactic complexity using clause density and mean clause length"""
        
        # Calculate clause density
        total_clauses = len(utterance_data.get("clauses", []))
        total_as_units = 1  # 1 AS-unit per utterance
        clause_density = total_clauses / total_as_units if total_as_units else 0
        
        # Calculate MLC (Mean Length of Clause)
        if utterance_data.get("clauses"):
            mlc = sum(len(clause["text"].split()) for clause in utterance_data["clauses"]) / len(utterance_data["clauses"])
        else:
            mlc = 0
        
        # Normalize scores
        normalized_clause_density = min(clause_density / ComplexityAnalyzer.MAX_CLAUSE_DENSITY, 1.0)  
        normalized_mlc = min(mlc / ComplexityAnalyzer.MAX_MLC, 1.0)
        
        # Calculate structure score
        structure_score = (
            (0.7 * normalized_clause_density) + 
            (0.3 * normalized_mlc)
        )
        
        return {
            "clause_density": round(clause_density, 2),
            "mlc": round(mlc, 2),
            "structure_score": round(structure_score, 2)
        }
