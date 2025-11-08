"""
Data Formatter Module
=====================
Handles data format conversion for evaluation
"""

from typing import Dict, Any


class DataFormatter:
    """Handles data format conversion"""
    
    @staticmethod
    def hover_to_gepa_format(example: Dict) -> Dict:
        """
        Convert a HoVer example to GEPA format
        
        Args:
            example: HoVer dataset example with 'claim', 'evidence', 'label'
        
        Returns:
            Dict in GEPA format with 'input', 'answer', 'additional_context'
        """
        evidence = example['evidence']
        
        # Handle evidence format
        if isinstance(evidence, str):
            sentences = [s.strip() for s in evidence.split('.') if s.strip()]
        elif isinstance(evidence, list):
            sentences = [str(s).strip() for s in evidence if str(s).strip()]
        else:
            sentences = [str(evidence)]
        
        formatted_context = '\n'.join(f"{i+1}. {s}" for i, s in enumerate(sentences))
        
        input_text = f"""Claim: {example['claim']}

Context:
{formatted_context}"""
        
        label_raw: Any = example.get('label')

        def map_hover_label(label: Any) -> str:
            """
            Map HoVer label to canonical form.

            Robust to ints, strings, and common synonyms. Empirically, many HoVer variants use:
            - 0 => SUPPORTED, 1 => NOT_SUPPORTED (REFUTES)
            If the label is a string, normalize and map accordingly.
            """
            # Handle numeric-like strings
            try:
                if isinstance(label, str) and label.strip().isdigit():
                    label = int(label.strip())
            except Exception:
                pass

            # Integer/float mapping (assume 0 = SUPPORTED, 1 = NOT_SUPPORTED)
            if isinstance(label, (int, float)):
                return "SUPPORTED" if int(label) == 0 else "NOT_SUPPORTED"

            # String mapping
            if isinstance(label, str):
                norm = label.strip().upper().replace(" ", "_")
                if norm in {"SUPPORTED", "SUPPORTS", "ENTAILMENT", "TRUE"}:
                    return "SUPPORTED"
                if norm in {"NOT_SUPPORTED", "REFUTES", "REFUTED", "FALSE", "NOTSUPPORT"}:
                    return "NOT_SUPPORTED"
                # Fallback: if unknown string, default to NOT_SUPPORTED to be conservative
                return "NOT_SUPPORTED"

            # Unknown type fallback
            return "NOT_SUPPORTED"

        answer = map_hover_label(label_raw)
        
        return {
            "input": input_text,
            "answer": answer,
            "additional_context": {
                "id": str(example.get('id', '')),
            }
        }
