"""Late-fusion reranker for multimodal RAG."""
from typing import List, Dict, Any

class LateFusionReranker:
    """
    Combines text manual retrieval candidates and visual blueprint diagram candidates
    using weighted late-fusion scoring.
    """
    def __init__(self, alpha: float = 0.55):
        """
        Args:
            alpha: Weight assigned to text similarity (0.0 to 1.0).
                   Visual similarity weight is (1.0 - alpha).
        """
        self.alpha = alpha

    def rerank(
        self,
        text_results: List[Dict[str, Any]],
        blueprint_results: List[Dict[str, Any]],
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Merges results and assigns fused scores.
        """
        fused: List[Dict[str, Any]] = []

        for item in text_results:
            weighted_score = self.alpha * item["score"]
            fused.append({
                **item,
                "raw_score": item["score"],
                "fused_score": round(weighted_score, 4),
                "fusion_weight": self.alpha
            })

        for item in blueprint_results:
            weighted_score = (1.0 - self.alpha) * item["score"]
            fused.append({
                **item,
                "raw_score": item["score"],
                "fused_score": round(weighted_score, 4),
                "fusion_weight": round(1.0 - self.alpha, 4)
            })

        # Sort combined results descending by fused score
        fused.sort(key=lambda x: x["fused_score"], reverse=True)
        return fused[:top_k]
