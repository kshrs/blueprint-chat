"""End-to-end Multimodal RAG Pipeline with Late-Fusion Reranking."""
import json
from pathlib import Path
from typing import Dict, Any, List
from .vector_store import DualVectorStore
from .reranker import LateFusionReranker

class MultimodalRAGPipeline:
    def __init__(self, data_dir: str, alpha: float = 0.55):
        self.store = DualVectorStore()
        self.reranker = LateFusionReranker(alpha=alpha)
        self.data_dir = Path(data_dir)
        self._load_and_index()

    def _load_and_index(self):
        manuals_path = self.data_dir / "manuals.json"
        blueprints_path = self.data_dir / "blueprints.json"

        with open(manuals_path, "r", encoding="utf-8") as f:
            manuals = json.load(f)
        with open(blueprints_path, "r", encoding="utf-8") as f:
            blueprints = json.load(f)

        self.store.index_text_manuals(manuals)
        self.store.index_blueprints(blueprints)

    def query(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Runs dual retrieval, applies late-fusion reranking, and synthesizes a grounded answer.
        """
        text_matches = self.store.search_text(query_text, top_k=top_k)
        bp_matches = self.store.search_blueprints(query_text, top_k=top_k)

        reranked = self.reranker.rerank(text_matches, bp_matches, top_k=top_k * 2)

        # Context synthesis
        text_contexts = [r for r in reranked if r["type"] == "text_manual"]
        bp_contexts = [r for r in reranked if r["type"] == "blueprint_visual"]

        answer = self._generate_answer(query_text, text_contexts, bp_contexts)

        return {
            "query": query_text,
            "answer": answer,
            "reranked_context": reranked,
            "text_candidates_retrieved": len(text_matches),
            "blueprint_candidates_retrieved": len(bp_matches),
        }

    def _generate_answer(
        self,
        query: str,
        texts: List[Dict[str, Any]],
        blueprints: List[Dict[str, Any]]
    ) -> str:
        """Deterministic response generation grounded strictly in retrieved multimodal evidence."""
        lines = []
        if texts:
            top_t = texts[0]
            lines.append(f"According to technical manual [{top_t['section']}]: \"{top_t['content']}\"")
        if blueprints:
            top_bp = blueprints[0]
            lines.append(f"Referencing visual schematic [{top_bp['title']} ({top_bp['diagram_type']})]: \"{top_bp['description']}\"")

        if not lines:
            return "No matching guidelines or blueprint schematics found for the specified query."

        return " ".join(lines)
