"""Vector Store implementation for text manuals and blueprint visual schematics."""
import math
import re
from typing import List, Dict, Any

def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into words."""
    return re.findall(r"\b\w+\b", text.lower())

def compute_tf_vector(tokens: List[str], vocabulary: List[str]) -> List[float]:
    """Compute normalized term frequency vector for lightweight, deterministic embedding."""
    if not tokens:
        return [0.0] * len(vocabulary)
    counts = {t: tokens.count(t) for t in tokens}
    vec = [counts.get(w, 0.0) / len(tokens) for w in vocabulary]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two unit vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    return max(0.0, min(1.0, dot))

class DualVectorStore:
    """Dual-index vector store managing text manual documents and visual blueprint schematics."""
    def __init__(self):
        self.text_records: List[Dict[str, Any]] = []
        self.blueprint_records: List[Dict[str, Any]] = []
        self.text_vocab: List[str] = []
        self.visual_vocab: List[str] = []

    def index_text_manuals(self, manuals: List[Dict[str, Any]]):
        """Index textual manual sections."""
        self.text_records = manuals
        tokens_all = []
        for doc in manuals:
            body = f"{doc.get('section', '')} {doc.get('content', '')} {' '.join(doc.get('tags', []))}"
            tokens_all.extend(tokenize(body))
        self.text_vocab = sorted(list(set(tokens_all)))

        for doc in self.text_records:
            body = f"{doc.get('section', '')} {doc.get('content', '')} {' '.join(doc.get('tags', []))}"
            doc["vector"] = compute_tf_vector(tokenize(body), self.text_vocab)

    def index_blueprints(self, blueprints: List[Dict[str, Any]]):
        """Index blueprint and architectural diagram metadata."""
        self.blueprint_records = blueprints
        tokens_all = []
        for bp in blueprints:
            body = f"{bp.get('title', '')} {bp.get('description', '')} {' '.join(bp.get('features', []))}"
            tokens_all.extend(tokenize(body))
        self.visual_vocab = sorted(list(set(tokens_all)))

        for bp in self.blueprint_records:
            body = f"{bp.get('title', '')} {bp.get('description', '')} {' '.join(bp.get('features', []))}"
            bp["vector"] = compute_tf_vector(tokenize(body), self.visual_vocab)

    def search_text(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top text manual passages by similarity."""
        query_vec = compute_tf_vector(tokenize(query), self.text_vocab)
        scored = []
        for doc in self.text_records:
            score = cosine_similarity(query_vec, doc.get("vector", []))
            scored.append({
                "id": doc["id"],
                "section": doc["section"],
                "content": doc["content"],
                "score": score,
                "type": "text_manual"
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def search_blueprints(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top blueprint diagrams by visual/schematic similarity."""
        query_vec = compute_tf_vector(tokenize(query), self.visual_vocab)
        scored = []
        for bp in self.blueprint_records:
            score = cosine_similarity(query_vec, bp.get("vector", []))
            scored.append({
                "id": bp["id"],
                "title": bp["title"],
                "description": bp["description"],
                "diagram_type": bp.get("diagram_type", "image"),
                "score": score,
                "type": "blueprint_visual"
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
