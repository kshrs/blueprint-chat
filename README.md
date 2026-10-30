# Blueprint-Chat: Multimodal RAG for Technical Manuals

A clean, modular Multimodal Retrieval-Augmented Generation (RAG) system for technical building manuals and floor plan diagrams, implementing **late-fusion reranking**.

Built as part of practical coursework exploration for the *IBM RAG and Agentic AI Professional Certificate*.

---

## 🌟 Key Features

1. **Multimodal Document Ingestion**:
   - Parses both text specification paragraphs (safety guidelines, room dimensions, building codes) and diagram/blueprint descriptions or image embeddings.
2. **Dual-Index Vector Store**:
   - Maintains separate Chroma / vector indexes for text chunks and visual blueprint schematics.
3. **Late-Fusion Reranking**:
   - Computes text similarity and visual/diagram similarity independently.
   - Combines normalized similarity scores using a weighted late-fusion formula:
     $$\text{Score} = \alpha \cdot \text{Score}_{\text{text}} + (1 - \alpha) \cdot \text{Score}_{\text{visual}}$$
   - Ensures cross-modal query alignment (e.g., matching a floor plan layout directly to its written fire exit requirement).
4. **Context-Grounding Generator**:
   - Synthesizes answers using retrieved multimodal context, mitigating hallucination.
   - Can run with dummy embeddings or plug into real embedding & LLM APIs (OpenAI, HuggingFace, Ollama, etc.).

---

## 📁 Repository Structure

```
blueprint-chat/
├── data/
│   ├── manuals.json          # Sample technical specifications & building manual guidelines
│   └── blueprints.json       # Structural diagrams & blueprint metadata
├── src/
│   ├── __init__.py
│   ├── vector_store.py       # Dual-index in-memory / Chroma-compatible store with cosine similarity
│   ├── reranker.py           # Late-fusion reranker scoring text and visual matches
│   └── rag_pipeline.py       # End-to-end multimodal retrieval and answer generation pipeline
├── main.py                   # CLI demo & query interface
├── requirements.txt          # Minimal dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Requirements
Python 3.10+ (tested on Python 3.12 / 3.14). No heavy GPU requirements needed for the demo mode.

```bash
pip install -r requirements.txt
```

### 2. Run the Demo
Execute the interactive query runner:

```bash
python3 main.py
```

You can also pass custom queries:
```bash
python3 main.py --query "What is the fire exit evacuation protocol for Level 2 labs?"
python3 main.py --query "Where are the high-voltage electrical panels located in Sector B?" --alpha 0.4
```

`--alpha` controls the weighting between text context (`alpha=1.0`) and blueprint layout features (`alpha=0.0`). The default late-fusion weight is `0.55`.

---

## 🔬 How Late-Fusion Works

Standard naive multimodal RAG projects either concatenate raw OCR text to image labels or flatten everything into a single dense vector, often losing fine-grained visual details. 

In **Blueprint-Chat**:
```
User Query: "Fire safety evacuation route for Lab B-201"
       │
       ├───► [Text Retriever]   ───► Top text safety regulations & building codes
       │
       └───► [Visual Retriever] ───► Top blueprint diagrams & room schematics
                   │
                   ▼
       [Late-Fusion Reranker]
                   │  (Weighted score merging & threshold filtering)
                   ▼
       [Unified Multimodal Context]
                   │
                   ▼
          [Answer Generation]
```

---

## 📄 License
MIT License. Created for course project documentation.
