# Custom GraphRAG Search Provider for MMCT

This project implements a **Graph-augmented Retrieval-Augmented Generation (GraphRAG)** search provider, built on top of NetworkX and FAISS, and integrated into the the MMCT framework as a custom `SearchProvider`. It allows indexing videos and chapters as a graph, performing semantic search, and querying related chapters via graph edges.

---

## Table of Contents
- [Custom GraphRAG Search Provider for MMCT](#custom-graphrag-search-provider-for-mmct)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Architecture](#architecture)
  - [Setup \& Installation](#setup--installation)
  - [Usage](#usage)
    - [Initializing the Search Provider](#initializing-the-search-provider)
    - [Indexing Documents](#indexing-documents)
    - [Searching](#searching)
  - [Code Structure](#code-structure)
  - [Example Workflow](#example-workflow)

---

## Features

* Maintains a **heterogeneous graph**:

  * Video nodes → chapter nodes → attribute nodes (metadata).
* Embedding-based semantic search over chapter embeddings (via FAISS).
* Graph-based expansion: after retrieving top chapters, fetch related chapters via shared attributes.
* Supports indexing (add new videos/chapters) and deletion.
* Wraps this functionality as an MMCT `SearchProvider`, pluggable via configuration.

---

## Architecture

1. **`DocumentGraph`** (in `document_graph_handler.py`)

   * Uses NetworkX to maintain nodes and edges:

     * Nodes: `video`, `chapter`, `attribute`
     * Edges: e.g. `video → chapter`, `chapter → attribute`
   * Maintains a stub of chapter embeddings for graph nodes.
   * Exposes `add_documents(...)`, `search(...)`, `fetch_related_chapters(...)`, etc.

2. **`FaissIndexManager`** (in `faiss_index_handler.py`)

   * Wraps a FAISS index (`IndexHNSWFlat` + `IndexIDMap`) keyed by integer IDs.
   * Maintains a bidirectional mapping `str_id ↔ int_id` to map chapter node IDs to FAISS IDs.
   * Supports `add_embeddings(...)`, `search(...)`, and saving/loading index and mappings.

3. **`CustomSearchProvider`** (in your provider implementation)

   * Subclasses `SearchProvider` in the MMCT framework.
   * Implements:

     * `search(...)` → wraps `DocumentGraph.search()`
     * `index_document(...)` → adds documents/videos to the graph + FAISS
     * `delete_document(...)` → removes nodes/embeddings (soft deletion in FAISS)

4. **Provider Factory Integration**

   * You configure `"custom_search"` as a provider via `SearchConfig`.
   * Use `provider_factory.create_search_provider(...)` to instantiate it.

---

## Setup & Installation

1. Place the modules under your MMCT extension directory (e.g. `mmct.providers.custom_providers.graph_rag/`):

   ```text
   mmct/
     providers/
       custom_providers/
         graph_rag/
           document_graph_handler.py
           faiss_index_handler.py
           document_graph_handler.py  ← alias or wrapper for DocumentGraph
   ```

2. Ensure dependencies are installed:

   ```bash
   pip install networkx faiss-cpu loguru scikit-learn
   ```

3. In your MMCT settings/config, specify:

   ```python
   from mmct.config.settings import SearchConfig

   search_config = SearchConfig(provider="custom_search")
   ```

4. The provider factory must recognize `"custom_search"` and import the class `CustomSearchProvider` from your module. For example, in your factory:

   ```python
   if provider_name == "custom_search":
       from mmct.providers.custom_providers.graph_rag.custom_search_provider import CustomSearchProvider
       return CustomSearchProvider(config)
   ```

---

## Usage

### Initializing the Search Provider

```python
from mmct.config.settings import SearchConfig
from mmct.providers.factory import provider_factory

search_config = SearchConfig(provider="custom_search")
search_provider = provider_factory.create_search_provider(
    search_config.provider, search_config.model_dump()
)
```

This will internally create an instance of `CustomSearchProvider` which instantiates the `DocumentGraph`.

### Indexing Documents

You should index your videos (and their chapters) before they can be searched.

This will:

* Add a video node and chapter nodes + edges in the graph.
* Add embedding vectors to the FAISS index.
* Persist both the graph and index to disk.

### Searching

```python
search_results = await search_provider.search(
    query="your natural language query",
    index_name="temp",
    embedding=your_query_embedding,  # e.g. 1×1536-d numpy vector
    top_k=5,
    top_n=3
)
```

* **`top_k`**: number of top chapters (via FAISS) to retrieve.
* **`top_n`**: number of *neighboring* chapters (via graph expansion) to return per top chapter.
* Returns a list of chapter info dicts, which may include nested neighbor chapters.

This:

* Removes nodes from the graph.
* Removes their ID mappings in the FAISS index (soft delete).
* Saves the updated graph and index mapping.

> ⚠️ Note: FAISS HNSW index does **not** support efficient deletion of vectors natively. The current implementation is a soft removal of the mapping; for heavy deletions you may need to rebuild the index from scratch.

---

## Code Structure

```
mmct/
  providers/
    custom_providers/
        graph_rag/
            document_graph_handler.py
            faiss_index_handler.py
    search_provider.py
```

* `document_graph_handler.py` — Implements the `DocumentGraph` class.
* `faiss_index_handler.py` — Implements `FaissIndexManager`.
* `custom_search_provider.py` — Implements the MMCT `SearchProvider` wrapper.
---

## Example Workflow

```python
# 1. Setup provider
search_config = SearchConfig(provider="custom_search")
search_provider = provider_factory.create_search_provider(
    search_config.provider, search_config.model_dump()
)

# Run search
query_embedding = ...  # e.g. output of text embedding model
results = await search_provider.search(
    query="some search query",
    embedding=query_embedding,
    top_k=5, 
    top_n=2
)
print(results)

```

---
