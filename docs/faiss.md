# FAISS Overview

This project uses FAISS in three places: phishing case document search, keyword search for risk signals, and chat/guide retrieval. The common pattern is `IndexFlatIP + IndexIDMap2` with L2-normalized vectors to get cosine similarity while allowing add/update/remove by ID.

## 1) App-level FAISS store (phishing case docs)

### Where it is initialized
File: `app/main.py`

```python
faiss_index_path = os.getenv("FAISS_INDEX_PATH", "./data/index.faiss")
embed_model_name = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
app.state.faiss_store = FaissStore(index_path=faiss_index_path, model_name=embed_model_name)
```

Why: The index and embedding model are created once at startup to avoid per-request model loading and to share a single in-memory FAISS instance across endpoints.

### Store implementation
File: `app/faiss/faiss_store.py`

```python
base = faiss.IndexFlatIP(self.dim)
idx = faiss.IndexIDMap2(base)
```

Why: `IndexFlatIP` is simple and stable for cosine similarity when vectors are L2-normalized, and `IndexIDMap2` enables `add_with_ids` / `remove_ids` to keep the index in sync with DB row IDs.

```python
emb = self.model.encode(texts, convert_to_numpy=True).astype("float32")
faiss.normalize_L2(emb)
```

Why: L2-normalization makes inner product equal cosine similarity, which is the expected similarity metric for semantic search.

### CRUD endpoints that sync FAISS
File: `app/api/v1/endpoints/phising_docs.py`

```python
faiss_store.add([doc.id], [doc.text])
faiss_store.save()
```

Why: The DB is the source of truth; FAISS acts as a fast retrieval cache. Each write updates FAISS immediately to keep search results current.

```python
faiss_store.remove([doc_id])
faiss_store.save()
...
rebuild_faiss_from_db(db, faiss_store)
```

Why: Some FAISS builds can fail on `remove_ids`. The rebuild fallback keeps the index consistent with DB even if a direct delete fails.

### Admin endpoints
File: `app/api/v1/endpoints/admin_faiss.py`

```python
faiss_store.index = faiss_store._load_or_create()
faiss_store.add([d.id for d in docs], [d.text for d in docs])
faiss_store.save()
```

Why: This provides a full rebuild from DB as a recovery path and a maintenance action for admins.

## 2) Keyword FAISS store (risk signal boosting)

There are two similar keyword stores:
- `app/db/models/phising_sign.py`
- `app/services/text_infer.py`

They both use TF-IDF vectors and a stable keyword ID mapping.

### Stable keyword ID mapping
File: `app/db/models/phising_sign.py` (also mirrored in `app/services/text_infer.py`)

```python
h = hashlib.md5(s.encode("utf-8")).hexdigest()
return int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF
```

Why: Python's built-in hash is randomized per process. A stable hash ensures keyword IDs stay the same across restarts so the FAISS index and JSON metadata remain aligned.

### Index creation and search
File: `app/db/models/phising_sign.py`

```python
base = faiss.IndexFlatIP(self.dim)
return faiss.IndexIDMap2(base)
```

Why: Same pattern as the main store: cosine similarity with stable add/remove by keyword ID.

```python
x = self.vec.transform(cleaned).toarray().astype("float32")
x = _l2_normalize(x)
```

Why: TF-IDF vectors are normalized so inner product equals cosine similarity.

### Keyword usage in detection
File: `app/db/models/phising_sign.py`

```python
if (not keywords) and self.kw_store is not None:
    faiss_hits = self.kw_store.search(sentence, topk=faiss_topk, min_sim=faiss_min_sim)
    keywords = [h["keyword"] for h in faiss_hits]
```

Why: If callers do not provide keywords, FAISS supplies them automatically to strengthen detection without requiring extra client input.

File: `app/services/text_infer.py`

```python
keyword_risk = max(keyword_risk, self.cfg.keyword_force_risk)
keyword_risk = min(1.0, keyword_risk + (kw_count * self.cfg.keyword_bonus_per_hit))
```

Why: This intentionally boosts risk when suspicious keywords are found, even if the AE model is not strongly suspicious, so keyword hits cannot be ignored.

### Keyword admin API
File: `app/api/v1/endpoints/faiss_keywords.py`

```python
result = kw_store.upsert(payload.keywords)
```

Why: The API lets operators upsert or rebuild the keyword index without code changes, enabling fast updates to keyword lists.

## 3) Chat and guide retrieval (FAISS over DB)

### Chat FAISS store
File: `app/api/v1/endpoints/chat/chat_faiss.py`

```python
base = faiss.IndexFlatIP(dim)
self._index = faiss.IndexIDMap2(base)
```

Why: For chat retrieval, a simple in-memory FAISS index is rebuilt from DB and supports quick updates via ID mapping.

```python
overfetch = min(max(k * 10, k), 200)
D, I = self._index.search(qv, overfetch)
```

Why: Overfetching retrieves more candidates so that post-filters (category/min_score) still return enough results.

### Guide FAISS store
File: `app/api/v1/endpoints/chat/chat_guide.py`

```python
return f"{d.title}\n{d.content}\nKEY:{d.key}\n{meta_txt}".strip()
```

Why: The guide index embeds title, content, key, and metadata together to maximize recall for retrieval.

## Paths and defaults

- Case FAISS index: `FAISS_INDEX_PATH` (default `./data/index.faiss`)
- Keyword FAISS index: `assets/faiss/keyword.index`
- Keyword metadata: `assets/faiss/keyword_meta.json`
- Sample keyword list: `assets/data/faiss-keywords.txt`

## Operational notes

- If the index appears inconsistent, use the rebuild endpoint in `app/api/v1/endpoints/admin_faiss.py`.
- Keyword indexes are stored on disk and survive restarts; chat/guide indexes are rebuilt from DB.

## RAG flow diagram

![RAG flow diagram](rag-flow.svg)

Why: The diagram shows the shared pattern: FAISS retrieves candidates quickly, then the system applies domain logic (DB lookups or risk scoring) to produce final outputs.
