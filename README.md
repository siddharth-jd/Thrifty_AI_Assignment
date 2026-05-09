# Simple RAG System

A minimal Retrieval Augmented Generation (RAG) pipeline built in Python

---

## Approach

The pipeline has four steps:

### 1. Retrieval (`retrieve`)

Uses **TF-IDF cosine similarity** to find the most relevant documents for a query.

- Each document and the query are converted into TF-IDF vectors (term frequency × inverse document frequency).
- Cosine similarity is computed between the query vector and every document vector.
- The top-k documents by score are returned.

TF-IDF was chosen because it's simple, interpretable, and works well for short documents

### 2. Prompt Construction (`build_prompt`)

Builds a clearly structured prompt with:

- A **Context** block (the retrieved documents as bullet points)
- A **Question** line
- An explicit instruction to answer only from the provided context

This separation ensures the answer generation step stays grounded and doesn't imagine stuff.

### 3. Answer Generation (`generate_answer`)

Rule based extraction (no LLM needed):

- Parses the context sentences and question from the prompt.
- Filters out stopwords (e.g. "what", "in", "the") to focus on content words.
- Scores each context sentence by Jaccard similarity with the question's content words.
- Returns the best matching sentence as the answer.

The answer is always a sentence from the retrieved context, so it is fully grounded by design.

### 4. Evaluation (`evaluate`)

Scores the answer from 0 to 1:

- Computes token level overlap between the answer and each context sentence.
- Score = `overlapping tokens / total answer tokens` (capped at 1.0).
- Measures **groundedness**: is the answer supported word for word by the context?

---

## Design Choices

| Choice                            | Reason                                                              |
| --------------------------------- | ------------------------------------------------------------------- |
| TF-IDF over embeddings            | No external libraries required; sufficient for short docs           |
| Hyphen-aware tokenizer            | Keeps terms like `Chandrayaan-2` as one token for accurate matching |
| Stopword filtering in generation  | Prevents common words like "in" from skewing sentence selection     |
| Jaccard similarity for generation | Penalises length; relevance matters more than word count            |
| Extraction based generation       | Guarantees the answer is grounded; no risk of hallucination         |

---

## Assumptions

- Documents and queries are short and factual, so TF-IDF is sufficient for retrieval.
- Answer generation is simulated via extraction, not a real LLM, as per the constraint that external APIs are not permitted.
- The evaluation metric measures **groundedness** (is the answer supported by context?) rather than factual correctness against a labelled ground truth.

---

## Instructions to Run

**Requirements:** Python 3.6 or higher. No third party packages needed.

**Step 1:** Place all four files in the same folder:

```
rag_system/
├── rag.py
├── docs.json
├── queries.json
└── README.md
```

**Step 2:** Open a terminal and navigate to the folder:

```bash
cd rag_system
```

**Step 3:** Run the script:

```bash
python rag.py
```

If `python` doesn't work, try:

```bash
python3 rag.py
```

---

## Sample Output

```
Query: Who launched Chandrayaan-3?
Retrieved Context: ['India launched Chandrayaan-3 in 2023.', 'Chandrayaan-2 had a partial failure during landing.']
Answer: India launched Chandrayaan-3 in 2023.
Score: 1.00
------------------------------------------------------------
Query: What happened in Chandrayaan-2?
Retrieved Context: ['India launched Chandrayaan-3 in 2023.', 'Chandrayaan-2 had a partial failure during landing.']
Answer: Chandrayaan-2 had a partial failure during landing.
Score: 1.00
------------------------------------------------------------
Query: Which organization is ISRO?
Retrieved Context: ['ISRO is the Indian Space Research Organisation.', 'Chandrayaan-2 had a partial failure during landing.']
Answer: ISRO is the Indian Space Research Organisation.
Score: 1.00
------------------------------------------------------------
```

---

## File Structure

```
rag_system/
├── rag.py          # Main pipeline implementation
├── docs.json       # Document corpus
├── queries.json    # User queries
└── README.md       # This file
```
