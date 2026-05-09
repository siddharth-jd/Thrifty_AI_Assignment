"""
Simple RAG (Retrieval-Augmented Generation) System
----------------------------------------------------
Loads documents and queries from JSON files, retrieves relevant context
using TF-IDF similarity, constructs a prompt, generates a rule based answer,
and evaluates the answer quality.
"""

import json
import math
import re
from collections import Counter
from typing import Dict, List


# ─────────────────────────────────────────────
# 1. RETRIEVAL
# ─────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """
    Lowercase and split text into tokens.
    Hyphenated words like 'Chandrayaan-2' are kept as one token
    so they match correctly across documents and queries.
    """
    return re.findall(r'\b\w+(?:-\w+)*\b', text.lower())


def build_tfidf(docs: List[Dict]) -> tuple:
    """
    Build TF-IDF vectors for all documents.
    Returns (tfidf_matrix, df) where tfidf_matrix[i] is a dict
    mapping term -> tfidf score for the i-th document, and df maps
    term -> number of documents containing that term.
    """
    tokenized = [tokenize(doc["text"]) for doc in docs]

    # Document frequency: how many docs contain each term
    df = Counter()
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1

    n = len(docs)
    tfidf_matrix = []

    for tokens in tokenized:
        tf = Counter(tokens)
        total = len(tokens)
        vec = {}
        if total > 0:
            for term, count in tf.items():
                term_tf = count / total
                term_idf = math.log((n + 1) / (df[term] + 1)) + 1  # smooth IDF
                vec[term] = term_tf * term_idf
        tfidf_matrix.append(vec)

    return tfidf_matrix, df


def cosine_similarity(vec_a: Dict, vec_b: Dict) -> float:
    """Cosine similarity between two TF-IDF vectors (dicts)."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0

    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    return dot / (norm_a * norm_b)


def retrieve(query: str, docs: List[Dict], tfidf_matrix: List[Dict], df: Counter, k: int = 2) -> List[str]:
    """
    Return the top-k document texts most relevant to the query,
    ranked by TF-IDF cosine similarity.
    Accepts pre-built tfidf_matrix and df to avoid recomputing per query.
    """
    query_tokens = tokenize(query)
    query_tf = Counter(query_tokens)
    total = len(query_tokens)
    n = len(docs)

    query_vec = {}
    if total > 0:
        for term, count in query_tf.items():
            term_tf = count / total
            term_idf = math.log((n + 1) / (df.get(term, 0) + 1)) + 1
            query_vec[term] = term_tf * term_idf

    # Score each document
    scores = []
    for i, doc_vec in enumerate(tfidf_matrix):
        score = cosine_similarity(query_vec, doc_vec)
        scores.append((score, i))

    scores.sort(reverse=True)
    top_k = scores[:k]

    return [docs[i]["text"] for _, i in top_k]


# ─────────────────────────────────────────────
# 2. PROMPT CONSTRUCTION
# ─────────────────────────────────────────────

def build_prompt(query: str, context: List[str]) -> str:
    """
    Build a structured prompt that clearly separates context from the question.
    The model (or rule-based system) is instructed to answer only from context.
    """
    context_block = "\n".join(f"- {c}" for c in context)
    prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer (based only on the context above):"
    )
    return prompt


# ─────────────────────────────────────────────
# 3. ANSWER GENERATION
# ─────────────────────────────────────────────

STOPWORDS = {
    "what", "who", "which", "where", "when", "how", "is", "are", "was",
    "were", "did", "do", "does", "the", "a", "an", "in", "on", "at",
    "to", "of", "and", "or", "for", "with", "that", "this", "it",
    "by", "from", "had", "has", "have", "be", "been", "its"
}


def meaningful_tokens(text: str) -> set:
    """Tokenize and remove stopwords to focus on content words."""
    return {t for t in tokenize(text) if t not in STOPWORDS}


def generate_answer(prompt: str) -> str:
    """
    Rule-based answer generation grounded in the provided context.
    Extracts the context lines and question from the prompt, then
    picks the context sentence whose content words best overlap with
    the content words of the question.
    """
    lines = prompt.split("\n")

    # Parse context lines (those starting with "- ")
    context_sentences = [
        line[2:] for line in lines if line.startswith("- ")
    ]

    # Parse the question
    question = ""
    for line in lines:
        if line.startswith("Question:"):
            question = line.replace("Question:", "").strip()
            break

    if not context_sentences or not question:
        return "I don't have enough information to answer."

    # Score each context sentence using content-word Jaccard similarity.
    # Ignoring stopwords avoids common function words ("in", "the") from
    # pulling in the wrong sentence.
    question_tokens = meaningful_tokens(question)
    best_sentence = context_sentences[0]
    best_score = -1.0

    for sentence in context_sentences:
        sentence_tokens = meaningful_tokens(sentence)
        intersection = question_tokens & sentence_tokens
        union = question_tokens | sentence_tokens
        jaccard = len(intersection) / len(union) if union else 0.0
        if jaccard > best_score:
            best_score = jaccard
            best_sentence = sentence

    return best_sentence.strip()


# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────

def evaluate(answer: str, context: List[str]) -> float:
    """
    Score the answer (0 to 1) based on how grounded it is in the context.

    Strategy:
    - Token overlap between the answer and each context sentence.
    - Score = (matched tokens) / (total tokens in answer)
    - Returns the max score across all context sentences.
    """
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0

    best_overlap = 0
    for sentence in context:
        ctx_tokens = set(tokenize(sentence))
        overlap = len(answer_tokens & ctx_tokens)
        if overlap > best_overlap:
            best_overlap = overlap

    score = best_overlap / len(answer_tokens)
    return round(min(score, 1.0), 2)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(docs_path: str, queries_path: str):
    """Load data and run the full RAG pipeline for each query."""
    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)
    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)

    # Build TF-IDF once for all queries
    tfidf_matrix, df = build_tfidf(docs)

    for item in queries:
        query = item["query"]

        # Step 1: Retrieve relevant documents
        context = retrieve(query, docs, tfidf_matrix, df, k=2)

        # Step 2: Build the prompt
        prompt = build_prompt(query, context)

        # Step 3: Generate an answer
        answer = generate_answer(prompt)

        # Step 4: Evaluate the answer
        score = evaluate(answer, context)

        # Display results
        print(f"Query: {query}")
        print(f"Retrieved Context: {context}")
        print(f"Answer: {answer}")
        print(f"Score: {score:.2f}")
        print("-" * 60)


if __name__ == "__main__":
    run_pipeline("docs.json", "queries.json")
