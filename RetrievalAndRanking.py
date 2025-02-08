from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np

# Load and preprocess documents from corpus.jsonl
def load_corpus(corpus_file):
    doc_texts = []
    doc_ids = []

    with open(corpus_file, 'r', encoding='utf-8') as file:
        for line in file:
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            text = doc.get("title", "") + " " + doc.get("text", "")
            doc_texts.append(text)
            doc_ids.append(doc_id)

    return doc_texts, doc_ids

# Convert corpus into TF-IDF vectors
corpus_file = "scifact/corpus.jsonl"
doc_texts, doc_ids = load_corpus(corpus_file)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(doc_texts)  # Document-term matrix

# Function to search using cosine similarity
def search_query_tfidf(query_text, top_k=10):
    query_vec = vectorizer.transform([query_text])  # Convert query to vector
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()  # Compute similarity

    ranked_indices = np.argsort(similarity_scores)[::-1]  # Rank by descending score
    ranked_results = [(doc_ids[i], similarity_scores[i]) for i in ranked_indices[:top_k]]

    return ranked_results

# Example Query
query = "scientific research on vaccines"
results = search_query_tfidf(query)
print("TF-IDF Results:", results)
