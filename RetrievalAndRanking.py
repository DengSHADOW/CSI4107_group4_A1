import math
from collections import defaultdict

def compute_tfidf(index, corpus_size):
    """
    Computes TF-IDF scores for terms in the index.
    
    Args:
        index (dict): Inverted index with term -> set of document IDs.
        corpus_size (int): Total number of documents.
    
    Returns:
        dict: TF-IDF scores in the format {term: {doc_id: tf-idf_score}}
    """
    tfidf_index = defaultdict(dict)
    
    for term, doc_ids in index.items():
        df = len(doc_ids)  # Document Frequency (DF)
        idf = math.log(corpus_size / (df + 1))  # Compute IDF (smoothing added)

        for doc_id in doc_ids:
            tf = 1 + math.log(len(doc_ids))  # Log-based TF
            tfidf_index[term][doc_id] = tf * idf  # Compute TF-IDF
    
    return tfidf_index
