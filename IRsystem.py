import re
import json
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class IRsystem:
    """
    Information Retrieval System using TF-IDF and Cosine Similarity.

    This class:
    - Loads and processes text (removing stopwords, punctuation).
    - Builds an inverted index mapping words to document IDs.
    - Stores cleaned document texts for efficient retrieval.
    - Performs document ranking using TF-IDF and Cosine Similarity.
    - Saves search results in TREC format.

    Attributes:
        stop_words (set): Set of stopwords for text cleaning.
        index (dict): Inverted index mapping words to document IDs.
        doc_texts (dict): Stores cleaned document texts.
        vectorizer (TfidfVectorizer): TF-IDF vectorizer instance.
    """

    def __init__(self, stopwords_file="StopWords.txt"):
        """
        Initialize the class, loading stopwords if provided.

        Args:
            stopwords_file (str): Path to stopwords file (optional).
        """
        # Load stopwords from StopWords.txt
        self.stop_words = set()
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as file:
                self.stop_words = set(word.strip().lower() for word in file.readlines())
        except FileNotFoundError:
            print("Stopwords file not found. Using an empty stopwords list.")
        
        # Empty Inverted index
        self.index = defaultdict(set)

        # Empty cleaned text dict (for retrieval) {Id: texts}
        self.docs = {}  

        # Initialize vectorizer for calculating TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words)

    def preprocess(self, text):
        """
        Cleans and tokenizes text by:
        - Removing HTML tags
        - Removing punctuation and numbers
        - Lowercasing
        - Removing stopwords

        Args:
            text (str): Raw text input.

        Returns:
            list: List of cleaned tokens.
        """
        text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove punctuation and numbers
        tokens = text.lower().split()  # Tokenize
        return [word for word in tokens if word not in self.stop_words]  # Remove stopwords

    def build_index(self, corpus_file):
        """
        Builds an inverted index and stores document texts for retrieval.

        Args:
            corpus_file (str): Path to corpus.jsonl file.

        Returns:
            None (Modifies self.index and self.docs)
        """
        with open(corpus_file, 'r', encoding='utf-8') as file:
            for line in file:
                doc = json.loads(line)
                doc_id = doc["_id"]
                text = doc.get("title", "") + " " + doc.get("text", "")
                
                # Preprocess text
                words = self.preprocess(text)
                self.docs[doc_id] = " ".join(words)  # Store cleaned text

                # Build inverted index
                for word in words:
                    self.index[word].add(doc_id)

        print("Inverted indexing, totally ", len(self.docs), "documents.")

    def RetrievalAndRanking(self, query_text, top_k=100):
        """
        Searches for relevant documents using TF-IDF and cosine similarity (using sklearn).

        Args:
            query_text (str): Raw user query.
            top_k (int): Number of top-ranked results.

        Returns:
            list: Ranked (doc_id, score) pairs.
        """

        # Preprocess the raw query
        query_tokens = self.preprocess(query_text)  

        # Find and retrieve relevant docs using index
        relevant_docs = set()
        for token in query_tokens:
            if token in self.index:
                relevant_docs.update(self.index[token])
        
        if not relevant_docs: # If no docs relevant
            return []

        # Extract all relevant documents' texts in dict from the cleaned texts dict
        doc_text_list = [self.docs[doc_id] for doc_id in relevant_docs if doc_id in self.docs]
        doc_ids = list(relevant_docs)

        # Apply TF-IDF Vectorization only on relevant documents 
        tfidf_matrix = self.vectorizer.fit_transform(doc_text_list) # Matrix for docs 
        query_vec = self.vectorizer.transform([" ".join(query_tokens)]) # Vector for processed query

        # Compute cosine similarity
        similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # Rank documents
        ranked_indices = np.argsort(similarity_scores)[::-1]
        ranked_results = [(doc_ids[i], similarity_scores[i]) for i in ranked_indices[:top_k]]

        return ranked_results
    
    def get_index(self):
        """Returns the inverted index."""
        return self.index

    def get_docs(self):
        """Returns the stored document texts."""
        return self.docs
