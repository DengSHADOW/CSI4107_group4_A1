from collections import defaultdict
import json
import re

# Preprocess a query by tokenizing, removing punctuation, numbers, stopwords, and extra spaces.
def preprocess(text, stop_words):
       
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize by splitting on whitespace
    tokens = text.lower().split()
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return filtered_tokens

# Build index: map unique vocab words to documents (IDs) containing them
# Use map of word --> set of IDs for fast lookup
def build_index(corpus_file, stop_words):
    """
    Builds an inverted index for the corpus.
    
    Args:
        corpus_file (str): Path to the corpus.jsonl file.
    
    Returns:
        dict: Inverted index mapping words to sets of document IDs.
    """
    
    # 
    index = defaultdict(set)
    
    with open(corpus_file, 'r', encoding='utf-8') as file:
        for line in file:
            doc = json.loads(line)
            doc_id = doc["_id"]

            # Build document vocabulary by combining title and content
            text = doc["title"] + " " + doc["text"]
            words = preprocess(text, stop_words)

            for word in words:
                # if word not in index:
                #     index[word] = set()
                index[word].add(doc_id)
    
    return index