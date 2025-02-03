import re
import json

# Load stopwords file
stop_words = set()
try:
    with open("StopWords.txt", 'r', encoding='utf-8') as file:
        stop_words = set(word.strip().lower() for word in file.readlines())
except FileNotFoundError:
    print("Stopwords file not found. Using an empty stopwords list.")
    

# Preprocess a query by tokenizing, removing punctuation, numbers, stopwords, and extra spaces.
def preprocess_text(text):
       
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
def build_index(corpus_file):
    index = {}
    
    
    with open(corpus_file, 'r', encoding='utf-8') as file:
        for line in file:
            doc = json.loads(line)
            doc_id = doc["_id"]

            # Build document vocabulary by combining title and content
            text = doc["title"] + " " + doc["text"]
            words = preprocess_text(text)

            for word in words:
                if word not in index:
                    index[word] = set()
                index[word].add(doc_id)
    
    return index

# Example text processing
query = "<p>This is an example query, with numbers 123 and punctuation!?!?!</p>"
processed_query = preprocess_text(query)
print(processed_query)

# Example index build (print most common words and how many docs they appear in)
index = build_index("./scifact/corpus.jsonl")
for key, value in index.items() :
    if len(value) > 1500:
        print(key, len(value))