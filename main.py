from PreprocessingAndIndexing import preprocess, build_index
# import RetrievalAndRanking
from collections import defaultdict


# Load stopwords file
stop_words = set()
try:
    with open("StopWords.txt", 'r', encoding='utf-8') as file:
        stop_words = set(word.strip().lower() for word in file.readlines())
except FileNotFoundError:
    print("Stopwords file not found. Using an empty stopwords list.")


# Example text processing
query = "<p>This is an example query, with numbers 123 and punctuation!?!?!</p>"
processed_query = preprocess(query, stop_words)
print(processed_query)

# Example index build (print most common words and how many docs they appear in)
index = build_index("./scifact/corpus.jsonl", stop_words)
for key, value in index.items() :
    if len(value) > 1500:
        print(key, len(value))