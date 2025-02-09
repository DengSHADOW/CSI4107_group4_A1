from IRsystem import IRsystem # import IRsystem class


# Example text processing
# query = "<p>This is an example query, with numbers 123 and punctuation!?!?!</p>"
# print(IR.preprocess(query))

# Example index build (print most common words and how many docs they appear in)
# IR.build_index("./scifact/corpus.jsonl")
# index = IR.get_index()
# for key, value in index.items() :
#     if len(value) > 1500:
#         print(key, len(value))

# Sample of 100 tokens from your vocabulary. Include the first 10 answers to the first 2 queries. 

# Run system to generate output
#   - Use only test queries (the queries with odd numbers 1.3.5, …)
#   - Each query has rank 100 documents in the decending order
#   - Output Results.txt
# Run IR system for test queries

IR = IRsystem()
IR.build_index("./scifact/corpus.jsonl")
IR.save_results("scifact/queries.jsonl", "Results.txt")



# Evaluation
#   - Include the Mean Average Precision (MAP) score computed with trec_eval for the results on the test queries.  


# Sample display (comment it if not use)
#   - How big was the vocabulary? 
#   - Include a sample of 100 tokens from your vocabulary. 
#   - Include the first 10 answers to the first 2 queries. 

# IR.display_samples("scifact/queries.jsonl") # default display top 10