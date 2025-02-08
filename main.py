from IRsystem import IRsystem # import IRsystem class

IR = IRsystem()
# Example text processing
query = "<p>This is an example query, with numbers 123 and punctuation!?!?!</p>"
print(IR.preprocess(query))

# Example index build (print most common words and how many docs they appear in)
IR.build_index("./scifact/corpus.jsonl")
index = IR.get_index()
for key, value in index.items() :
    if len(value) > 1000:
        print(key, len(value))


# Test queies (query_id: Query) [TODO]
#   - Use only test queries (the queries with odd numbers 1.3.5, …)
# Run IR system for test queies



# Output them with ranks in a file
#   - Results.txt
#   - Output form: query_id Q0 doc_id rank score tag
#   - top-100 results (already included in the code)



# Evaluation