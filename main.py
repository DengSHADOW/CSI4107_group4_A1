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


# 