import os
import pandas as pd
from processor import load_documents, clean_text, build_inverted_index
from vsm import VectorSpaceModel
from metrics import Evaluator

def load_queries(filepath):
    # read queries into simple list
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def load_relevance(filepath):
    # map query_id (line number) to list of relevant doc_ids
    relevance = {}
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # parse ints from line, ignoring non-digits
                relevance[i + 1] = [int(x) for x in line.split() if x.isdigit()]
    return relevance

# paths
base_path = "collection"
docs_path = os.path.join(base_path, "docs")
queries_path = os.path.join(base_path, "Queries.txt")
relevance_path = os.path.join(base_path, "Relevant.txt")
# load docs
print("--- ΦΑΣΗ 1: Φόρτωση Δεδομένων ---")
df_docs = load_documents(docs_path)
df_docs['tokens'] = df_docs['raw_text'].apply(clean_text) # Note that this affects dataframe manipulation in index building

# build index
index = build_inverted_index(df_docs)
print(f"Index built with {len(index)} terms.")

# init vsm & evaluator
print("\n--- ΦΑΣΗ 2: Αρχικοποίηση VSM ---")
vsm = VectorSpaceModel(index, len(df_docs))
evaluator = Evaluator()

# load queries & relevance
if os.path.exists(queries_path):
    queries = load_queries(queries_path)
    relevance_data = load_relevance(relevance_path)
    
    # setup single query test (Query #1)
    q_id = 1
    q_text = queries[q_id - 1] # list index starts at 0
    
    print(f"\nRunning Query #{q_id}: '{q_text}'")
    
    # search
    q_tokens = clean_text(q_text)
    results = vsm.search(q_tokens, scheme='alternative')
    results_show = results[:10] # top 10 results

    # print results
    print("\n--- Results (Top 10) ---")
    print(f"{'Rank':<5} {'DocID':<10} {'Score':<10}")
    print("-" * 30)
    
    for rank, (doc_id, score) in enumerate(results_show, 1):
        print(f"{rank:<5} {doc_id:<10} {score:.4f}")

    retrieved_ids = []
    for rank, (doc_id, score) in enumerate(results, 1):
        retrieved_ids.append(doc_id)

    # evaluation
    print("\n--- Evaluation ---")
    relevant_ids = relevance_data.get(q_id, [])
    
    if relevant_ids:
        # calc metrics
        p = evaluator.calculate_precision(retrieved_ids, relevant_ids)
        r = evaluator.calculate_recall(retrieved_ids, relevant_ids)
        f1 = evaluator.calculate_f1(p, r)
        p_at_5 = evaluator.calculate_precision_at_k(retrieved_ids, relevant_ids, k=10)
        
        print(f"Precision: {p:.4f}")
        print(f"Recall:    {r:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"P@5:       {p_at_5:.4f}")

        print("\nPlotting Precision-Recall Curve...")
        evaluator.plot_precision_recall_curve(retrieved_ids, relevant_ids)
    else:
        print("No relevant docs found for this query.")

else:
    print("Queries file not found.")