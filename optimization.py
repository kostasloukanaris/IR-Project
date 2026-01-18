import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from processor import load_documents, clean_text, build_inverted_index
from metrics import Evaluator
from vsm import VectorSpaceModel


docs_path = "collection/docs"
queries_path = "collection/Queries.txt"
relevance_path = "collection/Relevant.txt"


def load_queries_list(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def load_relevance_dict(filepath):
    relevance = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            relevance[i + 1] = [int(x) for x in line.split() if x.isdigit()]
    return relevance


def calculate_average_precision(retrieved_docs, relevant_docs):
    #calculates average precision (ap) for a single query
    if not relevant_docs:
        return 0.0
    
    relevant_set = set(relevant_docs)
    score = 0.0
    num_hits = 0.0
    
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_set:
            num_hits += 1.0
            #precision at this rank = hits / rank
            score += num_hits / (i + 1.0)
            
    #divide by total relevant docs
    return score / len(relevant_docs)

def get_interpolated_pr_curve(p_r_points):
    #interpolates precision at standard recall levels (0.0 to 1.0) for smooth plotting
    recall_levels = np.linspace(0.0, 1.0, 11)
    interpolated_p = []
    
    for r in recall_levels:
        #find max precision for recall >= r
        precisions = [p for p, rec in p_r_points if rec >= r]
        if precisions:
            interpolated_p.append(max(precisions))
        else:
            interpolated_p.append(0.0)
            
    return recall_levels, interpolated_p

def evaluate_and_track(retrieved_lists, relevance_data, evaluator):
    #calculates aggregate metrics for a list of results
    total_p = 0
    total_r = 0
    total_f1 = 0
    total_p10 = 0
    total_map = 0
    
    #store all (precision, recall) points for curve generation
    all_pr_points = []

    for q_id, retrieved_ids in retrieved_lists:
        relevant_ids = relevance_data.get(q_id, [])
        if relevant_ids:
            #metrics
            p = evaluator.calculate_precision(retrieved_ids, relevant_ids)
            r = evaluator.calculate_recall(retrieved_ids, relevant_ids)
            f1 = evaluator.calculate_f1(p, r)
            p10 = evaluator.calculate_precision_at_k(retrieved_ids, relevant_ids, k=10)
            ap = calculate_average_precision(retrieved_ids, relevant_ids)
            
            total_p += p
            total_r += r
            total_f1 += f1
            total_p10 += p10
            total_map += ap
            
            #collect points for pr curve
            current_hits = 0
            for k, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_ids:
                    current_hits += 1
                    current_p = current_hits / k
                    current_r = current_hits / len(relevant_ids)
                    all_pr_points.append((current_p, current_r))

    n = len(retrieved_lists)
    if n == 0: n = 1
    
    stats = {
        'Precision': total_p / n,
        'Recall': total_r / n,
        'F1': total_f1 / n,
        'P@10': total_p10 / n,
        'MAP': total_map / n
    }
    
    #calculate curve data
    curve_x, curve_y = get_interpolated_pr_curve(all_pr_points)
    
    return stats, (curve_x, curve_y)


if __name__ == "__main__":
    #load data
    print("Loading Data...")
    df_docs = load_documents(docs_path)
    raw_documents = df_docs['raw_text'].tolist() 
    doc_ids = df_docs['doc_id'].tolist()
    
    queries = load_queries_list(queries_path)
    relevance_data = load_relevance_dict(relevance_path)
    evaluator = Evaluator()

    
    
    #define parameters to test (36 combinations)
    params = {
        'ngram_range': [(1, 1), (1, 2)],        #unigrams vs bigrams
        'min_df': [1, 2, 5],                    #ignore terms appearing in < n docs
        'max_df': [0.8, 0.9, 1.0],              #ignore terms appearing in > n% docs
        'sublinear_tf': [True, False]           #logarithmic tf vs raw tf
    }
    
    keys = params.keys()
    combinations = list(itertools.product(*params.values()))
    
    print(f"\nPhase 1: Optimization (Testing {len(combinations)} configs)")
    
    #store best scores
    best_scores = {'MAP': -1, 'P@10': -1}
    best_params = {}
    
    for i, combo in enumerate(combinations, 1):
        args = dict(zip(keys, combo))
        try:
            vectorizer = TfidfVectorizer(norm='l2', stop_words='english', **args)
            tfidf_matrix = vectorizer.fit_transform(raw_documents) #build index and clean and calc idf and calc tf-idf weights for all docs 
            query_matrix = vectorizer.transform(queries) #transform queries to tf-idf vectors (we dont use fit cause we dont want to change the index we only want to transform the queries)
            similarity_matrix = cosine_similarity(query_matrix, tfidf_matrix)
            
            total_map = 0
            total_p10 = 0
            
            for q_idx, query_scores in enumerate(similarity_matrix):
                q_id = q_idx + 1

                # Pair each document ID with its calculated similarity score
                doc_id_score_pairs = zip(doc_ids, query_scores)

                # Sort these pairs in descending order based on the score.
                # The score is the second element (index 1) in each (doc_id, score) pair.
                ranked_doc_pairs = sorted(doc_id_score_pairs, key=lambda pair: pair[1], reverse=True)

                # Create a list of retrieved document IDs, but only include those
                # with a similarity score greater than 0.
                retrieved_ids = [doc_id for doc_id, score in ranked_doc_pairs if score > 0]
                
                relevant_ids = relevance_data.get(q_id, [])
                
                if relevant_ids:
                    #calc map and p@10 for decision
                    total_map += calculate_average_precision(retrieved_ids, relevant_ids)
                    total_p10 += evaluator.calculate_precision_at_k(retrieved_ids, relevant_ids, k=10)

            map_score = total_map / len(queries)
            p10_score = total_p10 / len(queries)
            
            print(f"[{i}] {args} -> MAP: {map_score:.4f} | P@10: {p10_score:.4f}")
            
            #decision logic: hierarchical comparison
            #prioritize map
            if map_score > best_scores['MAP']:
                best_scores = {'MAP': map_score, 'P@10': p10_score}
                best_params = args
            #if map is tied, use p@10
            elif map_score == best_scores['MAP']:
                if p10_score > best_scores['P@10']:
                    best_scores = {'MAP': map_score, 'P@10': p10_score}
                    best_params = args
                
        except Exception as e:
            print(f"Error: {e}")

    print(f"\nBEST PARAMS FOUND: {best_params}")
    print(f"Stats -> MAP: {best_scores['MAP']:.4f} | P@10: {best_scores['P@10']:.4f}\n")

    
    # --- Phase 2: Final Comparison of 3 Models ---
    
    results_summary = []
    
    # 1. Optimized Sklearn Model
    print("\n1. Running Optimized Sklearn Model...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(norm='l2', stop_words='english', **best_params)
    tfidf_matrix = vectorizer.fit_transform(raw_documents)
    
    idx_time = time.time() - t0
    
    t1 = time.time()
    query_matrix = vectorizer.transform(queries)
    similarity_matrix = cosine_similarity(query_matrix, tfidf_matrix)
    
    sklearn_results = []
    for q_idx, query_scores in enumerate(similarity_matrix):
        doc_id_score_pairs = zip(doc_ids, query_scores)

        ranked_doc_pairs = sorted(doc_id_score_pairs, key=lambda pair: pair[1], reverse=True)

        retrieved_ids = [doc_id for doc_id, score in ranked_doc_pairs if score > 0]

        sklearn_results.append((q_idx + 1, retrieved_ids))

    search_time = time.time() - t1
    
    stats_sklearn, curve_sklearn = evaluate_and_track(sklearn_results, relevance_data, evaluator)
    results_summary.append({
        'Model': 'Optimized Sklearn', 
        'Index Time': idx_time, 
        'Search Time': search_time, 
        **stats_sklearn
    })

    # --- Prepare for Custom Models ---
    print("\nPreparing Custom VSM (Tokenization)...")
    df_docs['tokens'] = df_docs['raw_text'].apply(clean_text)
    
    # 2. Custom VSM (Classic)
    print("2. Running Custom VSM (Classic)...")
    t0 = time.time()
    index = build_inverted_index(df_docs)
    vsm = VectorSpaceModel(index, len(df_docs))
    idx_time_custom = time.time() - t0 # Same index for both custom models
    
    t1 = time.time()
    custom_classic_results = []
    for q_id, q_text in enumerate(queries, 1):
        q_tokens = clean_text(q_text)

        search_results = vsm.search(q_tokens, scheme='classic')

        retrieved_ids = [doc_id for doc_id, score in search_results]

        custom_classic_results.append((q_id, retrieved_ids))

    search_time = time.time() - t1
    
    stats_classic, curve_classic = evaluate_and_track(custom_classic_results, relevance_data, evaluator)
    results_summary.append({
        'Model': 'Custom VSM (Classic)', 
        'Index Time': idx_time_custom, 
        'Search Time': search_time, 
        **stats_classic
    })

    # 3. Custom VSM (Alternative)
    print("3. Running Custom VSM (Alternative)...")
    # Index is already built, so index time is effectively 0
    t1 = time.time()
    custom_alt_results = []
    for q_id, q_text in enumerate(queries, 1):
        q_tokens = clean_text(q_text)

        search_results = vsm.search(q_tokens, scheme='alternative') 

        retrieved_ids = [doc_id for doc_id, score in search_results]

        custom_alt_results.append((q_id, retrieved_ids))

    search_time = time.time() - t1
    
    stats_alt, curve_alt = evaluate_and_track(custom_alt_results, relevance_data, evaluator)
    results_summary.append({
        'Model': 'Custom VSM (Alternative)', 
        'Index Time': idx_time_custom, # Re-using index time from classic model
        'Search Time': search_time, 
        **stats_alt
    })

    
    # --- Final Results ---
    
    print("\n" + "="*80)
    print(f"{'Model':<25} | {'Idx(s)':<8} | {'Srch(s)':<8} | {'MAP':<8} | {'P@10':<8} | {'F1':<8}")
    print("-" * 80)
    for res in results_summary:
        print(f"{res['Model']:<25} | {res['Index Time']:<8.4f} | {res['Search Time']:<8.4f} | {res['MAP']:<8.4f} | {res['P@10']:<8.4f} | {res['F1']:<8.4f}")
    print("="*80)
    
    # --- Plot PR Curves ---
    
    plt.figure(figsize=(10, 6))
    plt.plot(curve_sklearn[0], curve_sklearn[1], label=f"Sklearn (MAP={stats_sklearn['MAP']:.3f})", linewidth=2)
    plt.plot(curve_classic[0], curve_classic[1], label=f"Custom Classic (MAP={stats_classic['MAP']:.3f})", linestyle='--')
    plt.plot(curve_alt[0], curve_alt[1], label=f"Custom Alternative (MAP={stats_alt['MAP']:.3f})", linestyle='-.')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Comparison of Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('pr_curve_comparison.png')
    print("\nPlot saved as pr_curve_comparison.png")
