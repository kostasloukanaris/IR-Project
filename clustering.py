import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from processor import load_documents, clean_text


docs_path = "collection/docs"
graphs_dir = "clustering_graphs"

#best params for tf-idf vectorization
best_tfidf_params = {
    'ngram_range': (1, 1),
    'min_df': 1,
    'max_df': 0.8,
    'sublinear_tf': True,
    'norm': 'l2'
}

def ensure_dir(directory): #for graph dir
    #creates directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_optimal_k(data, max_k=10, title="Silhouette Analysis"):
    #runs k-means for k=2 to max_k using only silhouette score
    print(f"\n--- Finding Optimal K for {title} ---")
    silhouettes = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        
        score = silhouette_score(data, labels)
        silhouettes.append(score)
        print(f"k={k} | Silhouette Score: {score:.4f}")

    #plot silhouette score
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, silhouettes, marker='o', linestyle='--', color='tab:blue')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title(f'Optimal k Analysis (Silhouette): {title}')
    plt.grid(True)
    plt.tight_layout()
    
    #save to specific folder
    filename = os.path.join(graphs_dir, f"optimal_k_{title.replace(' ', '_')}.png")
    plt.savefig(filename)
    print(f"Saved analysis plot to {filename}")
    
    #return k with highest silhouette score
    best_k = k_range[np.argmax(silhouettes)]
    return best_k

def visualize_clusters(data, labels, title):
    #reduces data to 2d using pca and plots the clusters
    print(f"Visualizing {title}...")
    
    #convert to dense array if sparse (for tf-idf)
    dense_data = data.toarray() if hasattr(data, 'toarray') else data
    
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(dense_data)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.6, s=10)
    plt.title(f'Cluster Visualization (PCA): {title}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Cluster Label')
    
    filename = os.path.join(graphs_dir, f"cluster_viz_{title.replace(' ', '_')}.png")
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")

if __name__ == "__main__":
    #setup directory
    ensure_dir(graphs_dir)
    
    #load data
    print("Loading Data...")
    df_docs = load_documents(docs_path)
    raw_documents = df_docs['raw_text'].tolist()
    
    #TF-IDF VECTORIZER CLUSTERING
    print("\n" + "="*50)
    print("PART A: Clustering with TF-IDF")
    print("="*50)
    
    #generate vectors using best params
    vectorizer = TfidfVectorizer(stop_words='english', **best_tfidf_params)
    tfidf_matrix = vectorizer.fit_transform(raw_documents)
    
    #find best k automatically
    best_k_tfidf = find_optimal_k(tfidf_matrix, max_k=8, title="TF-IDF")
    print(f"-> Suggested k for TF-IDF: {best_k_tfidf}")
    
    #run final clustering
    kmeans_tfidf = KMeans(n_clusters=best_k_tfidf, random_state=42, n_init=10)
    labels_tfidf = kmeans_tfidf.fit_predict(tfidf_matrix)
    
    #visualize
    visualize_clusters(tfidf_matrix, labels_tfidf, "TF-IDF")

    #BERT EMBEDDINGS CLUSTERING
    print("\n" + "="*50)
    print("PART B: Clustering with BERT Embeddings")
    print("="*50)
    
    #load hugging face model
    print("Loading Hugging Face model (all-MiniLM-L6-v2)...")
    #point this to your downloaded folder if needed
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    #encode documents
    print("Generating dense embeddings...")
    embeddings_matrix = model.encode(raw_documents)
    
    #find best k automatically
    best_k_bert = find_optimal_k(embeddings_matrix, max_k=8, title="BERT Embeddings")
    print(f"-> Suggested k for Embeddings: {best_k_bert}")
    
    #run final clustering
    kmeans_bert = KMeans(n_clusters=best_k_bert, random_state=42, n_init=10)
    labels_bert = kmeans_bert.fit_predict(embeddings_matrix)
    
    #visualize
    visualize_clusters(embeddings_matrix, labels_bert, "BERT_Embeddings")
    
    print(f"\nDone! Check the '{graphs_dir}' folder for the plots.")
