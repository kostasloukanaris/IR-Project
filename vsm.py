import math

class VectorSpaceModel:
    def __init__(self, inverted_index, total_docs):
        self.inverted_index = inverted_index
        self.total_docs = total_docs
        self.idfs = {} 
        
        #calc idf for all terms
        self.calculate_idfs()

    def calculate_idfs(self):
        #calc idf = log(N/ni)
        for term, postings in self.inverted_index.items(): #term=word token, postings=list of [doc_id, tf]
            docs_with_term = len(postings) #num of docs containing the term

            #idf calculation
            idf_score = math.log10(self.total_docs / docs_with_term)
            self.idfs[term] = idf_score 

    def get_weight(self, tf, term, scheme):
        #get idf
        idf = self.idfs.get(term, 0)
        
        if scheme == 'classic':
            #classic tf-idf: tf * idf
            return tf * idf
            
        elif scheme == 'alternative':
            #log normalization-idf: (1 + log(tf)) * idf
            if tf > 0:
                return (1 + math.log10(tf)) * idf
            return 0

    def search(self, query_tokens, scheme='classic'):
        #prepare query vector
        query_counts = {}
        for word in query_tokens:
            query_counts[word] = query_counts.get(word, 0) + 1 #if you have counted it, increment, else start from 0
            
        query_weights = {}
        query_length = 0
        
        #calc query weights and length
        for term, count in query_counts.items():
            w = self.get_weight(count, term, scheme)
            query_weights[term] = w
            query_length += w * w #for |q| later for cosine similarity
            
        query_length = math.sqrt(query_length) #same here for |q|
        
        if query_length == 0:
            return []

        #score documents
        scores = {}
        
        for term, q_weight in query_weights.items():
            if term in self.inverted_index:
                postings = self.inverted_index[term]
                
                for doc_data in postings:
                    doc_id = doc_data[0]
                    doc_tf = doc_data[1]
                    
                    #calc doc weight
                    d_weight = self.get_weight(doc_tf, term, scheme)
                    
                    #accumulate dot product
                    if doc_id not in scores:
                        scores[doc_id] = 0
                    scores[doc_id] += q_weight * d_weight

        #finalize and sort
        results = []
        for doc_id, dot_product in scores.items():
            #cosine similarity: dot_product / query_length
            final_score = dot_product / query_length
            results.append([doc_id, final_score])
            
        #sort desc
        results.sort(key=lambda x: x[1], reverse=True) #sort by score (x[1]) descending (reverse=True)
        
        return results