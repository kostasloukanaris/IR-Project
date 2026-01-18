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
            
        scaled_query = {}
        query_length = 0
        
        #calc query weights and length
        for term, count in query_counts.items():
            scaled_comp_q = self.get_weight(count, term, scheme)
            scaled_query[term] = scaled_comp_q
            query_length += scaled_comp_q * scaled_comp_q #for |q| later for cosine similarity
            
        query_length = math.sqrt(query_length) #same here for |q|
        
        if query_length == 0:
            return []

        #score documents
        dot_prod = {}
        relevant_docs = set()

        for term, scaled_comp_q in scaled_query.items():
            if term in self.inverted_index:
                postings = self.inverted_index[term]
                
                for doc_id, doc_tf in postings:
                    relevant_docs.add(doc_id)
                    
                    #calc doc weight for the current term
                    scaled_comp_d = self.get_weight(doc_tf, term, scheme)
                    
                    #accumulate dot product
                    if doc_id not in dot_prod:
                        dot_prod[doc_id] = 0
                    dot_prod[doc_id] += scaled_comp_q * scaled_comp_d

        # Calculate document vector lengths on-the-fly for relevant docs
        doc_lengths = {}
        for doc_id in relevant_docs:
            len_sq = 0
            # To get all terms for a doc, we must iterate through the index
            # This is inefficient but necessary for on-the-fly calculation
            for term, postings in self.inverted_index.items():
                for post_doc_id, tf in postings:
                    if post_doc_id == doc_id:
                        weight = self.get_weight(tf, term, scheme)
                        len_sq += weight ** 2
                        break # Move to next term once doc is found
            doc_lengths[doc_id] = math.sqrt(len_sq)

        #finalize and sort
        results = []
        for doc_id, dot_product in dot_prod.items():
            doc_length = doc_lengths.get(doc_id, 1) # Default to 1 to avoid division by zero
            
            #full cosine similarity: dot_product / (query_length * doc_length)
            if query_length > 0 and doc_length > 0:
                final_score = dot_product / (query_length * doc_length)
            else:
                final_score = 0
            results.append([doc_id, final_score])
            
        #sort desc
        results.sort(key=lambda x: x[1], reverse=True) #sort by score (x[1]) descending (reverse=True)
        
        return results