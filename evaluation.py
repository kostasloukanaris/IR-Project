import matplotlib.pyplot as plt

class Evaluator:
    
    def calculate_precision(self, retrieved_ids, relevant_ids):  # of what was retrieved, how much is relevant
        #find docs that are in BOTH lists (correct answers) n convert lists to sets for faster matching

        relevant_set = set(relevant_ids)
        retrieved_set = set(retrieved_ids)
        
        #count how many retrieved docs were actually relevant
        true_positives = len(retrieved_set.intersection(relevant_set))
        
        if len(retrieved_set) == 0:
            return 0.0
            
        #correct_found / total_found
        return true_positives / len(retrieved_set)

    def calculate_recall(self, retrieved_ids, relevant_ids):  # of what is relevant, how much was retrieved
        relevant_set = set(relevant_ids)
        retrieved_set = set(retrieved_ids)
        
        true_positives = len(retrieved_set.intersection(relevant_set))
        
        if len(relevant_set) == 0:
            return 0.0
            
        #correct_found / total_existing_answers
        return true_positives / len(relevant_set)

    def calculate_f1(self, precision, recall):
        if (precision + recall) == 0:
            return 0.0
            
        #2 * (P * R) / (P + R)
        return 2 * (precision * recall) / (precision + recall)

    def calculate_precision_at_k(self, retrieved_ids, relevant_ids, k):

        #slice the list to keep only top k
        top_k_retrieved = retrieved_ids[:k]
        
        #calc precision but for a list of #k now
        return self.calculate_precision(top_k_retrieved, relevant_ids)

    def plot_precision_recall_curve(self, retrieved_ids, relevant_ids, filename="precision_recall_curve.png"):
        precisions = []
        recalls = []
        
        #calculate P and R for every possible cut-off (k=1, k=2, ...)
        for k in range(1, len(retrieved_ids) + 1):
            #get top k
            current_k_items = retrieved_ids[:k]
            
            p = self.calculate_precision(current_k_items, relevant_ids)
            r = self.calculate_recall(current_k_items, relevant_ids)
            
            precisions.append(p)
            recalls.append(r)
            
        #plot setup
        plt.figure(figsize=(8, 5))
        plt.plot(recalls, precisions, marker='.', label='Query')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
       
        plt.savefig(filename)
        plt.close()
        print(f"Graph successfully saved as '{filename}'")