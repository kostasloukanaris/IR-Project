import os
import string
import pandas as pd


def clean_text(text):
    #lowercase
    text = text.lower()
    
    #remove punctuation
    for char in string.punctuation:
        text = text.replace(char, "")
    
    #tokenization split into tokens by whitespace
    words = text.split()
    
    #remove some common stopwords
    my_stopwords = {
        "the", "is", "at", "of", "on", "and", "a", "an", "in", "to", "for", 
        "that", "it", "with", "as", "are", "by", "be", "this", "or", "from"
    }
    
    #remove stopwords and apply simple stemming (remove "s")
    cleaned_tokens = []
    for w in words:
        if w not in my_stopwords:
            if w.endswith("s") and len(w) > 3:
                w = w[:-1] #take all but last character ("s")
            cleaned_tokens.append(w) # add cleaned word to list
            
    return cleaned_tokens


def load_documents(directory_path, total_files=1239):
    
    doc_ids = []
    contents = []
    
    print("Loading documents...")
    
    
    for i in range(1, total_files + 1):
        #make file ids with leading zeros to match filenames
        file_id_str = str(i).zfill(5)
        
        #construct full file path
        file_path = os.path.join(directory_path, file_id_str)
        
       #if no extension, add .txt
        if not os.path.exists(file_path):
             file_path += ".txt"

        #check existence (handle empty texts)
        if os.path.exists(file_path):
            try:
                #read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
                    
                    doc_ids.append(i) # Keep the ID as a number (to match Relevant.txt)
                    contents.append(raw_text)
            except Exception as e:
                print(f"Failed to read file {file_id_str}: {e}")
        else:
            #if file does not exist, skip and the ids will remain consistent
            pass

   
    df = pd.DataFrame({
        "doc_id": doc_ids,
        "raw_text": contents
    })
    return df



def build_inverted_index(df):
    index = {}
    
    print("Building inverted index...")
   
    for _, row in df.iterrows():
        d_id = row['doc_id']
        tokens = row['tokens'] 
        
       #calc term frequencies for the document
        tf_counts = {}
        for token in tokens:
            tf_counts[token] = tf_counts.get(token, 0) + 1
            
       #update inverted index
        for term, count in tf_counts.items():
            if term not in index:
                index[term] = []
            # Store: [Document ID, Term Frequency]
            index[term].append([d_id, count])
            
    return index