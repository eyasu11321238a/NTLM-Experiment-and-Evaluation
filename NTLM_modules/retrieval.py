# retrieval.py

import pandas as pd
from tqdm import tqdm
import pyterrier as pt
from term_frequency import calculate_term_frequencies, tokenize
from scoring import score_document

def create_dirichlet_pipeline(index, mu=350, num_results=100, metadata=['text']): #metadata can be ['body'] or ['text']
    """
    Initializes a Dirichlet language model retrieval pipeline with customizable parameters.
    
    Parameters:
    - index: PyTerrier Index object, the preloaded index.
    - mu: int, the smoothing parameter for the Dirichlet language model.
    - num_results: int, the number of results to retrieve for each query.
    - metadata: list of str, the list of metadata fields to retrieve from the index (default is ['body'] or ['text']).
    
    Returns:
    - dirichlet_pipeline: PyTerrier pipeline object, ready to transform topics for retrieval.
    """
    # Create a text loader with specified metadata fields
    text_loader = pt.text.get_text(index, metadata=metadata)
    
    # Set up the retrieval pipeline with the DirichletLM model
    dirichlet_pipeline = (
        pt.BatchRetrieve(index, wmodel="DirichletLM", controls={'mu': mu}, 
                        num_results=num_results, verbose=True) 
        >> text_loader
    )
    
    return dirichlet_pipeline

def retrieve_and_rank_documents(topics, embeddings, index, alpha, dirichlet_weight, ntlm_weight):
    """
    Retrieve and rank documents using neural translation language model and Dirichlet model scores.
    
    Parameters:
    - topics: DataFrame containing query topics
    - bert_embeddings: BertEmbeddingManager instance
    - index: PyTerrier index
    - alpha: float, smoothing parameter
    - dirichlet_weight: float, weight for Dirichlet scores
    - ntlm_weight: float, weight for NTLM scores
    
    Returns:
    - DataFrame with ranked results
    """
    results = []
    # Create pipeline with default parameters
    dirichlet_pipeline = create_dirichlet_pipeline(index)
    dirichlet_results = dirichlet_pipeline.transform(topics)
    
    # Create document text dictionary
    doc_texts = {row['docno']: row['text'] for _, row in dirichlet_results.iterrows()}
    
    # Calculate term frequencies
    term_frequencies, collection_frequencies, total_terms_in_collection = calculate_term_frequencies(
        doc_texts, embeddings.tokenizer)

    for idx, row in tqdm(topics.iterrows(), total=len(topics), desc="Processing Topics"):
        topic_id = row['qid']
        query_tokens = tokenize(row['query'], embeddings.tokenizer)
        scores = []
        
        retrieved_docs = dirichlet_results.loc[dirichlet_results["qid"] == topic_id][['docno', 'score']].values

        for doc_id, dirichlet_score in retrieved_docs:
            doc_text = doc_texts[doc_id]
            doc_tokens = tokenize(doc_text, embeddings.tokenizer)
            doc_term_freq, total_terms_in_doc = term_frequencies[doc_id]
            
            ntlm_score = score_document(
                query_tokens, 
                doc_tokens, 
                embeddings, 
                doc_term_freq, 
                total_terms_in_doc, 
                collection_frequencies, 
                total_terms_in_collection, 
                alpha
            )
            scores.append((doc_id, ntlm_score, dirichlet_score))
        
        # Combine scores using weights
        combined_scores = [
            (doc_id, ntlm_weight * ntlm_score + dirichlet_weight * dirichlet_score)
            for doc_id, ntlm_score, dirichlet_score in scores
        ]
        
        # Sort by combined score
        ranked_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
        
        # Create results entries
        for rank, (doc_id, score) in enumerate(ranked_scores, 1):
            results.append((topic_id, doc_id, rank, score))
    
    return pd.DataFrame(results, columns=['qid', 'docno', 'rank', 'score'])





