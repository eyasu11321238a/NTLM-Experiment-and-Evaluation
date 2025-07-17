#term_frequency.py

from collections import defaultdict


def tokenize(text, tokenizer):
    """Tokenize text using provided tokenizer."""
    return tokenizer.tokenize(text)

def calculate_term_frequencies(doc_texts, tokenizer):
    """Calculate term frequencies for documents and collection."""
    term_frequencies = {}
    collection_frequencies = defaultdict(int)
    total_terms_in_collection = 0
    
    for doc_id, text in doc_texts.items():
        term_freq = defaultdict(int)
        document_terms = tokenize(text, tokenizer)
        total_terms_in_doc = len(document_terms)
        
        for term in document_terms:
            term_freq[term] += 1
            collection_frequencies[term] += 1
            total_terms_in_collection += 1
        
        term_frequencies[doc_id] = (term_freq, total_terms_in_doc)
    
    return term_frequencies, collection_frequencies, total_terms_in_collection

def p_u_given_d(term, doc_term_freq, total_terms_in_doc):
    """Calculate probability of term u given document d."""
    return doc_term_freq[term] / total_terms_in_doc if total_terms_in_doc > 0 else 0

def p_u_given_C(term, collection_frequencies, total_terms_in_collection):
    """Calculate probability of term u in collection C."""
    return collection_frequencies[term] / total_terms_in_collection if total_terms_in_collection > 0 else 0
