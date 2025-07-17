import numpy as np
from term_frequency import p_u_given_d, p_u_given_C
import random
# Compute cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Improved translation probability calculation
def compute_translation_probability(target_word, candidate_term, word_embeddings):
    if target_word in word_embeddings and candidate_term in word_embeddings:
        target_vector = word_embeddings[target_word]
        candidate_vector = word_embeddings[candidate_term]
        similarity = cosine_similarity(target_vector, candidate_vector)
        # Apply softmax to get a probability distribution
        return np.exp(similarity) / np.sum(np.exp(similarity))
    else:
        return 0.0
# Compute log-likelihood ratio
def compute_log_likelihood_ratio(translation_prob, p_qi_C, alpha, n):
    return (np.log(translation_prob / (p_qi_C)) + n * np.log(alpha)) ##############(p_qi_C * alpha)

# Score documents based on translation probabilities
def score_document(query, document, word_embeddings, doc_term_freq, total_terms_in_doc, collection_frequencies, total_terms_in_collection, alpha):
    score = 0.0
    n = len(query)

    for query_term in query:
        if query_term not in word_embeddings:
            continue  # Skip query terms not in word embeddings

        translation_prob_sum = 0.0

        for doc_term in document:
            if doc_term not in word_embeddings:
                continue  # Skip document terms not in word embeddings

            p_qi_d = p_u_given_d(query_term, doc_term_freq, total_terms_in_doc)
            p_qi_C = p_u_given_C(query_term, collection_frequencies, total_terms_in_collection)
            translation_prob = compute_translation_probability(query_term, doc_term, word_embeddings)
            translation_prob_sum += (translation_prob * p_qi_d)  # ......Eq(3)
        
        if translation_prob_sum > 0:
            log_likelihood_ratio = compute_log_likelihood_ratio(translation_prob_sum, p_qi_C, alpha, n) # .....Eq(1)
            score += log_likelihood_ratio

    return score
