
import gensim
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import math

class WordTranslator:
    def __init__(self, word_embeddings_path: str):
        """
        Initialize the word translator with pre-trained word embeddings
        """
        self.word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            word_embeddings_path, binary=True
        )
        self.vocabulary = set(self.word_embeddings.index_to_key)
        
    def get_nearest_neighbors(self, word: str, k: int = 8) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors for a given word based on cosine similarity
        """
        if word not in self.vocabulary:
            return []
        
        similar_words = self.word_embeddings.most_similar(word, topn=k)
        return similar_words
    
    def calculate_translation_probability(self, source_word: str, target_word: str,
                                       temperature: float = 0.9) -> float:
        """
        Calculate translation probability pt(w|u) using NTLM approach
        Args:
            source_word: The source word u
            target_word: The target word w
            temperature: Temperature parameter for controlling probability distribution
        """
        if source_word not in self.vocabulary or target_word not in self.vocabulary:
            return 0.0
        
        # Calculate cosine similarity
        similarity = self.word_embeddings.similarity(source_word, target_word)
        
        # Convert similarity to probability using softmax-like normalization
        # Higher temperature leads to more uniform probabilities
        probability = math.exp(similarity / temperature)
        
        # Get normalization term (sum of exp(sim(u,w')/temperature) for all w' in vocabulary)
        # Note: For efficiency, we only consider top-k nearest neighbors
        nearest_neighbors = self.get_nearest_neighbors(source_word, k=8)
        normalization_term = sum(math.exp(sim / temperature) 
                               for _, sim in nearest_neighbors)
        
        return probability / normalization_term if normalization_term > 0 else 0.0
    
    def get_translation_candidates(self, source_word: str, 
                                 threshold: float = 0.01) -> List[Tuple[str, float]]:
        """
        Get all possible translation candidates with their probabilities
        Args:
            source_word: The source word to translate
            threshold: Minimum probability threshold for considering a translation
        """
        if source_word not in self.vocabulary:
            return []
        
        # Get nearest neighbors as potential translation candidates
        candidates = self.get_nearest_neighbors(source_word, k=8)
        
        # Calculate translation probabilities for each candidate
        translation_probs = []
        for candidate, _ in candidates:
            prob = self.calculate_translation_probability(source_word, candidate)
            if prob >= threshold:
                translation_probs.append((candidate, prob))
        
        # Sort by probability in descending order
        return sorted(translation_probs, key=lambda x: x[1], reverse=True)

# Example usage
def main():
    # Initialize translator with your embeddings
    #word_embeddings_path = r"C:\Users\dolla\NTLM\GoogleNews-vectors-negative300.bin"
    #word_embeddings_path = r"C:\Users\dolla\Downloads\old and gas CBOW word embedding\model.bin"  #word2vec_model
    #word_embeddings_path = r"C:\Eyasu\Thesis_New\thesis_eyasu\Data\Word_Embedding\custom_skipgram_model.bin"
    
    word_embeddings_path = r"C:\Eyasu\Thesis_New\thesis_eyasu\Data\Word_Embedding\word2vec_models\AP_skipgram_model.bin" 
    #word_embeddings_path = r"C:\Eyasu\Thesis_New\thesis_eyasu\Data\Word_Embedding\word2vec_models\AP_cbow_model.bin"
    #word_embeddings_path = r"C:\Eyasu\Thesis_New\thesis_eyasu\Data\Word_Embedding\word2vec_models\WSJ_skipgram_model.bin"
    #word_embeddings_path = r"C:\Eyasu\Thesis_New\thesis_eyasu\Data\Word_Embedding\word2vec_models\WSJ_cbow_model.bin" 


    translator = WordTranslator(word_embeddings_path)
    
    # Example translations
    source_word = "visitor" #geology
    translations = translator.get_translation_candidates(source_word)
    
    print(f"Translation candidates for '{source_word}':")
    for word, prob in translations:
        print(f"{word}: {prob:.4f}")

if __name__ == "__main__":
    main()