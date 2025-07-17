import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from typing import List, Tuple
import math

class GPT2WordTranslator:
    def __init__(self, gpt2_model_name: str = 'gpt2'):
        """
        Initialize the word translator with GPT-2 input embeddings
        """
        # Load GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.model = GPT2Model.from_pretrained(gpt2_model_name)
        
        # Extract the input embeddings (token embeddings) from GPT-2
        self.word_embeddings = self.model.wte  # GPT-2 uses wte for token embeddings
        
        # Create vocabulary mapping
        self.vocab = self.tokenizer.get_vocab()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Convert embeddings to numpy for easier manipulation
        with torch.no_grad():
            self.embedding_matrix = self.word_embeddings.weight.detach().cpu().numpy()
        
        # Store embedding dimension
        self.embedding_dim = self.embedding_matrix.shape[1]
        
    def get_word_embedding(self, word: str) -> np.ndarray:
        """
        Get the embedding vector for a given word.
        Note: GPT-2 tokenizer might split words differently, so we'll take the first token's embedding
        """
        # Add space before word to match GPT-2's tokenization
        word_with_space = f" {word}"
        tokens = self.tokenizer.encode(word_with_space, add_special_tokens=False)
        
        if not tokens:
            return None
        
        # Take the first token's embedding
        token_id = tokens[0]
        return self.embedding_matrix[token_id]
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        if vec1 is None or vec2 is None:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def get_nearest_neighbors(self, word: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors for a given word based on cosine similarity.
        """
        source_embedding = self.get_word_embedding(word)
        if source_embedding is None:
            return []
        
        similarities = []
        for idx, embedding in enumerate(self.embedding_matrix):
            # Convert token ID back to word, removing any special formatting
            target_word = self.reverse_vocab[idx].strip()
            # Skip empty strings and special tokens
            if not target_word or target_word.startswith('<') or target_word.startswith('Ġ'):
                continue
                
            similarity = self.cosine_similarity(source_embedding, embedding)
            similarities.append((target_word, similarity))
        
        # Sort by similarity and return top-k
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
    
    def calculate_translation_probability(self, source_word: str, target_word: str,
                                       temperature: float = 0.9) -> float:
        """
        Calculate translation probability pt(w|u) using NTLM approach.
        """
        source_embedding = self.get_word_embedding(source_word)
        target_embedding = self.get_word_embedding(target_word)
        
        if source_embedding is None or target_embedding is None:
            return 0.0
        
        # Cosine similarity between source and target word embeddings
        similarity = self.cosine_similarity(source_embedding, target_embedding)
        
        # Softmax-like probability with temperature scaling
        probability = math.exp(similarity / temperature)
        
        # Normalize over nearest neighbors
        nearest_neighbors = self.get_nearest_neighbors(source_word, k=10)
        normalization_term = sum(math.exp(sim / temperature) for _, sim in nearest_neighbors)
        
        return probability / normalization_term if normalization_term > 0 else 0.0
    
    def get_translation_candidates(self, source_word: str, threshold: float = 0.01) -> List[Tuple[str, float]]:
        """
        Get all possible translation candidates with their probabilities.
        """
        # Get nearest neighbors as potential translation candidates
        candidates = self.get_nearest_neighbors(source_word, k=10)
        
        # Calculate translation probabilities for each candidate
        translation_probs = []
        for candidate, _ in candidates:
            prob = self.calculate_translation_probability(source_word, candidate)
            if prob >= threshold:
                translation_probs.append((candidate, prob))
        
        # Sort by probability in descending order
        return sorted(translation_probs, key=lambda x: x[1], reverse=True)


def main():
    # Initialize translator with GPT-2 embeddings
    translator = GPT2WordTranslator('gpt2')
    
    # Example translations
    source_word = "gold"
    translations = translator.get_translation_candidates(source_word)
    
    print(f"Translation candidates for '{source_word}':")
    for word, prob in translations:
        print(f"{word}: {prob:.4f}")

if __name__ == "__main__":
    main()