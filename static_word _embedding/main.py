import os
import re
import string
import math
import nltk
from typing import List, Tuple, Dict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec, KeyedVectors

nltk.download('punkt')
nltk.download('stopwords')

class WordTranslator:
    def __init__(self, word_embeddings_path: str):
        """
        Initialize the word translator with pre-trained word embeddings
        """
        self.word_embeddings = KeyedVectors.load_word2vec_format(
            word_embeddings_path, binary=True
        )
        self.vocabulary = set(self.word_embeddings.index_to_key)
        
    def get_nearest_neighbors(self, word: str, k: int = 8) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors for a given word based on cosine similarity
        """
        if word not in self.vocabulary:
            return []
        
        return self.word_embeddings.most_similar(word, topn=k)
    
    def calculate_translation_probability(self, source_word: str, target_word: str,
                                          temperature: float = 0.1) -> float:
        """
        Calculate translation probability pt(w|u) using NTLM approach
        """
        if source_word not in self.vocabulary or target_word not in self.vocabulary:
            return 0.0
        
        similarity = self.word_embeddings.similarity(source_word, target_word)
        probability = math.exp(similarity / temperature)
        
        nearest_neighbors = self.get_nearest_neighbors(source_word, k=8)
        normalization_term = sum(math.exp(sim / temperature) for _, sim in nearest_neighbors)
        
        return probability / normalization_term if normalization_term > 0 else 0.0
    
    def get_translation_candidates(self, source_word: str, 
                                   threshold: float = 0.01) -> List[Tuple[str, float]]:
        """
        Get all possible translation candidates with their probabilities
        """
        if source_word not in self.vocabulary:
            return []
        
        candidates = self.get_nearest_neighbors(source_word, k=8)
        translation_probs = []
        for candidate, _ in candidates:
            prob = self.calculate_translation_probability(source_word, candidate)
            if prob >= threshold:
                translation_probs.append((candidate, prob))
        
        return sorted(translation_probs, key=lambda x: x[1], reverse=True)

class SkipgramTrainer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> List[str]:
        text = re.sub(r'<[^>]+>', '', text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        return [token for token in tokens if token not in self.stop_words and token.isalpha()]

    def prepare_training_data(self, doc_texts: Dict[str, str]) -> List[List[str]]:
        return [self.preprocess_text(text) for text in doc_texts.values() if text]

    def train_skipgram_model(self, training_data: List[List[str]], vector_size=300, 
                             window=5, min_count=5, workers=4, epochs=50) -> Word2Vec:
        model = Word2Vec(sentences=training_data, vector_size=vector_size, window=window,
                         min_count=min_count, workers=workers, sg=1, epochs=epochs)
        return model

    def save_model(self, model: Word2Vec, save_path: str):
        model.wv.save_word2vec_format(save_path, binary=True)

class CBOWTrainer(SkipgramTrainer):
    def train_cbow_model(self, training_data: List[List[str]], vector_size=300, window=5, 
                         min_count=5, workers=4, epochs=30, negative=15, 
                         alpha=0.025, min_alpha=0.0001) -> Word2Vec:
        alpha_delta = (alpha - min_alpha) / epochs
        model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, 
                         workers=workers, sg=0, negative=negative, alpha=alpha, 
                         min_alpha=min_alpha, compute_loss=True)
        
        model.build_vocab(training_data)
        
        total_examples = len(training_data)
        for epoch in range(epochs):
            current_alpha = alpha - (alpha_delta * epoch)
            model.alpha = current_alpha
            model.min_alpha = current_alpha
            model.train(training_data, total_examples=total_examples, epochs=1, compute_loss=True)
        
        return model

# AP Data
# Function to parse the TREC file
def AP_parse_trec_file(trec_file_path):
    doc_texts = {}
    current_doc_id = None
    current_text = []
    
    encodings = ['utf-8', 'latin-1', 'ISO-8859-1']
    for encoding in encodings:
        try:
            with open(trec_file_path, 'r', encoding=encoding, errors='ignore') as file:
                for line in file:
                    if line.startswith('<DOCNO>'):
                        current_doc_id = line.strip().replace('<DOCNO>', '').replace('</DOCNO>', '').strip()
                    elif line.startswith('</TEXT>'):
                        if current_doc_id:
                            doc_texts[current_doc_id] = ' '.join(current_text)
                            current_doc_id = None
                            current_text = []
                    elif current_doc_id:
                        if not (line.startswith('<DOC>') or line.startswith('</DOC>') or line.startswith('<FILEID>') or
                                line.startswith('<FIRST>') or line.startswith('<SECOND>') or line.startswith('<HEAD>') or
                                line.startswith('</BYLINE>') or
                                line.startswith('<DATELINE>') or line.startswith('<TEXT>')):
                            current_text.append(line.strip())
            break
        except UnicodeDecodeError:
            continue  

    return doc_texts



def main():
    # Initialize trainer (choose SkipgramTrainer or CBOWTrainer)
    trainer = SkipgramTrainer()
    
    # Parse TREC file (assuming the function is defined and data path is set)
    trec_file_path = os.path.join("..", "Data", "AP_Doc", "ap", "concatenated", "concatenated_documents.txt")
    doc_texts = AP_parse_trec_file(trec_file_path)  # or WSJ_parse_trec_file(trec_file_path)
    
    # Prepare training data
    print("Preparing training data...")
    training_data = trainer.prepare_training_data(doc_texts)
    
    # Train model (choose Skipgram or CBOW model training method)
    print("Training Skipgram model...")
    model = trainer.train_skipgram_model(training_data)
    
    # Save model
    save_path = "path/to/save/skipgram_model.bin"
    trainer.save_model(model, save_path)
    
    # Test the model with WordTranslator
    print("\nTesting translation capabilities...")
    translator = WordTranslator(save_path)
    
    test_words = ["oil", "gas", "energy", "company"]
    for word in test_words:
        print(f"\nTranslation candidates for '{word}':")
        translations = translator.get_translation_candidates(word)
        for target_word, prob in translations:
            print(f"{target_word}: {prob:.4f}")

if __name__ == "__main__":
    main()
