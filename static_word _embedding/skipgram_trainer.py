from typing import List, Dict
import re, string, nltk
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

class SkipgramTrainer:
    def __init__(self):
        """Initialize the trainer with necessary NLTK resources"""
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by removing special characters, converting to lowercase,
        removing stopwords and tokenizing
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens 
                 if token not in self.stop_words 
                 and token.isalpha()]
        
        return tokens

    def prepare_training_data(self, doc_texts: Dict[str, str]) -> List[List[str]]:
        """Convert document dictionary into format suitable for Word2Vec training"""
        training_data = []
        
        for doc_id, text in doc_texts.items():
            tokens = self.preprocess_text(text)
            if tokens:  # Only add if document contains valid tokens
                training_data.append(tokens)
                
        return training_data

    def train_skipgram_model(self, 
                            training_data: List[List[str]], 
                            vector_size: int = 300,
                            window: int = 5,
                            min_count: int = 5,
                            workers: int = 4,
                            epochs: int = 50) -> Word2Vec:
        """
        Train Skip-gram model using preprocessed data
        
        Args:
            training_data: List of tokenized documents
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Minimum frequency of words to consider
            workers: Number of CPU cores to use
            epochs: Number of training epochs
        """
        model = Word2Vec(sentences=training_data,
                        vector_size=vector_size,
                        window=window,
                        min_count=min_count,
                        workers=workers,
                        sg=1,  # Skip-gram model (sg=1)
                        epochs=epochs)
        
        return model

    def save_model(self, model: Word2Vec, save_path: str):
        """Save the trained model in word2vec binary format"""
        model.wv.save_word2vec_format(save_path, binary=True)
