from typing import List, Dict
import re, string, nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

class CBOWTrainer:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> List[str]:
        text = re.sub(r'<[^>]+>', '', text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = [token for token in word_tokenize(text) 
                  if token not in self.stop_words and token.isalpha()]
        return tokens

    def prepare_training_data(self, doc_texts: Dict[str, str]) -> List[List[str]]:
        return [self.preprocess_text(text) for text in doc_texts.values() if self.preprocess_text(text)]

    def train_cbow_model(self, training_data: List[List[str]], **kwargs) -> Word2Vec:
        return Word2Vec(sentences=training_data, sg=0, **kwargs)

    def save_model(self, model: Word2Vec, save_path: str):
        model.wv.save_word2vec_format(save_path, binary=True)
