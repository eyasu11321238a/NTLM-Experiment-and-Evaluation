import torch
import numpy as np
from transformers import BertModel, BertTokenizer  
#from transformers import GPT2Tokenizer, GPT2Model #... Note: Don't import gpt2 and BERT models at the same time
#from transformers import RobertaTokenizer, RobertaModel


class BertEmbeddingManager:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        
        # Extract the input embeddings (token embeddings)
        self.word_embeddings = self.model.embeddings.word_embeddings
        self.vocab = self.tokenizer.get_vocab()
        
        # Convert embeddings to numpy
        with torch.no_grad():
            self.embedding_matrix = self.word_embeddings.weight.numpy()
        
        self.embedding_dict = {
            token: self.embedding_matrix[idx].astype(np.float32)
            for token, idx in self.vocab.items()
            if not token.startswith('[') and not token.startswith('##') 
        }

    def get_embedding(self, token):
        """Get embedding for a single token."""
        return self.embedding_dict.get(token)

    def __getitem__(self, token):
        return self.get_embedding(token)

    def __contains__(self, token):
        return token in self.embedding_dict






class BertEmbeddingManager1:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

        # Extract the input embeddings (token embeddings)
        self.word_embeddings = self.model.embeddings.word_embeddings
        self.vocab = self.tokenizer.get_vocab()

        # Extract the positional embeddings
        self.position_embeddings = self.model.embeddings.position_embeddings

        # Convert token embeddings to numpy
        with torch.no_grad():
            self.embedding_matrix = self.word_embeddings.weight.numpy()

        # Create a dictionary of token embeddings
        self.embedding_dict = {
            token: self.embedding_matrix[idx].astype(np.float32)
            for token, idx in self.vocab.items()
            if not token.startswith('[') and not token.startswith('##')
        }

    def get_contextualized_embedding(self, tokens):
        """
        Get the contextualized embedding for a sequence of tokens.
        """
        input_ids = self.tokenizer.encode(tokens, return_tensors='pt')
        with torch.no_grad():
            output = self.model(input_ids)[0]  # Get the output embeddings
        
        # Combine token embeddings and positional embeddings
        token_embeddings = output[:, :input_ids.size(-1), :]
        position_ids = torch.arange(input_ids.size(-1), device=input_ids.device)
        position_embeddings = self.position_embeddings(position_ids)
        contextualized_embeddings = token_embeddings + position_embeddings

        return contextualized_embeddings.numpy()

    def get_embedding(self, token):
        """Get embedding for a single token."""
        return self.embedding_dict.get(token)

    def __getitem__(self, token):
        return self.get_embedding(token)

    def __contains__(self, token):
        return token in self.embedding_dict








class monoBertEmbeddingManager:
    def __init__(self, model_name='castorini/monobert-large-msmarco'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        
        # Extract the input embeddings (token embeddings)
        self.word_embeddings = self.model.embeddings.word_embeddings
        self.vocab = self.tokenizer.get_vocab()
        
        # Convert embeddings to numpy
        with torch.no_grad():
            self.embedding_matrix = self.word_embeddings.weight.numpy()
        
        self.embedding_dict = {
            token: self.embedding_matrix[idx].astype(np.float32)
            for token, idx in self.vocab.items()
            if not token.startswith('[') and not token.startswith('##') 
        }

    def get_embedding(self, token):
        """Get embedding for a single token."""
        return self.embedding_dict.get(token)

    def __getitem__(self, token):
        return self.get_embedding(token)

    def __contains__(self, token):
        return token in self.embedding_dict



class GptEmbeddingManager:
    def __init__(self, model_name='gpt2'):
        # Load GPT-2 tokenizer and model from Hugging Face
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        
        # Extract the token embeddings (word embeddings)
        self.word_embeddings = self.model.get_input_embeddings()
        
        # Create vocabulary mapping
        self.vocab = self.tokenizer.get_vocab()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Convert embeddings to numpy for easier manipulation
        with torch.no_grad():
            self.embedding_matrix = self.word_embeddings.weight.cpu().numpy()
            
        # Create a dictionary mapping tokens to their embeddings
        self.embedding_dict = {
            token: self.embedding_matrix[idx].astype(np.float32)
            for token, idx in self.vocab.items()
        }

    def get_embedding(self, token):
        """Get embedding for a single token."""
        if token in self.embedding_dict:
            return self.embedding_dict[token]
        return None

    def __getitem__(self, token):
        """Allow dictionary-like access to embeddings."""
        return self.get_embedding(token)
    
    def __contains__(self, token):
        """Check if token exists in embedding dictionary."""
        return token in self.embedding_dict



class RobertaEmbeddingManager:
    def __init__(self, model_name='roberta-base'):
        # Load Roberta tokenizer and model from Hugging Face
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        
        # Extract the token embeddings (word embeddings)
        self.word_embeddings = self.model.embeddings.word_embeddings
        
        # Create vocabulary mapping
        self.vocab = self.tokenizer.get_vocab()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Convert embeddings to numpy for easier manipulation
        with torch.no_grad():
            self.embedding_matrix = self.word_embeddings.weight.numpy()
            
        # Create a dictionary mapping tokens to their embeddings
        self.embedding_dict = {
            token: self.embedding_matrix[idx].astype(np.float32)
            for token, idx in self.vocab.items()
            if not token.startswith('Ġ')  # Roberta uses 'Ġ' to signify spaces in tokens
        }

    def get_embedding(self, token):
        """Get embedding for a single token"""
        if token in self.embedding_dict:
            return self.embedding_dict[token]
        return None
    
    def __getitem__(self, token):
        """Allow dictionary-like access to embeddings"""
        return self.get_embedding(token)
    
    def __contains__(self, token):
        """Check if token exists in embedding dictionary"""
        return token in self.embedding_dict
