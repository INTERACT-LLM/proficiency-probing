"""
Text Embedder Module: Extracts embeddings from transformer models at various layers.
"""
import torch
from typing import Optional, Union, List, Dict, Tuple
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np


class TextEmbedder:
    """
    A text embedder that extracts representations from a specified layer of a transformer model.    
    
    Attributes:
        model_name: Name or path of the HuggingFace model
        device: Device to run the model on ('cuda' or 'cpu')
        layer_index: Which layer to extract embeddings from (default: last layer, -1)
        pooling: How to pool token embeddings ('mean', 'cls', 'max')
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
        layer_index: int = -1,
        pooling: str = "mean"
    ):
        """
        Initialize the TextEmbedder.
        
        Args:
            model_name: Name or path of the HuggingFace model
            device: Device to run on. If None, automatically detects CUDA availability
            layer_index: Which layer to extract from (-1 for last layer)
            pooling: Pooling strategy - 'mean', 'cls', or 'max'
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_name = model_name
        self.layer_index = layer_index
        self.pooling = pooling
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        ).to(self.device)
        self.model.eval() # Set model to evaluation mode (no LLM training is done in this repo!!)

        
    def _pool_embeddings(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool token embeddings into a single vector per sequence.
        
        Args:
            hidden_states: Token embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Pooled embeddings [batch_size, hidden_dim]
        """
        if self.pooling == "cls":
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]
        elif self.pooling == "mean":
            # Mean pooling (accounting for attention mask)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling == "max":
            # Max pooling (accounting for attention mask)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states.clone()
            hidden_states[mask_expanded == 0] = -1e9
            return torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
  
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            show_progress: Whether to show progress bar            
        Returns:
            Array of embeddings [num_texts, embedding_dim]
        """
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size) # Create batches of indices
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding texts")
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get model outputs
                outputs = self.model(**encoded)
                
                # Extract from hidden states
                hidden_states = outputs.hidden_states[self.layer_index]
                embeddings = self._pool_embeddings(hidden_states, encoded["attention_mask"])
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def get_embedding_dim(self, sample_text: str = "sample") -> int:
        """
        Get the dimensionality of embeddings produced by this embedder.
        
        Args:
            sample_text: A sample text to determine embedding dimension
            
        Returns:
            Embedding dimension
        """
        sample_embedding = self.embed_texts([sample_text], show_progress=False)
        return sample_embedding.shape[1]
