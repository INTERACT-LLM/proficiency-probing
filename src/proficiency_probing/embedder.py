"""
Text Embedder Module: Extracts embeddings from transformer models at various layers/heads.
"""

import torch
from typing import Optional, Union, List, Dict, Tuple
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np


class TextEmbedder:
    """
    A flexible text embedder that can extract representations from any layer or attention head
    of a transformer model.
    
    Attributes:
        model_name: Name or path of the HuggingFace model
        device: Device to run the model on ('cuda' or 'cpu')
        layer_index: Which layer to extract embeddings from (default: last layer, -1)
        head_index: Which attention head to extract from (None means use all heads/full representation)
        pooling: How to pool token embeddings ('mean', 'cls', 'max')
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
        layer_index: int = -1,
        head_index: Optional[int] = None,
        pooling: str = "mean"
    ):
        """
        Initialize the TextEmbedder.
        
        Args:
            model_name: Name or path of the HuggingFace model
            device: Device to run on. If None, automatically detects CUDA availability
            layer_index: Which layer to extract from (-1 for last layer)
            head_index: Which attention head to use (None for full representation)
            pooling: Pooling strategy - 'mean', 'cls', or 'max'
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_name = model_name
        self.layer_index = layer_index
        self.head_index = head_index
        self.pooling = pooling
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            output_attentions=(head_index is not None)
        ).to(self.device)
        self.model.eval()
        
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
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling == "max":
            # Max pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states.clone()
            hidden_states[mask_expanded == 0] = -1e9
            return torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    def _extract_attention_head_embeddings(
        self,
        attentions: Tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract embeddings from a specific attention head.
        
        Args:
            attentions: Tuple of attention tensors from all layers
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Head embeddings [batch_size, seq_len * seq_len] (flattened attention patterns)
        """
        # Get attention from specified layer
        layer_attention = attentions[self.layer_index]  # [batch, num_heads, seq_len, seq_len]
        
        # Extract specific head
        head_attention = layer_attention[:, self.head_index, :, :]  # [batch, seq_len, seq_len]
        
        # Flatten and apply mask
        batch_size = head_attention.size(0)
        flat_attention = head_attention.view(batch_size, -1)
        
        return flat_attention
    
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
        
        iterator = range(0, len(texts), batch_size)
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
                
                if self.head_index is not None:
                    # Extract from attention head
                    embeddings = self._extract_attention_head_embeddings(
                        outputs.attentions,
                        encoded["attention_mask"]
                    )
                else:
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
