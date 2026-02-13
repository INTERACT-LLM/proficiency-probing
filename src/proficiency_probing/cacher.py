"""
Class for caching embeddings to avoid redundant computations. This can be used to store embeddings in memory during a session, and optionally save/load them from disk for persistence across sessions.
Does not save TextEmbedder itself, to avoid caching the LLM model. Instead, it caches the resulting embeddings and the parameters used to generate them (e.g., model name, layer index, pooling method) to ensure compatibility when loading from disk.
"""
from typing import Optional
import os
import numpy as np
from .embedder import TextEmbedder

class EmbeddingCache:
    def __init__(self, embedder: Optional[TextEmbedder] = None, texts : Optional[list] = None, cache_path: Optional[str] = None, batch_size: int = 32, max_length: int = 512, show_progress: bool = True):
        self.embedder : Optional[TextEmbedder] = embedder
        self.text_input = texts
        self.cached_path : Optional[str] = cache_path # Optional path for disk caching
        self.batch_size = batch_size
        self.max_length = max_length
        self.show_progress = show_progress

        if (embedder is None or texts is None) and cache_path is None:
            raise ValueError("Must provide either an embedder and texts or a cache path to load from.")

        if embedder is not None and texts is not None:
            # Metadata for loading cache
            self.cached_texts : list = texts
            self.cached_model_name : str = embedder.model_name
            self.cached_layer_index : int = embedder.layer_index
            self.cached_pooling : str = embedder.pooling
            self.cached_embeddings : Optional[np.ndarray] = None  # In-memory cache

        if cache_path is not None:
            # Check if cache exists on disk and load it
            if os.path.exists(self.cached_path):
                print("Cache path provided and file exists. Loading embeddings from disk.")
                self.load_from_disk(self.cached_path)
            # else: save path for saving later

    def get_embeddings(self) -> np.ndarray:
        # Logic: Check memory → Check disk → Compute fresh
        if self.cached_embeddings is not None: # Always use cached embeddings if possible, even if cache_path is provided, to avoid redundant disk I/O.
            print("Using cached embeddings from memory.")
            return self.cached_embeddings
        elif (self.cached_path is not None) and os.path.exists(self.cached_path): # Check disk cache if path provided and in-memory cache is empty (alternative to running load_from_disk in __init__, allows for dynamic loading)
            print("Loading embeddings from disk cache.")
            self.load_from_disk(self.cached_path)
            return self.cached_embeddings
        else: # No cache available, compute embeddings
            print("No cache found. Computing embeddings.")
            if self.embedder is None or self.text_input is None:
                raise ValueError("Cannot compute embeddings without an embedder and text input.")
            self.cached_embeddings = self.embedder.embed_texts(self.text_input,
                                                               batch_size=self.batch_size,
                                                               max_length=self.max_length,
                                                               show_progress=self.show_progress)
            if self.cached_path is not None:
                if not os.path.exists(self.cached_path):
                    print(f"Cache path {self.cached_path} does not exist. Creating directories.")
                    os.makedirs(os.path.dirname(self.cached_path), exist_ok=True)
                print("Saving embeddings to disk cache.")
                self.save_to_disk(self.cached_path)
            return self.cached_embeddings
        
    
    
    def save_to_disk(self, path):
        """
        Saves the cached embeddings and associated metadata to disk in a compressed .npz format.
        This includes the embeddings themselves, the original texts, and the parameters used to generate the embeddings (model name, layer index, pooling method) to ensure compatibility when loading later.
        """
        np.savez_compressed(
            path,
            embeddings=self.cached_embeddings,
            texts=self.text_input,
            model_name=self.embedder.model_name,
            layer_index=self.embedder.layer_index,
            pooling=self.embedder.pooling,
            batch_size=self.batch_size,
            max_length=self.max_length,
            show_progress=self.show_progress
        )
        print(f"Succesfully saved embeddings to: {path}")
        
    
    def load_from_disk(self, path):
        """
        Loads cached embeddings and associated metadata from a compressed .npz file on disk.
        There is no need to load the TextEmbedder itself, as no new embeddings are needed.
        """
        data = np.load(path, allow_pickle=True)  # Need allow_pickle=True for non-arrays
        self.cached_embeddings : np.ndarray = data['embeddings']
        self.cached_texts = list(data['texts'])
        self.cached_model_name = str(data['model_name'])
        self.cached_layer_index = int(data['layer_index'])
        self.cached_pooling = str(data['pooling'])
        self.batch_size = int(data['batch_size'])
        self.max_length = int(data['max_length'])
        self.show_progress = bool(data['show_progress'])
        print(f"Loaded embeddings from {path}:")
        print(f"  Model: {self.cached_model_name}")
        print(f"  Layer: {self.cached_layer_index}")
        print(f"  Pooling: {self.cached_pooling}")
        print(f"  Number of embeddings: {len(self.cached_texts)}")
        self._validate_cache_compatibility()

            
    def _validate_cache_compatibility(self):
        """Check if loaded cache matches current embedder config and texts."""
        warnings = []
        
        # Check config compatibility
        if self.embedder is not None:
            if self.cached_model_name != self.embedder.model_name:
                warnings.append(f"Model mismatch: cached={self.cached_model_name}, current={self.embedder.model_name}")
            if self.cached_layer_index != self.embedder.layer_index:
                warnings.append(f"Layer index mismatch: cached={self.cached_layer_index}, current={self.embedder.layer_index}")
            if self.cached_pooling != self.embedder.pooling:
                warnings.append(f"Pooling method mismatch: cached={self.cached_pooling}, current={self.embedder.pooling}")
        

        # Check text matching
        if self.text_input is None and self.cached_texts is None:
            warnings.append("No texts provided in either cache or current input.")
        elif self.cached_embeddings is None:
            warnings.append("No cached embeddings found, cannot validate texts.")
        
        if self.cached_texts is not None and self.text_input is not None and (self.cached_texts != self.text_input):
            # Check if the two lists have the same texts but in different order (ignoring duplicates)
            if set(self.cached_texts) == set(self.text_input):
                warnings.append("Text order mismatch: cached texts contain the same texts but in a different order than current texts.")
            else:
                warnings.append("Text mismatch: cached texts do not match current texts.")
        
        if warnings:
            for w in warnings:
                import warnings as warn_module
                warn_module.warn(w, UserWarning)


