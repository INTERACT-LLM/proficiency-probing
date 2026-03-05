"""
Class for caching embeddings to avoid redundant computations.
"""
from typing import Optional, Union, List
import os
import numpy as np
from .embedder import TextEmbedder


class EmbeddingCache:
    def __init__(
        self,
        embedder: Optional[Union[TextEmbedder]] = None,
        texts: Optional[list] = None,
        cache_path: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
        show_progress: bool = True,
        layer_indices: Optional[Union[int, List[int]]] = None  # NEW
    ):
        self.embedder = embedder
        self.text_input = texts
        self.cached_path = cache_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.show_progress = show_progress
        self.layer_indices = layer_indices  # NEW

        if (embedder is None or texts is None) and cache_path is None:
            raise ValueError("Must provide either an embedder and texts or a cache path to load from.")

        if embedder is not None and texts is not None:
            self.cached_texts = texts
            self.cached_model_name = embedder.model_name
            self.cached_layer_index = embedder.layer_index if hasattr(embedder, 'layer_index') else None
            self.cached_pooling = embedder.pooling
            self.cached_embeddings = None

        if cache_path is not None:
            if os.path.exists(self.cached_path):
                print("Cache path provided and file exists. Loading embeddings from disk.")
                self.load_from_disk(self.cached_path)

    def get_embeddings(self) -> Union[np.ndarray, dict]:
        if self.cached_embeddings is not None:
            print("Using cached embeddings from memory.")
            return self.cached_embeddings
        elif self.cached_path is not None and os.path.exists(self.cached_path):
            print("Loading embeddings from disk cache.")
            self.load_from_disk(self.cached_path)
            return self.cached_embeddings
        else:
            print("No cache found. Computing embeddings.")
            if self.embedder is None or self.text_input is None:
                raise ValueError("Cannot compute embeddings without an embedder and text input.")
            self.cached_embeddings = self.embedder.embed_texts(
                self.text_input,
                batch_size=self.batch_size,
                max_length=self.max_length,
                show_progress=self.show_progress,
                layer_indices=self.layer_indices  # NEW
            )
            if self.cached_path is not None:
                dir_path = os.path.dirname(self.cached_path)
                
                if not os.path.exists(dir_path):
                    print(f"Cache path {dir_path} does not exist. Creating directories.")
                    os.makedirs(dir_path, exist_ok=True)
                print("Saving embeddings to disk cache.")
                self.save_to_disk(self.cached_path)
            return self.cached_embeddings

    def save_to_disk(self, path):
        base_meta = dict(
            texts=self.text_input,
            model_name=self.embedder.model_name,
            layer_index=self.embedder.layer_index,
            pooling=self.embedder.pooling,
            batch_size=self.batch_size,
            max_length=self.max_length,
            show_progress=self.show_progress,
        )
        if isinstance(self.cached_embeddings, dict):
            layer_data = {f"layer_idx_{k}": v for k, v in self.cached_embeddings.items()}
            np.savez_compressed(path, is_multi_layer=True, **layer_data, **base_meta)
        else:
            np.savez_compressed(path, is_multi_layer=False, embeddings=self.cached_embeddings, **base_meta)
        print(f"Successfully saved embeddings to: {path}")

    def load_from_disk(self, path):
        data = np.load(path, allow_pickle=True)
        self.cached_texts = list(data['texts'])
        self.cached_model_name = str(data['model_name'])
        self.cached_layer_index = int(data['layer_index'])
        self.cached_pooling = str(data['pooling'])
        self.batch_size = int(data['batch_size'])
        self.max_length = int(data['max_length'])
        self.show_progress = bool(data['show_progress'])

        if bool(data['is_multi_layer']):
            self.cached_embeddings = {
                int(k.replace("layer_idx_", "")): data[k]
                for k in data.files if k.startswith("layer_idx_")
            }
        else:
            self.cached_embeddings = data['embeddings']

        print(f"Loaded embeddings from {path}:")
        print(f"  Model: {self.cached_model_name}")
        if self.layer_indices is not None:
            print(f"  Requested layers: {self.layer_indices}")
        else:
            print(f"  Layer: {self.cached_layer_index}")
        print(f"  Pooling: {self.cached_pooling}")
        print(f"  Number of embeddings: {len(self.cached_texts)}")
        self._validate_cache_compatibility()

    def _validate_cache_compatibility(self):
        warnings = []

        if self.embedder is not None:
            if self.cached_model_name != self.embedder.model_name:
                warnings.append(f"Model mismatch: cached={self.cached_model_name}, current={self.embedder.model_name}")
            if self.cached_pooling != self.embedder.pooling:
                warnings.append(f"Pooling method mismatch: cached={self.cached_pooling}, current={self.embedder.pooling}")

            # NEW: validate layers depending on single vs multi
            if isinstance(self.cached_embeddings, dict):
                cached_layers = set(self.cached_embeddings.keys())
                requested = set(self.layer_indices if isinstance(self.layer_indices, list) else [self.layer_indices])
                if cached_layers != requested:
                    warnings.append(f"Layer mismatch: cached={cached_layers}, requested={requested}")
            else:
                if self.cached_layer_index != self.embedder.layer_index:
                    warnings.append(f"Layer index mismatch: cached={self.cached_layer_index}, current={self.embedder.layer_index}")

        if self.text_input is None and self.cached_texts is None:
            warnings.append("No texts provided in either cache or current input.")
        elif self.cached_embeddings is None:
            warnings.append("No cached embeddings found, cannot validate texts.")

        if self.cached_texts is not None and self.text_input is not None and self.cached_texts != self.text_input:
            if set(self.cached_texts) == set(self.text_input):
                warnings.append("Text order mismatch: cached texts contain the same texts but in a different order.")
            else:
                warnings.append("Text mismatch: cached texts do not match current texts.")

        if warnings:
            import warnings as warn_module
            for w in warnings:
                warn_module.warn(w, UserWarning)