import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import torch
import asyncio
import concurrent.futures
from dataclasses import dataclass

from transformers import CLIPProcessor, CLIPModel


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings generation."""
    clip_model_name: str = "openai/clip-vit-base-patch32"
    batch_size: int = 8
    max_workers: int = 4
    device: str = "auto"  # "auto", "cpu", "cuda"
    enable_ocr: bool = False
    normalize_embeddings: bool = True


class EmbeddingsGenerator:
    """Generate CLIP embeddings and extract text from video frames."""
    
    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize the embeddings generator.
        
        Args:
            config: Configuration object for embedding parameters
        """
        self.config = config or EmbeddingConfig()
        self.device = self._get_device()
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(self.config.clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)
        
        # Move model to device
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
    
    def _get_device(self) -> str:
        """Determine the best device to use."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return self.config.device
    

    async def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate CLIP embedding for text.
        
        Args:
            text: Input text string
            
        Returns:
            CLIP text embedding as numpy array
        """
        try:
            # Run the heavy computation in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _generate_embedding():
                inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**inputs)
                    
                    if self.config.normalize_embeddings:
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    embedding = text_features.cpu().numpy()[0]
                
                return embedding
            
            # Run in thread pool to avoid blocking the event loop
            embedding = await loop.run_in_executor(None, _generate_embedding)
            return embedding
            
        except Exception as e:
            return np.zeros(512, dtype=np.float32)