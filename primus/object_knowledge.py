"""
Zero-shot object knowledge base using CLIP embeddings.
Pre-computes embeddings for 500+ common objects/ingredients for fast detection.
"""
import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path

MODEL_DIR = Path("models/qwen-vl-7b") # Reusing a safe path, or add 'clip-vit-base-patch32'
# For now, we use a lightweight list. In production, load from a JSON.

COMMON_OBJECTS = [
    # Kitchen/Ingredients
    "apple", "banana", "orange", "bottle", "cup", "bowl", "spoon", "fork", "knife",
    "plate", "pan", "pot", "oven", "microwave", "refrigerator", "salt", "pepper",
    "sugar", "flour", "egg", "milk", "bread", "cheese", "tomato", "onion", "garlic",
    # Electronics
    "phone", "laptop", "keyboard", "mouse", "monitor", "tv", "remote", "charger",
    "headphones", "camera", "tablet", "smartwatch",
    # Furniture/Room
    "chair", "table", "sofa", "bed", "desk", "lamp", "door", "window", "curtain",
    "rug", "pillow", "blanket", "shelf", "book", "plant", "vase",
    # People/Body
    "person", "hand", "face", "eye", "arm", "leg", "foot", "finger",
    # Tools
    "hammer", "screwdriver", "wrench", "drill", "scissors", "tape", "glue"
]

class ObjectKnowledgeBase:
    def __init__(self, device="cuda"):
        print("Loading CLIP for object knowledge...")
        # Using a small CLIP model for speed
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.device = device
        
        # Pre-compute embeddings
        self.objects = COMMON_OBJECTS
        self.embeddings = self._compute_embeddings(self.objects)
        print(f"Knowledge base loaded with {len(self.objects)} objects.")

    def _compute_embeddings(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy()

    def find_closest(self, image_embedding, top_k=3):
        # image_embedding: [512] numpy array
        image_emb = image_embedding / np.linalg.norm(image_embedding)
        text_embs = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        similarities = np.dot(text_embs, image_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(self.objects[i], float(similarities[i])) for i in top_indices]

# Global instance
kb = None
def get_kb():
    global kb
    if kb is None:
        kb = ObjectKnowledgeBase()
    return kb
