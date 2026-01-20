# gme_inference.py - GME wrapper that works with your ModelManager
import torch
from typing import List
import logging

logger = logging.getLogger(__name__)

class GMEModelWrapper:
    """Wrapper to use GME model loaded by ModelManager for forensic embeddings"""
    
    def __init__(self, model_manager, model_name: str, gpu_id: int):
        self.model_manager = model_manager
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.model = None
        self.tokenizer = None
        
    def _get_model_instance(self):
        """Get the model instance from ModelManager"""
        if not self.model:
            try:
                self.model = self.model_manager.get_model(self.model_name, self.gpu_id)
                logger.info(f"✅ GME model instance acquired: {self.model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to get GME model instance: {e}")
                return None
        return self.model
    
    def get_text_embeddings(
        self, 
        texts: List[str], 
        instruction: str = None,
        is_query: bool = True
    ) -> List[float]:
        """Get text embeddings using the loaded GME model"""
        try:
            model = self._get_model_instance()
            if not model:
                return []
            
            # For forensic analysis, we want high-quality embeddings
            if instruction and is_query:
                enhanced_texts = [f"{instruction} {text}" for text in texts]
            else:
                enhanced_texts = texts
            
            # Truncate for memory efficiency
            processed_texts = [text[:512] for text in enhanced_texts]
            
            # Generate embeddings using the loaded model
            # Note: This depends on the actual GME model API
            # You may need to adjust this based on how GME models work
            embeddings = []
            
            for text in processed_texts:
                # TODO: Implement native GME embedding generation (currently using transformer fallback)
                # The exact API depends on how the GME model was loaded
                try:
                    if hasattr(model, 'create_embedding'):
                       result = model.create_embedding(text)
                       if result and 'data' in result and len(result['data']) > 0:
                           embedding = result['data'][0]['embedding']
                           embeddings.append(embedding)
                           continue
                    # Assuming the model has an encode or get_embeddings method
                    if hasattr(model, 'encode'):
                        embedding = model.encode(text)
                    elif hasattr(model, 'get_text_embeddings'):
                        embedding = model.get_text_embeddings([text])
                        embedding = embedding[0] if len(embedding) > 0 else []
                    else:
                        # Fallback: try to use as transformer model
                        inputs = model.tokenizer(
                            text, 
                            return_tensors="pt", 
                            truncation=True, 
                            max_length=512,
                            padding=True
                        )
                        with torch.no_grad():
                            outputs = model(**inputs)
                            # Mean pooling of last hidden state
                            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                            embedding = embedding.cpu().numpy().tolist()
                    
                    embeddings.append(embedding)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for text: {e}")
                    continue
            
            # Return the first embedding (for single text input)
            return embeddings[0] if embeddings else []
            
        except Exception as e:
            logger.error(f"Error in GME text embeddings: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension by testing with sample text"""
        try:
            test_embedding = self.get_text_embeddings(["test"])
            return len(test_embedding) if test_embedding else 0
        except:
            return 0