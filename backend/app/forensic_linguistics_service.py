# forensic_linguistics_service.py
import asyncio
import json
import logging
import re
import time
import hashlib
import pickle
from .model_manager import ModelManager
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
# Core ML/NLP imports
import torch
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from collections import Counter, defaultdict
import statistics
from tqdm import tqdm # For progress bars in async tasks
# Web scraping and search
import httpx
from bs4 import BeautifulSoup
import tweepy
from urllib.parse import urlparse, urljoin

# Local imports
from .web_search_service import WebSearchService

logger = logging.getLogger(__name__)
TARGET_EMBEDDING_DIM = 4096 
# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    logger.warning("Could not download NLTK data")

try:
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

@dataclass
class StyleVector:
    """Comprehensive stylometric feature vector - OVERHAULED for topic-independent authorship"""
    
    # === CORE STYLOMETRIC FEATURES (Topic-Independent) ===
    
    # Character-level (Most reliable for cross-topic)
    char_trigrams: Dict[str, float]  # Character 3-grams (e.g., " th", "the", "he ")
    char_bigrams: Dict[str, float]   # Character 2-grams
    
    # Function words (Subconscious patterns)
    function_word_frequencies: Dict[str, float]  # Per 1000 words
    function_word_bigrams: Dict[str, float]      # "of the", "in a", etc.
    
    # Lexical complexity
    avg_word_length: float
    avg_sentence_length: float
    sentence_length_variance: float  # Consistency measure
    vocab_richness: float  # Type-token ratio
    hapax_legomena_ratio: float  # Words appearing once
    yule_k: float  # Vocabulary diversity
    word_length_distribution: Dict[int, float]  # Distribution of word lengths
    
    # Syntactic patterns (Structure over content)
    pos_distribution: Dict[str, float]
    pos_bigrams: Dict[str, float]    # POS 2-grams (e.g., "DT-NN")
    pos_trigrams: Dict[str, float]   # POS 3-grams (e.g., "DT-NN-VB")
    dependency_patterns: Dict[str, float]
    sentence_complexity: float
    clause_patterns: Dict[str, float]  # Subordinate vs coordinate
    
    # Punctuation and formatting
    punctuation_ratios: Dict[str, float]
    punctuation_sequences: Dict[str, float]  # "..", "!?", etc.
    capitalization_ratio: float
    all_caps_ratio: float
    contraction_ratio: float  # "don't" vs "do not"
    avg_paragraph_length: float
    
    # Stylistic markers (Unconscious habits)
    modal_verb_usage: float
    passive_voice_ratio: float
    question_ratio: float
    exclamation_ratio: float
    sentence_starter_patterns: Dict[str, float]  # How sentences begin
    
    # === OPTIONAL/MINIMAL SEMANTIC FEATURES ===
    semantic_embeddings: List[float]  # ONLY for topic mismatch detection
    semantic_embedding_dim: int
    
    # Metadata
    extracted_date: Optional[str] = None
    platform: Optional[str] = None

@dataclass
class TextDocument:
    """Represents a document in our corpus"""
    content: str
    source_url: str
    date: Optional[datetime]
    platform: str
    author: str
    title: Optional[str] = None
    metadata: Dict = None

@dataclass
class SimilarityScore:
    """Similarity analysis result - OVERHAULED scoring"""
    overall_score: float
    
    # Core stylometric scores (topic-independent)
    function_word_score: float    # 35% weight - most reliable
    syntactic_score: float         # 30% weight - structure patterns
    character_ngram_score: float   # 20% weight - unconscious habits  
    lexical_complexity_score: float # 10% weight - vocabulary patterns
    punctuation_score: float       # 5% weight - formatting habits
    
    # Optional/diagnostic
    topic_mismatch_warning: Optional[str]  # If topics are very different
    
    confidence: float
    breakdown: Dict[str, float]

class ForensicLinguisticsService:
    def __init__(self, model_manager, cache_dir: str = "./forensic_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.progress_cache = {}
        self.progress_lock = asyncio.Lock()
        self.active_embedding_model = None  # No embedding model loaded at startup - load manually via UI
        # --- CORRECTED INITIALIZATION ORDER ---
        self.model_manager = model_manager
        self.embedding_models = {
            'gme': {'enabled': False, 'dimensions': None, 'priority': 1},
            'bge_m3': {'enabled': False, 'dimensions': 1024, 'priority': 2},
            'gte_qwen2': {'enabled': False, 'dimensions': 3584, 'priority': 3},
            'inf_retriever': {'enabled': False, 'dimensions': 3584, 'priority': 4},
            'sentence_t5': {'enabled': False, 'dimensions': 768, 'priority': 5},
            'star': {'enabled': False, 'dimensions': 1024, 'priority': 6},
            'roberta': {'enabled': False, 'dimensions': 768, 'priority': 7},
            'jina_v3': {'enabled': False, 'dimensions': 1024, 'priority': 8},
            'nomic_v1_5': {'enabled': False, 'dimensions': 768, 'priority': 9},  
            'arctic_embed': {'enabled': False, 'dimensions': 768, 'priority': 10},
            'mxbai_large': {'enabled': False, 'dimensions': 1024, 'priority': 11},
            'multilingual_e5': {'enabled': False, 'dimensions': 1024, 'priority': 12},
            'qwen3_8b': {'enabled': False, 'dimensions': None, 'priority': 13},
            'qwen3_4b': {'enabled': False, 'dimensions': None, 'priority': 14},
            'frida': {'enabled': False, 'dimensions': 1024, 'priority': 15},
        }
        # --- END CORRECTION ---

        # Initialize web search service
        self.web_search = WebSearchService()
        
        # Initialize STAR model for authorship embeddings
        self.star_tokenizer = None
        self.star_model = None
        # NOTE: STAR model disabled - "Princeton-NLP/STAR" doesn't exist on HuggingFace
        # self._init_star_model()
        
        # RoBERTa model - loaded manually via UI, not at startup
        self.roberta_tokenizer = None
        self.roberta_model = None
        # NOTE: RoBERTa auto-load disabled - use the Forensic Models UI to load embeddings manually
        # self._init_roberta_model()
        
        # Stylometric analyzers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.scaler = StandardScaler()

        # EXPANDED function words for robust stylometric analysis
        # These are the most reliable indicators - people use them subconsciously
        self.function_words = {
            'articles': ['a', 'an', 'the'],
            
            'pronouns': [
                # Personal
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 
                'me', 'him', 'her', 'us', 'them',
                # Possessive
                'my', 'your', 'his', 'her', 'its', 'our', 'their',
                'mine', 'yours', 'hers', 'ours', 'theirs',
                # Demonstrative  
                'this', 'that', 'these', 'those',
                # Relative
                'who', 'whom', 'whose', 'which', 'that',
                # Reflexive
                'myself', 'yourself', 'himself', 'herself', 'itself', 
                'ourselves', 'yourselves', 'themselves'
            ],
            
            'prepositions': [
                'in', 'on', 'at', 'by', 'for', 'with', 'to', 'of', 'from', 'about',
                'into', 'onto', 'upon', 'within', 'without', 'throughout', 'between',
                'among', 'beside', 'besides', 'beneath', 'below', 'under', 'over',
                'above', 'across', 'through', 'during', 'until', 'since', 'before',
                'after', 'behind', 'beyond', 'against', 'toward', 'towards', 'around'
            ],
            
            'conjunctions': [
                # Coordinating
                'and', 'or', 'but', 'nor', 'for', 'yet', 'so',
                # Subordinating
                'because', 'although', 'though', 'while', 'whereas', 'if', 'unless',
                'until', 'when', 'whenever', 'since', 'as', 'than', 'whether'
            ],
            
            'modal_verbs': ['can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'must', 'ought'],
            
            'auxiliary_verbs': ['be', 'am', 'is', 'are', 'was', 'were', 'been', 'being', 
                               'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing'],
            
            'determiners': ['a', 'an', 'the', 'this', 'that', 'these', 'those', 'my', 'your', 
                           'his', 'her', 'its', 'our', 'their', 'some', 'any', 'each', 'every',
                           'either', 'neither', 'much', 'many', 'few', 'little', 'several', 
                           'all', 'both', 'half', 'no', 'none'],
            
            'adverbs_common': [
                # Frequency
                'always', 'often', 'usually', 'sometimes', 'rarely', 'never', 'ever',
                # Degree
                'very', 'quite', 'rather', 'too', 'so', 'enough', 'almost', 'nearly',
                # Manner
                'well', 'just', 'really', 'actually', 'probably', 'perhaps', 'maybe',
                # Time
                'now', 'then', 'soon', 'still', 'yet', 'already',
                # Place
                'here', 'there', 'everywhere', 'anywhere', 'somewhere', 'nowhere'
            ],
            
            'negations': ['no', 'not', 'never', 'neither', 'nor', 'none', 'nobody', 'nothing', 'nowhere'],
            
            'wh_words': ['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how']
        }
        
        # Flatten for easy lookup
        self.function_words_flat = set()
        for category in self.function_words.values():
            self.function_words_flat.update(category)
        
        # Platform-specific scrapers
        self.scrapers = {
            'twitter': self._scrape_twitter,
            'truth_social': self._scrape_truth_social,
            'speeches': self._scrape_speeches,
            'press_releases': self._scrape_press_releases,
            'interviews': self._scrape_interviews
        }

    def _init_star_model(self):
        """Initialize STAR model for authorship attribution"""
        try:
            logger.info("Initializing STAR model for stylometric analysis...")
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            model_name = "Princeton-NLP/STAR"
            self.star_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.star_model = AutoModel.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                self.star_model = self.star_model.to('cuda')
                logger.info("STAR model loaded on CUDA")
            else:
                logger.info("STAR model loaded on CPU")
            
            self.star_model.eval()
            self.embedding_models['star']['enabled'] = True
            logger.info("✅ STAR model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize STAR model: {e}")
            self.embedding_models['star']['enabled'] = False
            self.star_tokenizer = None
            self.star_model = None
            
    def _init_roberta_model(self):
        """Initialize RoBERTa for additional semantic analysis"""
        try:
            logger.info("Initializing RoBERTa model for semantic analysis...")
            from transformers import RobertaTokenizer, RobertaModel
            import torch
            
            model_name = "roberta-base"
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.roberta_model = RobertaModel.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                self.roberta_model = self.roberta_model.to('cuda')
                logger.info("RoBERTa model loaded on CUDA")
            else:
                logger.info("RoBERTa model loaded on CPU")
            
            self.roberta_model.eval()
            self.embedding_models['roberta']['enabled'] = True
            logger.info("✅ RoBERTa model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RoBERTa model: {e}")
            self.embedding_models['roberta']['enabled'] = False
            self.roberta_tokenizer = None
            self.roberta_model = None

    async def unload_forensic_models(self):
        """Unload roberta and star models from memory to free VRAM"""
        try:
            import torch
            
            # Unload RoBERTa model
            if self.roberta_model is not None:
                logger.info("🧹 Unloading RoBERTa model...")
                del self.roberta_model
                self.roberta_model = None
                self.roberta_tokenizer = None
                self.embedding_models['roberta']['enabled'] = False
                logger.info("✅ RoBERTa model unloaded")
            
            # Unload STAR model
            if self.star_model is not None:
                logger.info("🧹 Unloading STAR model...")
                del self.star_model
                self.star_model = None
                self.star_tokenizer = None
                self.embedding_models['star']['enabled'] = False
                logger.info("✅ STAR model unloaded")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✅ CUDA cache cleared")
            
            # Also unload any model manager models
            if hasattr(self, 'model_manager'):
                await self.model_manager.unload_model_purpose('forensic_embeddings')
            
            return True
        except Exception as e:
            logger.error(f"❌ Error unloading forensic models: {e}", exc_info=True)
            return False

    def _get_gguf_patterns_for_model(self, model_name: str, model_key: str) -> List[str]:
        """Get specific GGUF filename patterns for a model to avoid collisions"""
        
        # Exact model-specific patterns
        specific_patterns = {
            'jina_v3': ['jina-embeddings-v3.gguf', 'jinaai-jina-embeddings-v3.gguf'],
            'nomic_v1_5': ['nomic-embed-text-v1.5.gguf', 'nomic-ai-nomic-embed-text-v1.5.gguf'],
            'arctic_embed': ['arctic-embed-m-v1.5.gguf', 'snowflake-arctic-embed-m-v1.5.gguf'],
            'gte_qwen2': ['gte-qwen2-7b-instruct.gguf', 'alibaba-nlp-gte-qwen2-7b-instruct.gguf'],
            'inf_retriever': ['inf-retriever-v1.gguf', 'infly-inf-retriever-v1.gguf'],
            'sentence_t5': ['sentence-t5-xxl.gguf'],
            'mxbai_large': ['mxbai-embed-large-v1.gguf', 'mixedbread-ai-mxbai-embed-large-v1.gguf'],
            'multilingual_e5': ['multilingual-e5-large.gguf', 'sentence-transformers-multilingual-e5-large.gguf'],
            'qwen3_8b': ['qwen3-8b.gguf', 'qwen3-qwen3-8b.gguf'],
            'qwen3_4b': ['qwen3-4b.gguf', 'qwen3-qwen3-4b.gguf'],
            'frida': ['frida.gguf', 'ai-forever-frida.gguf']
        }
        
        if model_key in specific_patterns:
            return specific_patterns[model_key]
        
        # Fallback to generic patterns for other models
        return [
            f"{model_name.split('/')[-1]}.gguf",
            f"{model_name.replace('/', '-')}.gguf",
            f"{model_name.replace('/', '_')}.gguf"
        ]            
    def _get_embedding_gpu_config(self):
        """Determine GPU configuration for embeddings based on system mode"""
        # Use GPU 0 for forensic embedding models
        logging.info("Using GPU 0 for forensic embedding models")
        return {"gpu_id": 0, "device": "cuda:0", "allow_multi_gpu": False}
    async def initialize_embedding_model(self, model_type: str, gpu_id: int = 0):
        """Initialize any embedding model, unload existing ones first, and set the new one as active."""

        model_configs = {
            'star': "Princeton-NLP/STAR",
            'roberta': "roberta-base",
            'gme': "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
            'bge_m3': "BAAI/bge-m3",
            'gte_qwen2': "Alibaba-NLP/gte-Qwen2-7B-instruct",
            'inf_retriever': "infly/inf-retriever-v1",
            'sentence_t5': "sentence-transformers/sentence-t5-xxl",
            'jina_v3': "jinaai/jina-embeddings-v3",
            'nomic_v1_5': "nomic-ai/nomic-embed-text-v1.5",
            'arctic_embed': "Snowflake/snowflake-arctic-embed-l-v2.0",
            'mxbai_large': "MixedBread/mxbai-embed-large-v1",
            'multilingual_e5': "intfloat/multilingual-e5-large",
            'qwen3_8b': "Qwen/qwen3-embedding-8b",
            'qwen3_4b': "Qwen/qwen3-embedding-4b",
            'frida': "AI-Forever/frida-embedding"
        }

        if model_type not in model_configs:
            raise ValueError(f"Unknown model type: {model_type}")

        model_name = model_configs[model_type]

        try:
            success = False  # Initialize success variable
            
            logger.info(f"🧹 Unloading existing forensic model before loading '{model_type}'")
            await self.model_manager.unload_model_purpose('forensic_embeddings')
            
            for key in self.embedding_models:
                if key not in ['star', 'roberta']:
                    self.embedding_models[key]['enabled'] = False   
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"🔍 [Forensic] Initializing '{model_type}' ({model_name}) on GPU {gpu_id}")

            # Handle different model types
            if model_type == 'star':
                self._init_star_model()
                success = self.embedding_models['star']['enabled']
            elif model_type == 'roberta':
                self._init_roberta_model()
                success = self.embedding_models['roberta']['enabled']
            elif model_type == 'bge_m3':
                success = self._init_bge_m3_model(model_name, model_type)
            elif model_type in ['jina_v3', 'nomic_v1_5', 'arctic_embed', 'gte_qwen2', 'inf_retriever', 'sentence_t5', 'mxbai_large', 'multilingual_e5', 'frida']:
                success = await self._init_sentence_transformers_model(model_name, model_type)
            else:
                # Handle GME and other model_manager models
                await self.model_manager.load_model_for_purpose(
                    purpose="forensic_embeddings",
                    model_name=model_name,
                    gpu_id=gpu_id,
                    context_length=8192 if model_type == 'bge_m3' else 4096
                )
                success = True

            # Check if initialization was successful
            if not success:
                logger.error(f"❌ Failed to initialize {model_type} model.")
                return False

            # Set model as active and enabled
            self.embedding_models[model_type]['enabled'] = True
            self.active_embedding_model = model_type
            logger.info(f"✅ Set '{model_type}' as the new active embedding model.")

            # --- NEW: Direct Verification Logic ---
            logger.info(f"Verifying new model '{model_type}' with a direct embedding test...")
            test_embedding = []
            
            if model_type in ['gme', 'qwen3_8b', 'qwen3_4b']:
                # For models loaded by ModelManager, use the dedicated function
                test_embedding = self._get_model_manager_embeddings("test")
            elif model_type == 'bge_m3':
                test_embedding = self._get_bge_m3_embeddings("test")
            elif model_type in ['jina_v3', 'nomic_v1_5', 'arctic_embed', 'gte_qwen2', 'inf_retriever', 'sentence_t5', 'mxbai_large', 'multilingual_e5', 'frida']:
                # For sentence-transformers models
                test_embedding = self._get_sentence_transformers_embeddings("test", model_type)
            else:
                # Fallback for any other type (like star, roberta)
                logger.warning(f"Using standard _get_semantic_embeddings for verification of '{model_type}'.")
                test_embedding = self._get_semantic_embeddings("test")

            if test_embedding and len(test_embedding) > 0:
                self.embedding_models[model_type]['dimensions'] = len(test_embedding)
                self.active_embedding_model = model_type
                logger.info(f"✅ Set '{model_type}' as the new active embedding model.")
                logger.info(f"✅ {model_type.upper()} model initialized and verified successfully with {len(test_embedding)} dimensions.")
                return True
            else:
                self.embedding_models[model_type]['enabled'] = False
                # Don't revert to 'star' - keep existing active model or fall back to roberta
                logger.error(f"❌ Failed to initialize {model_type}, test embedding failed or returned empty.")
                # Attempt to unload the failed model to free VRAM
                await self.model_manager.unload_model_purpose('forensic_embeddings')
                return False
            # --- END of NEW LOGIC ---

        except Exception as e:
            logger.error(f"❌ Error initializing {model_type}: {e}", exc_info=True)
            self.embedding_models[model_type]['enabled'] = False
            # Don't revert to 'star' - keep existing active model
            return False

    async def _init_sentence_transformers_model(self, model_name: str, model_key: str):
        """Initialize embedding model with GGUF + sentence-transformers support"""
        try:
            import torch

            # Get GPU configuration based on system mode
            gpu_config = self._get_embedding_gpu_config()
            target_gpu_id = gpu_config["gpu_id"]
            device = gpu_config["device"]
            allow_multi_gpu = gpu_config["allow_multi_gpu"]

            logger.info(f"Loading {model_name} in {getattr(self.model_manager, 'gpu_usage_mode', 'unknown')} mode - target: {device}")

            # 1. FIRST: Try to find GGUF version using your existing ModelManager
            if hasattr(self, 'model_manager'):
                if model_key == 'jina_v3':
                    gguf_patterns = ['jina-embeddings-v3.gguf']
                elif model_key == 'nomic_v1_5':
                    gguf_patterns = ['nomic-embed-text-v1.5.gguf']
                elif model_key == 'arctic_embed':
                    gguf_patterns = ['arctic-embed-m-v1.5.gguf']
                elif model_key == 'frida':
                    gguf_patterns = ['frida-embed-m-v1.5.gguf']
                elif model_key in ['mxbai_large', 'multilingual_e5', 'qwen3_8b', 'qwen3_4b']:
                    gguf_patterns = self._get_gguf_patterns_for_model(model_name, model_key)
                else:
                    gguf_patterns = [
                        f"{model_name.split('/')[-1]}.gguf",
                        f"{model_name.replace('/', '-')}.gguf",
                    ]

                for pattern in gguf_patterns:
                    try:
                        model_path = self.model_manager.models_dir / pattern
                        if not model_path.exists():
                            continue

                        # Directly await the model loading since we are in an async function
                        await self.model_manager.load_model(
                            model_name=pattern,
                            gpu_id=target_gpu_id,
                            n_ctx=4096
                        )

                        gguf_model = self.model_manager.get_model(pattern, target_gpu_id)

                        if gguf_model:
                            test_result = gguf_model.embed("test")
                            if test_result and test_result.get("status") == "success":
                                test_embedding = test_result.get("embedding", [])
                                if test_embedding:
                                    setattr(self, f'{model_key}_model', gguf_model)
                                    setattr(self, f'{model_key}_is_gguf', True)
                                    self.embedding_models[model_key]['enabled'] = True
                                    self.embedding_models[model_key]['dimensions'] = len(test_embedding)
                                    logger.info(f"✅ {model_name} loaded as GGUF on {device} ({len(test_embedding)}D)")
                                    return True

                    except Exception as gguf_error:
                        logger.debug(f"GGUF pattern {pattern} failed: {gguf_error}")
                        continue

                logger.info(f"❌ No GGUF version found for {model_name}")
                # --- THIS IS THE FIX ---
                # For GTE, if no GGUF is found, stop immediately.
                if model_key == 'gte_qwen2':
                    logger.error(f"GGUF for {model_name} not found. Download is disabled for this model.")
                    return False
                # --- END FIX ---
            if hasattr(self, 'model_manager') and hasattr(self.model_manager, 'models_dir'):
                from sentence_transformers import SentenceTransformer

                local_models_dir = Path(self.model_manager.models_dir)
                possible_names = [
                    model_name.replace('/', '-'),
                    model_name.replace('/', '_'),
                    model_name.split('/')[-1],
                    f"embeddings/{model_name.split('/')[-1]}"
                ]

                for folder_name in possible_names:
                    local_model_path = local_models_dir / folder_name
                    if local_model_path.exists() and local_model_path.is_dir():
                        logger.info(f"✅ Found sentence-transformers model in your models folder: {local_model_path}")
                        try:
                            # Load using transformers directly for local paths
                            from transformers import AutoModel, AutoTokenizer

                            tokenizer = AutoTokenizer.from_pretrained(str(local_model_path), trust_remote_code=True)
                            model = AutoModel.from_pretrained(
                                str(local_model_path),
                                trust_remote_code=True,
                                device_map=device if device != 'cpu' else None
                            )

                            # Create a simple wrapper that mimics SentenceTransformer.encode()
                            class LocalSentenceTransformer:
                                def __init__(self, model, tokenizer, device):
                                    self.model = model
                                    self.tokenizer = tokenizer
                                    self.device = device

                                def encode(self, text):
                                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                                    if self.device != 'cpu':
                                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                                    with torch.no_grad():
                                        outputs = self.model(**inputs)
                                        # Mean pooling
                                        embeddings = outputs.last_hidden_state.mean(dim=1)
                                        return embeddings.cpu().numpy()[0]

                            wrapper = LocalSentenceTransformer(model, tokenizer, device)
                            test_embedding = wrapper.encode("test")

                            if test_embedding is not None and len(test_embedding) > 0:
                                setattr(self, f'{model_key}_model', model)
                                setattr(self, f'{model_key}_is_gguf', False)
                                self.embedding_models[model_key]['enabled'] = True
                                self.embedding_models[model_key]['dimensions'] = len(test_embedding)
                                logger.info(f"✅ {model_name} loaded as sentence-transformers from local folder on {device} ({len(test_embedding)}D)")
                                return True

                        except Exception as local_error:
                            logger.warning(f"Failed to load sentence-transformers from local folder: {local_error}")
                            if allow_multi_gpu and "out of memory" in str(local_error).lower():
                                logger.info(f"CUDA OOM detected, trying CPU fallback for {model_name}")
                                try:
                                    model = SentenceTransformer(
                                        str(local_model_path),
                                        device='cpu',
                                        trust_remote_code=True
                                    )
                                    test_embedding = model.encode("test")
                                    if test_embedding is not None and len(test_embedding) > 0:
                                        setattr(self, f'{model_key}_model', model)
                                        setattr(self, f'{model_key}_is_gguf', False)
                                        self.embedding_models[model_key]['enabled'] = True
                                        self.embedding_models[model_key]['dimensions'] = len(test_embedding)
                                        logger.info(f"✅ {model_name} loaded on CPU fallback ({len(test_embedding)}D)")
                                        return True
                                except Exception as cpu_error:
                                    logger.warning(f"CPU fallback also failed: {cpu_error}")
                            continue
            
            # --- START OF MOVED BLOCK ---
            from sentence_transformers import SentenceTransformer
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            local_model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"

            if local_model_dir.exists():
                logger.info(f"✅ Found model in HF cache: {local_model_dir}")
                try:
                    model = SentenceTransformer(
                        str(local_model_dir),
                        device=device,
                        trust_remote_code=True
                    )

                    test_embedding = model.encode("test")
                    if test_embedding is not None and len(test_embedding) > 0:
                        setattr(self, f'{model_key}_model', model)
                        setattr(self, f'{model_key}_is_gguf', False)
                        self.embedding_models[model_key]['enabled'] = True
                        self.embedding_models[model_key]['dimensions'] = len(test_embedding)
                        logger.info(f"✅ {model_name} loaded from HF cache on {device} ({len(test_embedding)}D)")
                        return True

                except Exception as cache_error:
                    logger.warning(f"Failed to load from HF cache: {cache_error}")
                    if allow_multi_gpu and "out of memory" in str(cache_error).lower():
                        logger.info(f"CUDA OOM from cache, trying CPU fallback for {model_name}")
                        try:
                            model = SentenceTransformer(str(local_model_dir), device='cpu', trust_remote_code=True)
                            test_embedding = model.encode("test")
                            if test_embedding is not None and len(test_embedding) > 0:
                                setattr(self, f'{model_key}_model', model)
                                setattr(self, f'{model_key}_is_gguf', False)
                                self.embedding_models[model_key]['enabled'] = True
                                self.embedding_models[model_key]['dimensions'] = len(test_embedding)
                                logger.info(f"✅ {model_name} loaded from cache on CPU fallback ({len(test_embedding)}D)")
                                return True
                        except Exception as cpu_error:
                            logger.warning(f"CPU fallback from cache failed: {cpu_error}")

            logger.info(f"📥 Downloading {model_name} from HuggingFace...")

            try:
                model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
                test_embedding = model.encode("test")

                if test_embedding is not None and len(test_embedding) > 0:
                    setattr(self, f'{model_key}_model', model)
                    setattr(self, f'{model_key}_is_gguf', False)
                    self.embedding_models[model_key]['enabled'] = True
                    self.embedding_models[model_key]['dimensions'] = len(test_embedding)
                    logger.info(f"✅ {model_name} downloaded and loaded on {device} ({len(test_embedding)}D)")
                    return True
                else:
                    logger.error(f"❌ {model_name} failed test embedding")
                    return False

            except Exception as download_error:
                logger.warning(f"Download to {device} failed: {download_error}")
                if allow_multi_gpu and "out of memory" in str(download_error).lower():
                    logger.info(f"CUDA OOM on download, trying CPU fallback for {model_name}")
                    try:
                        model = SentenceTransformer(model_name, trust_remote_code=True, device='cpu')
                        test_embedding = model.encode("test")
                        if test_embedding is not None and len(test_embedding) > 0:
                            setattr(self, f'{model_key}_model', model)
                            setattr(self, f'{model_key}_is_gguf', False)
                            self.embedding_models[model_key]['enabled'] = True
                            self.embedding_models[model_key]['dimensions'] = len(test_embedding)
                            logger.info(f"✅ {model_name} downloaded and loaded on CPU fallback ({len(test_embedding)}D)")
                            return True
                    except Exception as cpu_error:
                        logger.error(f"CPU fallback on download failed: {cpu_error}")

                return False

        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            self.embedding_models[model_key]['enabled'] = False
            return False
        
    def _init_bge_m3_model(self, model_name: str, model_key: str):
        """Initialize BGE-M3 model using the FlagEmbedding library."""
        try:
            from FlagEmbedding import BGEM3FlagModel
            
            logger.info(f"Loading {model_name} using BGEM3FlagModel...")
            # We don't specify a device, letting the library auto-detect CUDA
            model = BGEM3FlagModel(model_name, use_fp16=True)
            
            test_embedding = model.encode('test')['dense_vecs']
            
            if test_embedding is not None and len(test_embedding) > 0:
                setattr(self, f'{model_key}_model', model)
                setattr(self, f'{model_key}_is_gguf', False) # It's not a GGUF
                self.embedding_models[model_key]['enabled'] = True
                self.embedding_models[model_key]['dimensions'] = len(test_embedding)
                logger.info(f"✅ {model_name} loaded successfully ({len(test_embedding)}D)")
                return True
            return False
        except ImportError:
            logger.error("❌ FlagEmbedding library not found. Please run 'pip install -U FlagEmbedding'")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}", exc_info=True)
            return False
    def _get_bge_m3_embeddings(self, text: str) -> List[float]:
        """Get embeddings specifically from the BGE-M3 model."""
        try:
            model = getattr(self, 'bge_m3_model', None)
            if not model:
                return []
            
            # BGE-M3 returns a dictionary, we need the 'dense_vecs'
            output = model.encode(text, return_dense=True, return_sparse=False, return_colbert_vecs=False)
            embedding = output.get('dense_vecs', [])

            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
        except Exception as e:
            logger.warning(f"BGE-M3 embedding failed: {e}")
            return []
    def _get_sentence_transformers_embeddings(self, text: str, model_key: str) -> List[float]:
        """Get embeddings from GGUF or sentence-transformers model"""
        try:
            model = getattr(self, f'{model_key}_model', None)
            is_gguf = getattr(self, f'{model_key}_is_gguf', False)

            if not model:
                return []

            if is_gguf:
                # Use GGUF model's embed method
                result = model.embed(text)
                if result and result.get("status") == "success":
                    embedding = result.get("embedding", [])
                    
                    # --- FIX FOR GGUF PER-TOKEN EMBEDDINGS ---
                    # Check if the GGUF model returned a list of embeddings (one for each token)
                    if embedding and isinstance(embedding, (list, np.ndarray)) and embedding[0] and isinstance(embedding[0], (list, np.ndarray)):
                        logger.debug(f"Detected per-token embeddings from GGUF model (token count: {len(embedding)}). Performing mean pooling.")
                        try:
                            # Convert to numpy array, calculate mean across all token embeddings, convert back to list
                            pooled_embedding = np.array(embedding, dtype=np.float32).mean(axis=0).tolist()
                            logger.debug(f"GGUF pooled embedding dimension: {len(pooled_embedding)}D")
                            return pooled_embedding
                        except Exception as e:
                            logger.error(f"Failed to perform mean pooling on GGUF output: {e}")
                            return [] # Return empty on failure
                    # --- END FIX ---
                    
                    return embedding if isinstance(embedding, list) else embedding.tolist()
                return []
            else:
                # Use sentence-transformers model
                embedding = model.encode(text)
                
                # DEBUG: Check what GTE is actually returning
                if model_key == 'gte_qwen2':
                    logger.error(f"GTE raw output type: {type(embedding)}")
                    logger.error(f"GTE raw output shape: {getattr(embedding, 'shape', 'no shape attr')}")
                    logger.error(f"GTE raw output length: {len(embedding) if hasattr(embedding, '__len__') else 'no len'}")
                    if hasattr(embedding, '__len__') and len(embedding) > 0:
                        logger.error(f"GTE first element type: {type(embedding[0])}")
                        if hasattr(embedding[0], '__len__'):
                            logger.error(f"GTE first element length: {len(embedding[0])}")
                
                result = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

                # FIX: GTE specifically returns nested lists
                if model_key == 'gte_qwen2' and result and isinstance(result[0], list):
                    result = result[0]  # Take the inner list for GTE
                    logger.debug(f"Fixed GTE nested list: {len(result)}D")

                return result

        except Exception as e:
            logger.warning(f"{model_key} embedding failed: {e}")
            return []
        
    async def build_corpus(self, 
                          person_name: str, 
                          platforms: List[str] = None,
                          date_range: Tuple[datetime, datetime] = None,
                          max_documents: int = 1000) -> List[TextDocument]:
        """Build a comprehensive corpus for a public figure"""
        
        if platforms is None:
            platforms = ['twitter', 'speeches', 'press_releases', 'interviews']
            
        logger.info(f"Building corpus for {person_name} from platforms: {platforms}")
        
        corpus = []
        
        for platform in platforms:
            try:
                logger.info(f"Scraping {platform} for {person_name}...")
                
                if platform in self.scrapers:
                    documents = await self.scrapers[platform](person_name, date_range, max_documents // len(platforms))
                    corpus.extend(documents)
                    logger.info(f"Collected {len(documents)} documents from {platform}")
                else:
                    # Generic web search fallback
                    documents = await self._generic_search(person_name, platform, date_range, max_documents // len(platforms))
                    corpus.extend(documents)
                    
            except Exception as e:
                logger.error(f"Error scraping {platform}: {e}")
                continue
                
        # Cache the corpus
        self._cache_corpus(person_name, corpus)
        
        logger.info(f"Built corpus with {len(corpus)} total documents for {person_name}")
        return corpus
    async def initialize_gme_model(self, model_name: str, gpu_id: int = 0):
        """Initialize GME model using ModelManager"""
        try:
            logger.info(f"🔍 [Forensic] Loading GME model: {model_name} on GPU {gpu_id}")

            await self.model_manager.load_model_for_purpose(
                purpose="forensic_embeddings",
                model_name=model_name,
                gpu_id=gpu_id,
                context_length=4096
            )

            # Temporarily enable GME for testing
            self.embedding_models['gme']['enabled'] = True

            # Test if it works
            test_embedding = self._get_gme_embeddings("test")
            if test_embedding:
                self.embedding_models['gme']['dimensions'] = len(test_embedding)
                logger.info(f"✅ GME model loaded successfully ({len(test_embedding)}D)")
                return True
            else:
                # Disable if test failed
                self.embedding_models['gme']['enabled'] = False
                logger.error("❌ GME model failed to produce test embedding")
                return False

        except Exception as e:
            logger.error(f"❌ GME initialization failed: {e}")
            self.embedding_models['gme']['enabled'] = False
            return False
    async def _scrape_twitter(self, person_name: str, date_range: Tuple[datetime, datetime] = None, max_docs: int = 500) -> List[TextDocument]:
        """Scrape Twitter/X posts"""
        documents = []
        
        # Twitter archive search patterns
        search_patterns = [
            f'"{person_name}" site:twitter.com',
            f'"{person_name}" site:x.com',
            f'{person_name.lower().replace(" ", "")} twitter archive',
            f'{person_name} tweets site:thetrumparchive.com' if 'trump' in person_name.lower() else None
        ]
        
        for pattern in search_patterns:
            if pattern is None:
                continue
                
            try:
                results = await self.web_search.search_duckduckgo(pattern, max_results=20)
                scraped_results = await self.web_search.scrape_content(results)
                
                for result in scraped_results:
                    if result.scraped_successfully and result.content:
                        # Extract tweets from the scraped content
                        tweets = self._extract_tweets_from_content(result.content, person_name)
                        
                        for tweet_text, tweet_date in tweets:
                            if len(tweet_text.strip()) > 10:  # Filter very short content
                                doc = TextDocument(
                                    content=tweet_text,
                                    source_url=result.url,
                                    date=tweet_date,
                                    platform='twitter',
                                    author=person_name,
                                    metadata={'extracted_from': 'web_scrape'}
                                )
                                documents.append(doc)
                                
                                if len(documents) >= max_docs:
                                    return documents
                                    
            except Exception as e:
                logger.warning(f"Error in Twitter search pattern '{pattern}': {e}")
                continue
                
        return documents

    async def _scrape_speeches(self, person_name: str, date_range: Tuple[datetime, datetime] = None, max_docs: int = 200) -> List[TextDocument]:
        """Scrape speech transcripts"""
        documents = []
        
        # Speech-specific search patterns
        search_patterns = [
            f'"{person_name}" speech transcript',
            f'"{person_name}" remarks transcript site:whitehouse.gov',
            f'"{person_name}" speech text',
            f'"{person_name}" transcript rally',
            f'"{person_name}" press conference transcript'
        ]
        
        for pattern in search_patterns:
            try:
                results = await self.web_search.search_duckduckgo(pattern, max_results=15)
                scraped_results = await self.web_search.scrape_content(results)
                
                for result in scraped_results:
                    if result.scraped_successfully and result.content:
                        # Clean and process speech content
                        speech_content = self._clean_speech_transcript(result.content)
                        
                        if len(speech_content) > 500:  # Substantial content only
                            doc = TextDocument(
                                content=speech_content,
                                source_url=result.url,
                                date=self._extract_date_from_content(result.content),
                                platform='speeches',
                                author=person_name,
                                title=result.title
                            )
                            documents.append(doc)
                            
                            if len(documents) >= max_docs:
                                return documents
                                
            except Exception as e:
                logger.warning(f"Error in speech search: {e}")
                continue
                
        return documents
    
    async def _update_progress(self, task_id: str, progress: float, status: str, result: Optional[Dict] = None):
        """Safely update the progress of a task."""
        async with self.progress_lock:
            self.progress_cache[task_id] = {
                "progress": round(progress, 2),
                "status": status,
                "result": result,
                "timestamp": time.time()
            }
            logger.info(f"🔍 [Progress {task_id[:8]}] {progress:.1f}% - {status}")
    def get_progress(self, task_id: str) -> Optional[Dict]:
        """Get the progress of a task."""
        return self.progress_cache.get(task_id)

    async def _scrape_press_releases(self, person_name: str, date_range: Tuple[datetime, datetime] = None, max_docs: int = 200) -> List[TextDocument]:
        """Scrape press releases and official statements"""
        documents = []
        
        search_patterns = [
            f'"{person_name}" press release',
            f'"{person_name}" statement official',
            f'"{person_name}" announces site:gov',
            f'"{person_name}" campaign statement'
        ]
        
        for pattern in search_patterns:
            try:
                results = await self.web_search.search_duckduckgo(pattern, max_results=10)
                scraped_results = await self.web_search.scrape_content(results)
                
                for result in scraped_results:
                    if result.scraped_successfully and result.content:
                        # Clean press release content
                        content = self._clean_press_release(result.content)
                        
                        if len(content) > 200:
                            doc = TextDocument(
                                content=content,
                                source_url=result.url,
                                date=self._extract_date_from_content(result.content),
                                platform='press_releases',
                                author=person_name,
                                title=result.title
                            )
                            documents.append(doc)
                            
                            if len(documents) >= max_docs:
                                return documents
                                
            except Exception as e:
                logger.warning(f"Error in press release search: {e}")
                continue
                
        return documents

    async def _scrape_interviews(self, person_name: str, date_range: Tuple[datetime, datetime] = None, max_docs: int = 100) -> List[TextDocument]:
        """Scrape interview transcripts"""
        documents = []
        
        search_patterns = [
            f'"{person_name}" interview transcript',
            f'"{person_name}" interview text CNN Fox',
            f'"{person_name}" Q&A transcript'
        ]
        
        for pattern in search_patterns:
            try:
                results = await self.web_search.search_duckduckgo(pattern, max_results=8)
                scraped_results = await self.web_search.scrape_content(results)
                
                for result in scraped_results:
                    if result.scraped_successfully and result.content:
                        # Extract interview content
                        interview_content = self._extract_interview_content(result.content, person_name)
                        
                        if len(interview_content) > 300:
                            doc = TextDocument(
                                content=interview_content,
                                source_url=result.url,
                                date=self._extract_date_from_content(result.content),
                                platform='interviews',
                                author=person_name,
                                title=result.title
                            )
                            documents.append(doc)
                            
                            if len(documents) >= max_docs:
                                return documents
                                
            except Exception as e:
                logger.warning(f"Error in interview search: {e}")
                continue
                
        return documents

    async def _scrape_truth_social(self, person_name: str, date_range: Tuple[datetime, datetime] = None, max_docs: int = 300) -> List[TextDocument]:
        """Scrape Truth Social posts"""
        documents = []
        
        search_patterns = [
            f'"{person_name}" site:truthsocial.com',
            f'{person_name} Truth Social posts',
            f'"{person_name}" truthsocial archive'
        ]
        
        for pattern in search_patterns:
            try:
                results = await self.web_search.search_duckduckgo(pattern, max_results=15)
                scraped_results = await self.web_search.scrape_content(results)
                
                for result in scraped_results:
                    if result.scraped_successfully and result.content:
                        # Extract posts from Truth Social content
                        posts = self._extract_social_posts(result.content, person_name)
                        
                        for post_text, post_date in posts:
                            if len(post_text.strip()) > 10:
                                doc = TextDocument(
                                    content=post_text,
                                    source_url=result.url,
                                    date=post_date,
                                    platform='truth_social',
                                    author=person_name
                                )
                                documents.append(doc)
                                
                                if len(documents) >= max_docs:
                                    return documents
                                    
            except Exception as e:
                logger.warning(f"Error in Truth Social search: {e}")
                continue
                
        return documents

    async def _generic_search(self, person_name: str, platform: str, date_range: Tuple[datetime, datetime] = None, max_docs: int = 50) -> List[TextDocument]:
        """Generic search fallback for other platforms"""
        documents = []
        
        query = f'"{person_name}" {platform} quotes statements'
        
        try:
            results = await self.web_search.search_duckduckgo(query, max_results=10)
            scraped_results = await self.web_search.scrape_content(results)
            
            for result in scraped_results:
                if result.scraped_successfully and result.content:
                    # Try to extract relevant quotes/statements
                    content = self._extract_relevant_content(result.content, person_name)
                    
                    if len(content) > 100:
                        doc = TextDocument(
                            content=content,
                            source_url=result.url,
                            date=self._extract_date_from_content(result.content),
                            platform=platform,
                            author=person_name,
                            title=result.title
                        )
                        documents.append(doc)
                        
                        if len(documents) >= max_docs:
                            break
                            
        except Exception as e:
            logger.warning(f"Error in generic search for {platform}: {e}")
            
        return documents
    
    def _get_star_embeddings(self, text: str) -> List[float]:
        """Get embeddings from the STAR model."""
        if not self.star_model or not self.star_tokenizer:
            return []
        try:
            inputs = self.star_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.star_model(**inputs)
            return outputs.pooler_output[0].cpu().numpy().tolist()
        except Exception as e:
            logger.warning(f"STAR embeddings failed: {e}")
            return []

    def _get_roberta_embeddings(self, text: str) -> List[float]:
        """Get embeddings from the RoBERTa model."""
        if not self.roberta_model or not self.roberta_tokenizer:
            return []
        try:
            inputs = self.roberta_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
            return outputs.pooler_output[0].cpu().numpy().tolist()
        except Exception as e:
            logger.warning(f"RoBERTa embeddings failed: {e}")
            return []    
    def _get_jina_v3_embeddings(self, text: str) -> List[float]:
        """Get embeddings from Jina v3 with task-specific LoRA adapter"""
        try:
            model = getattr(self, 'jina_v3_model', None)
            is_gguf = getattr(self, 'jina_v3_is_gguf', False)
            
            if not model:
                return []
                
            if is_gguf:
                result = model.embed(text)
                if result and result.get("status") == "success":
                    return result.get("embedding", [])
                return []
            else:
                # For forensic analysis, use the "separation" task for clustering-like analysis
                # This seems most appropriate for stylometric analysis
                embedding = model.encode(text, task="separation")
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.warning(f"Jina v3 embedding failed: {e}")
            return []

    def _get_nomic_v1_5_embeddings(self, text: str) -> List[float]:
        """Get embeddings from Nomic v1.5 with instruction prefix"""
        try:
            model = getattr(self, 'nomic_v1_5_model', None)
            is_gguf = getattr(self, 'nomic_v1_5_is_gguf', False)
            
            if not model:
                return []
                
            # For forensic analysis, treat text as documents to be analyzed
            prefixed_text = f"clustering: {text}"
            
            if is_gguf:
                result = model.embed(prefixed_text)
                if result and result.get("status") == "success":
                    return result.get("embedding", [])
                return []
            else:
                embedding = model.encode(prefixed_text)
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.warning(f"Nomic v1.5 embedding failed: {e}")
            return []

    def _get_arctic_embeddings(self, text: str) -> List[float]:
        """Get embeddings from Snowflake Arctic"""
        try:
            model = getattr(self, 'arctic_embed_model', None)
            is_gguf = getattr(self, 'arctic_embed_is_gguf', False)
            
            if not model:
                return []
                
            if is_gguf:
                result = model.embed(text)
                if result and result.get("status") == "success":
                    return result.get("embedding", [])
                return []
            else:
                # Arctic doesn't need prefix for document embeddings, only for queries
                embedding = model.encode(text)
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.warning(f"Arctic embedding failed: {e}")
            return []    
    async def extract_style_vector(self, text: str, platform: str = None) -> StyleVector:
        """Extract comprehensive stylometric features from text"""
        
        # Basic preprocessing
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Remove punctuation for some calculations
        words_no_punct = [w for w in words if w.isalpha()]
        
        if len(words_no_punct) == 0:
            # Return default vector for empty text
            return self._empty_style_vector()
        
        # Lexical features
        avg_word_length = np.mean([len(w) for w in words_no_punct])
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Vocabulary richness
        vocab_size = len(set(words_no_punct))
        total_words = len(words_no_punct)
        vocab_richness = vocab_size / total_words if total_words > 0 else 0
        
        # Hapax legomena (words appearing once)
        word_freq = Counter(words_no_punct)
        hapax_count = sum(1 for freq in word_freq.values() if freq == 1)
        hapax_ratio = hapax_count / total_words if total_words > 0 else 0
        
        # Yule's K (vocabulary diversity)
        yule_k = self._calculate_yule_k(word_freq)
        
        # POS tagging and syntactic features
        pos_tags = pos_tag(words)
        pos_counts = Counter([tag for word, tag in pos_tags])
        total_pos = sum(pos_counts.values())
        pos_distribution = {tag: count/total_pos for tag, count in pos_counts.items()}
        
        # Dependency parsing (if spaCy available)
        dependency_patterns = {}
        sentence_complexity = 0
        if nlp:
            loop = asyncio.get_event_loop()
            doc = await loop.run_in_executor(None, nlp, text[:1000000])
            dependency_patterns = self._extract_dependency_patterns(doc)
            sentence_complexity = np.mean([len(list(sent.root.subtree)) for sent in doc.sents]) if list(doc.sents) else 0
        
        # Punctuation analysis
        punctuation_chars = ['.', ',', '!', '?', ';', ':', '-', '"', "'"]
        total_chars = len(text)
        punctuation_ratios = {
            char: text.count(char) / total_chars for char in punctuation_chars
        }
        
        # Capitalization
        cap_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Paragraph analysis
        paragraphs = text.split('\n\n')
        avg_paragraph_length = np.mean([len(p.split()) for p in paragraphs if p.strip()])
        
        # Semantic embeddings
        loop = asyncio.get_event_loop()
        semantic_embeddings = await loop.run_in_executor(None, self._get_semantic_embeddings, text)
        original_embedding_dim = len(semantic_embeddings)
    
        # Pad with zeros if shorter, truncate if longer
        current_len = len(semantic_embeddings)
        if current_len < TARGET_EMBEDDING_DIM:
            # Pad the vector with zeros at the end
            semantic_embeddings.extend([0.0] * (TARGET_EMBEDDING_DIM - current_len))
        elif current_len > TARGET_EMBEDDING_DIM:
            # Truncate the vector
            semantic_embeddings = semantic_embeddings[:TARGET_EMBEDDING_DIM]
        # Sentiment analysis (basic)
        sentiment_scores = self._analyze_sentiment(text)
        
        # Function word analysis
        function_word_ratios = self._analyze_function_words(words_no_punct)
        
        # Modal verb usage
        modal_verbs = self.function_words['modal_verbs']
        modal_count = sum(words_no_punct.count(modal) for modal in modal_verbs)
        modal_verb_usage = modal_count / total_words if total_words > 0 else 0
        
        # Passive voice detection (basic heuristic)
        passive_indicators = ['was', 'were', 'been', 'being'] + [w + 'ed' for w in ['call', 'ask', 'tell', 'show']]
        passive_count = sum(text.lower().count(indicator) for indicator in passive_indicators)
        passive_voice_ratio = passive_count / len(sentences) if sentences else 0
        
        # Question and exclamation ratios
        question_ratio = text.count('?') / len(sentences) if sentences else 0
        exclamation_ratio = text.count('!') / len(sentences) if sentences else 0
        
        # Calculate new features for overhauled system
        # Character n-grams (simple implementation)
        char_trigrams = {}
        char_bigrams = {}
        text_lower = text.lower()
        for i in range(len(text_lower) - 2):
            trigram = text_lower[i:i+3]
            char_trigrams[trigram] = char_trigrams.get(trigram, 0) + 1
        for i in range(len(text_lower) - 1):
            bigram = text_lower[i:i+2]
            char_bigrams[bigram] = char_bigrams.get(bigram, 0) + 1
        
        # Normalize to frequencies
        total_trigrams = sum(char_trigrams.values())
        total_bigrams = sum(char_bigrams.values())
        if total_trigrams > 0:
            char_trigrams = {k: v/total_trigrams for k, v in char_trigrams.items()}
        if total_bigrams > 0:
            char_bigrams = {k: v/total_bigrams for k, v in char_bigrams.items()}
        
        # Function word frequencies (per 1000 words)
        function_word_frequencies = {}
        total_words_count = len(words_no_punct) if words_no_punct else 1
        for word in self.function_words_flat:
            count = words_no_punct.count(word)
            function_word_frequencies[word] = (count / total_words_count) * 1000
        
        # Function word bigrams
        function_word_bigrams = {}
        for i in range(len(words_no_punct) - 1):
            if words_no_punct[i] in self.function_words_flat and words_no_punct[i+1] in self.function_words_flat:
                bigram = f"{words_no_punct[i]}_{words_no_punct[i+1]}"
                function_word_bigrams[bigram] = function_word_bigrams.get(bigram, 0) + 1
        
        # Sentence length variance
        sent_lengths = [len(sent.split()) for sent in sentences]
        sentence_length_variance = np.var(sent_lengths) if len(sent_lengths) > 1 else 0.0
        
        # Word length distribution
        word_length_distribution = {}
        for word in words_no_punct:
            length = len(word)
            word_length_distribution[length] = word_length_distribution.get(length, 0) + 1
        total_words_for_dist = sum(word_length_distribution.values())
        if total_words_for_dist > 0:
            word_length_distribution = {k: v/total_words_for_dist for k, v in word_length_distribution.items()}
        
        # POS n-grams
        pos_bigrams = {}
        pos_trigrams = {}
        pos_only = [tag for _, tag in pos_tags]
        for i in range(len(pos_only) - 1):
            bigram = f"{pos_only[i]}_{pos_only[i+1]}"
            pos_bigrams[bigram] = pos_bigrams.get(bigram, 0) + 1
        for i in range(len(pos_only) - 2):
            trigram = f"{pos_only[i]}_{pos_only[i+1]}_{pos_only[i+2]}"
            pos_trigrams[trigram] = pos_trigrams.get(trigram, 0) + 1
        
        # Punctuation sequences
        punctuation_sequences = {}
        for i in range(len(text) - 1):
            if text[i] in '.,!?;:\'"' and text[i+1] in '.,!?;:\'"':
                seq = text[i:i+2]
                punctuation_sequences[seq] = punctuation_sequences.get(seq, 0) + 1
        
        # All caps ratio
        words_all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)
        all_caps_ratio = words_all_caps / len(words) if words else 0
        
        # Contraction ratio
        contractions = ["n't", "'ll", "'ve", "'re", "'d", "'m", "'s"]
        contraction_count = sum(text.lower().count(c) for c in contractions)
        contraction_ratio = contraction_count / len(words) if words else 0
        
        # Sentence starter patterns
        sentence_starter_patterns = {}
        for sent in sentences:
            sent_words = sent.split()
            if sent_words:
                starter = sent_words[0].lower()
                if starter in self.function_words_flat or len(starter) <= 3:
                    sentence_starter_patterns[starter] = sentence_starter_patterns.get(starter, 0) + 1
        
        # Clause patterns (simplified)
        clause_patterns = {'subordinate': 0, 'coordinate': 0}
        subordinating = ['because', 'although', 'if', 'when', 'while', 'since', 'unless', 'until']
        coordinating = ['and', 'but', 'or', 'nor', 'for', 'yet', 'so']
        for word in words_no_punct:
            if word in subordinating:
                clause_patterns['subordinate'] += 1
            elif word in coordinating:
                clause_patterns['coordinate'] += 1
        
        return StyleVector(
            # Character-level
            char_trigrams=char_trigrams,
            char_bigrams=char_bigrams,
            
            # Function words
            function_word_frequencies=function_word_frequencies,
            function_word_bigrams=function_word_bigrams,
            
            # Lexical
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length,
            sentence_length_variance=sentence_length_variance,
            vocab_richness=vocab_richness,
            hapax_legomena_ratio=hapax_ratio,
            yule_k=yule_k,
            word_length_distribution=word_length_distribution,
            
            # Syntactic
            pos_distribution=pos_distribution,
            pos_bigrams=pos_bigrams,
            pos_trigrams=pos_trigrams,
            dependency_patterns=dependency_patterns,
            sentence_complexity=sentence_complexity,
            clause_patterns=clause_patterns,
            
            # Punctuation
            punctuation_ratios=punctuation_ratios,
            punctuation_sequences=punctuation_sequences,
            capitalization_ratio=cap_ratio,
            all_caps_ratio=all_caps_ratio,
            contraction_ratio=contraction_ratio,
            avg_paragraph_length=avg_paragraph_length,
            
            # Stylistic
            modal_verb_usage=modal_verb_usage,
            passive_voice_ratio=passive_voice_ratio,
            question_ratio=question_ratio,
            exclamation_ratio=exclamation_ratio,
            sentence_starter_patterns=sentence_starter_patterns,
            
            # Semantic (minimal)
            semantic_embeddings=semantic_embeddings,
            semantic_embedding_dim=original_embedding_dim,
            
            # Metadata
            platform=platform
        )

    async def compare_styles(self, target_text: str, corpus_vectors: List[StyleVector], task_id: Optional[str] = None, starting_progress: float = 0.0) -> SimilarityScore:
        """Compare target text against corpus style vectors and report progress if task_id is provided."""
        
        target_vector = await self.extract_style_vector(target_text)
        
        if not corpus_vectors:
            return SimilarityScore(
                overall_score=0.0,
                function_word_score=0.0,
                syntactic_score=0.0,
                character_ngram_score=0.0,
                lexical_complexity_score=0.0,
                punctuation_score=0.0,
                topic_mismatch_warning="No corpus available",
                confidence=0.0,
                breakdown={}
            )
        
        # === CALCULATE ALL 5 SCORES SEPARATELY ===
        function_word_scores = []
        syntactic_scores = []
        character_ngram_scores = []
        lexical_complexity_scores = []
        punctuation_scores = []
        semantic_scores = []
        
        total_vectors = len(corpus_vectors)
        
        for i, corpus_vector in enumerate(corpus_vectors):
            # Function word similarity (most reliable)
            function_word_scores.append(self._calculate_function_word_similarity(target_vector, corpus_vector))
            
            # Syntactic patterns
            syntactic_scores.append(self._calculate_syntactic_similarity(target_vector, corpus_vector))
            
            # Character n-gram similarity
            character_ngram_scores.append(self._calculate_character_ngram_similarity(target_vector, corpus_vector))
            
            # Lexical complexity
            lexical_complexity_scores.append(self._calculate_lexical_similarity(target_vector, corpus_vector))
            
            # Punctuation style
            punctuation_scores.append(self._calculate_punctuation_similarity(target_vector, corpus_vector))
            
            # Semantic (for topic mismatch detection only)
            semantic_scores.append(
                self._calculate_semantic_similarity(
                    target_vector.semantic_embeddings,
                    corpus_vector.semantic_embeddings,
                    target_vector.semantic_embedding_dim,
                    corpus_vector.semantic_embedding_dim
                )
            )
            
            # If a task_id is provided, report progress for this intensive loop.
            if task_id:
                # This part of the process will cover 45% of the progress bar (from 50% to 95%)
                progress = starting_progress + ((i + 1) / total_vectors) * 45
                await self._update_progress(task_id, progress, f"Comparing with document {i + 1}/{total_vectors}")
        
        # Aggregate all scores
        function_word_score = np.mean(function_word_scores) if function_word_scores else 0.0
        syntactic_score = np.mean(syntactic_scores) if syntactic_scores else 0.0
        character_ngram_score = np.mean(character_ngram_scores) if character_ngram_scores else 0.0
        lexical_complexity_score = np.mean(lexical_complexity_scores) if lexical_complexity_scores else 0.0
        punctuation_score = np.mean(punctuation_scores) if punctuation_scores else 0.0
        semantic_score = np.mean(semantic_scores) if semantic_scores else 0.0
        
        # === WEIGHTS - High Semantic Weight (Option 2) ===
        # Semantic has HIGHEST weight to clearly show embedding model differences
        weights = {
            'function_words': 0.25,      # Still very reliable - unconscious patterns
            'semantic': 0.30,             # HIGHEST - embedding model performance comparison
            'syntactic': 0.20,            # Sentence structure
            'character_ngrams': 0.15,    # Unconscious habits
            'lexical': 0.05,              # Vocabulary complexity
            'punctuation': 0.05           # Formatting style
        }
        
        logger.info(f"[SCORING] FuncWords: {function_word_score:.3f}, Syntactic: {syntactic_score:.3f}, "
                   f"CharNgrams: {character_ngram_score:.3f}, Lexical: {lexical_complexity_score:.3f}, "
                   f"Punct: {punctuation_score:.3f}, Semantic: {semantic_score:.3f}")
        
        overall_score = (
            weights['function_words'] * function_word_score +
            weights['syntactic'] * syntactic_score +
            weights['character_ngrams'] * character_ngram_score +
            weights['lexical'] * lexical_complexity_score +
            weights['punctuation'] * punctuation_score +
            weights['semantic'] * semantic_score  # Include semantic so embedding choice matters
        )
        
        # Better confidence calculation
        # Penalize variance but boost if function words (most reliable) are strong
        score_list = [function_word_score, syntactic_score, character_ngram_score, lexical_complexity_score, punctuation_score]
        score_variance = np.var(score_list)
        confidence = max(0.0, 1.0 - score_variance)
        
        # Bonus confidence if function words (most reliable) are high
        if function_word_score > 0.7:
            confidence = min(1.0, confidence * 1.15)
        
        # Penalty if topic very different (but don't penalize too much - this is expected)
        if semantic_score < 0.2:
            logger.info(f"[SCORING] Low semantic similarity ({semantic_score:.3f}) - texts may be on different topics (expected for stylometry)")
            confidence *= 0.90  # Minimal penalty
        
        breakdown = {
            'function_word_similarity': function_word_score,
            'syntactic_similarity': syntactic_score,
            'character_ngram_similarity': character_ngram_score,
            'lexical_complexity_similarity': lexical_complexity_score,
            'punctuation_similarity': punctuation_score,
            'semantic_coherence': semantic_score,
            'corpus_size': len(corpus_vectors),
            'feature_variance': score_variance
        }

        # Check for topic mismatch warning
        topic_warning = None
        if semantic_score < 0.15:
            topic_warning = f"Warning: Very low semantic similarity ({semantic_score:.1%}). Texts are on very different topics."
        
        return SimilarityScore(
            overall_score=overall_score,
            function_word_score=function_word_score,
            syntactic_score=syntactic_score,
            character_ngram_score=character_ngram_score,
            lexical_complexity_score=lexical_complexity_score,
            punctuation_score=punctuation_score,
            topic_mismatch_warning=topic_warning,
            confidence=confidence,
            breakdown=breakdown
        )

    async def analyze_statement(self, task_id: str, statement: str, person_name: str):
        """
        Asynchronously analyze a statement and report granular progress.
        This is the main orchestrator for the analysis task.
        """
        try:
            # Stage 1: Initialization (0% -> 5%)
            await self._update_progress(task_id, 0, "Initializing analysis...")
            corpus = self._load_cached_corpus(person_name)

            if not corpus:
                raise ValueError(f"No corpus found for {person_name}. Build corpus first.")

            await self._update_progress(task_id, 5, f"Corpus loaded ({len(corpus)} documents).")

            # Stage 2: Style Vector Extraction (5% -> 50%)
            corpus_vectors = []
            corpus_vectors_by_platform = defaultdict(list)
            total_docs = len(corpus)
            for i, doc in enumerate(corpus):
                try:
                    # This part of the process will cover 45% of the progress bar
                    progress = 5 + ((i + 1) / total_docs) * 45
                    await self._update_progress(task_id, progress, f"Extracting style vector {i + 1}/{total_docs}")
                    
                    vector = await self.extract_style_vector(doc.content, doc.platform)
                    corpus_vectors.append(vector)
                    corpus_vectors_by_platform[doc.platform].append(vector)  # Cache by platform
                except Exception as e:
                    logger.warning(f"Skipping a document due to error during vector extraction: {e}")

            if not corpus_vectors:
                raise ValueError("Could not extract any valid style vectors from the corpus.")

            # Stage 3: Similarity Comparison (50% -> 95%)
            await self._update_progress(task_id, 50, "Comparing statement to corpus...")
            similarity = await self.compare_styles(statement, corpus_vectors, task_id, starting_progress=50.0)

            # Stage 4A: Platform Analysis (95% -> 97%)
            await self._update_progress(task_id, 95, "Analyzing by platform...")
            platform_analysis = await self._analyze_by_platform_cached(statement, corpus_vectors_by_platform)

            # Stage 4B: Temporal Analysis (97% -> 98%)  
            await self._update_progress(task_id, 97, "Computing temporal consistency...")
            temporal_analysis = self._temporal_consistency_analysis(corpus_vectors)

            # Stage 4C: Final Report Assembly (98% -> 100%)
            await self._update_progress(task_id, 98, "Assembling final report...")
            report = {
                'statement': statement[:200] + '...' if len(statement) > 200 else statement,
                'person_analyzed': person_name,
                'corpus_size': len(corpus),
                'analysis_timestamp': datetime.now().isoformat(),
                'similarity_scores': {
                    'overall_similarity': round(similarity.overall_score, 3),
                    'function_words': round(similarity.function_word_score, 3),  # Most reliable
                    'syntactic_patterns': round(similarity.syntactic_score, 3),
                    'character_patterns': round(similarity.character_ngram_score, 3),
                    'lexical_complexity': round(similarity.lexical_complexity_score, 3),
                    'punctuation_style': round(similarity.punctuation_score, 3),
                    'semantic_score': round(similarity.breakdown.get('semantic_coherence', 0.0), 3),  # Varies by embedding model
                    'confidence': round(similarity.confidence, 3),
                    'topic_warning': similarity.topic_mismatch_warning
                },
                'interpretation': self._interpret_similarity_score(similarity.overall_score),
                'detailed_breakdown': similarity.breakdown,
                'platform_analysis': platform_analysis,
                'temporal_analysis': temporal_analysis,
                'recommendations': self._generate_recommendations(similarity)
            }

            await self._update_progress(task_id, 100, "Analysis complete.", result=report)
        
        except Exception as e:
            logger.error(f"Error during analysis task {task_id}: {e}", exc_info=True)
            await self._update_progress(task_id, 100, f"Error: {str(e)}", result={'error': str(e)})

    async def _analyze_by_platform_cached(self, statement: str, corpus_vectors_by_platform: Dict[str, List]) -> Dict[str, float]:
        """
        Analyze similarity by platform using pre-computed vectors to avoid re-extraction.
        """
        platform_scores = {}
        
        for platform, platform_vectors in corpus_vectors_by_platform.items():
            if len(platform_vectors) < 3:
                continue
                
            # Use cached vectors - no more extract_style_vector calls!
            similarity = await self.compare_styles(statement, platform_vectors)
            platform_scores[platform] = similarity.overall_score
        
        return platform_scores

    # Helper methods for text processing
    
    def _extract_tweets_from_content(self, content: str, person_name: str) -> List[Tuple[str, Optional[datetime]]]:
        """Extract individual tweets from scraped content"""
        tweets = []
        
        # Common patterns for tweet extraction
        patterns = [
            r'"([^"]{10,280})"',  # Quoted tweets
            r'(?:^|\n)([^@\n]{10,280})(?:\n|$)',  # Line-based tweets
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                if len(match.strip()) > 10 and person_name.lower() in content.lower():
                    tweets.append((match.strip(), None))  # Date extraction would be more complex
        
        return tweets[:50]  # Limit results
    
    def _clean_speech_transcript(self, content: str) -> str:
        """Clean speech transcript content"""
        # Remove common transcript artifacts
        content = re.sub(r'\[.*?\]', '', content)  # Remove [APPLAUSE], [LAUGHTER], etc.
        content = re.sub(r'\(.*?\)', '', content)  # Remove stage directions
        content = re.sub(r'PRESIDENT.*?:', '', content)  # Remove speaker labels
        content = re.sub(r'Q\s+', '', content)  # Remove Q&A markers
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        
        return content.strip()
    
    def _clean_press_release(self, content: str) -> str:
        """Clean press release content"""
        # Remove boilerplate
        content = re.sub(r'FOR IMMEDIATE RELEASE.*?\n', '', content)
        content = re.sub(r'Contact:.*?\n', '', content)
        content = re.sub(r'###.*', '', content)
        
        return content.strip()
    
    def _extract_interview_content(self, content: str, person_name: str) -> str:
        """Extract interview responses from content"""
        # This is a simplified extraction - real implementation would be more sophisticated
        lines = content.split('\n')
        interview_lines = []
        
        for line in lines:
            # Look for lines that might be the person speaking
            if any(name_part.lower() in line.lower() for name_part in person_name.split()):
                interview_lines.append(line)
        
        return '\n'.join(interview_lines)
    
    def _extract_social_posts(self, content: str, person_name: str) -> List[Tuple[str, Optional[datetime]]]:
        """Extract social media posts from content"""
        # Similar to tweet extraction but more generic
        return self._extract_tweets_from_content(content, person_name)
    
    def _extract_relevant_content(self, content: str, person_name: str) -> str:
        """Extract content relevant to the person"""
        # Simple relevance extraction based on name proximity
        sentences = sent_tokenize(content)
        relevant_sentences = []
        
        for sentence in sentences:
            if any(name_part.lower() in sentence.lower() for name_part in person_name.split()):
                relevant_sentences.append(sentence)
        
        return ' '.join(relevant_sentences)
    
    def _extract_date_from_content(self, content: str) -> Optional[datetime]:
        """Extract date from content using regex patterns"""
        date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    from dateutil.parser import parse
                    return parse(match.group(1))
                except:
                    continue
        
        return None
    
    # Stylometric calculation methods
    
    def _calculate_yule_k(self, word_freq: Counter) -> float:
        """Calculate Yule's K measure of vocabulary diversity"""
        if not word_freq:
            return 0.0
        
        freq_of_freqs = Counter(word_freq.values())
        n = sum(word_freq.values())
        
        if n <= 1:
            return 0.0
        
        sum_freq_squared = sum(freq * (freq_count ** 2) for freq, freq_count in freq_of_freqs.items())
        yule_k = 10000 * (sum_freq_squared - n) / (n ** 2)
        
        return yule_k
    
    def _extract_dependency_patterns(self, doc) -> Dict[str, float]:
        """Extract dependency parsing patterns"""
        patterns = defaultdict(int)
        total_deps = 0
        
        for token in doc:
            if token.dep_:
                patterns[token.dep_] += 1
                total_deps += 1
        
        if total_deps == 0:
            return {}
        
        return {dep: count/total_deps for dep, count in patterns.items()}
    def _get_semantic_embeddings(self, text: str) -> List[float]:
        """Enhanced semantic embeddings with multiple model support"""

        # Try the explicitly set active model first
        if self.active_embedding_model and self.embedding_models.get(self.active_embedding_model, {}).get('enabled'):
            model_key = self.active_embedding_model
            try:
                # Unified logic for all models
                if model_key in ['gme', 'qwen3_8b', 'qwen3_4b']:
                    embeddings = self._get_model_manager_embeddings(text)
                elif model_key == 'jina_v3':
                    embeddings = self._get_jina_v3_embeddings(text)
                elif model_key == 'nomic_v1_5':
                    embeddings = self._get_nomic_v1_5_embeddings(text)
                elif model_key == 'arctic_embed':
                    embeddings = self._get_arctic_embeddings(text)
                elif model_key == 'bge_m3':
                    embeddings = self._get_bge_m3_embeddings(text)
                elif model_key in ['gte_qwen2', 'inf_retriever', 'sentence_t5', 'mxbai_large', 'multilingual_e5', 'frida']:
                    embeddings = self._get_sentence_transformers_embeddings(text, model_key)
                elif model_key == 'star':
                    embeddings = self._get_star_embeddings(text)
                elif model_key == 'roberta':
                    embeddings = self._get_roberta_embeddings(text)
                else:
                    logger.warning(f"Unknown model type: {model_key}")
                    embeddings = []

                if embeddings:
                    logger.debug(f"Using {model_key.upper()} embeddings: {len(embeddings)}D")
                    return embeddings
            except Exception as e:
                logger.warning(f"Active model {model_key} failed, falling back to priority list. Error: {e}")


        # Fallback to priority order if active model fails or isn't set
        sorted_models = sorted(
            self.embedding_models.items(), 
            key=lambda x: x[1].get('priority', 99)
        )

        for model_key, model_info in sorted_models:
            if not model_info.get('enabled', False):
                continue

            try:
                # Unified logic for all models
                if model_key in ['gme', 'qwen3_8b', 'qwen3_4b']:
                    embeddings = self._get_model_manager_embeddings(text)
                elif model_key == 'jina_v3':
                    embeddings = self._get_jina_v3_embeddings(text)
                elif model_key == 'nomic_v1_5':
                    embeddings = self._get_nomic_v1_5_embeddings(text)
                elif model_key == 'arctic_embed':
                    embeddings = self._get_arctic_embeddings(text)
                elif model_key == 'bge_m3':
                    embeddings = self._get_bge_m3_embeddings(text)
                elif model_key in ['gte_qwen2', 'inf_retriever', 'sentence_t5', 'mxbai_large', 'multilingual_e5', 'frida']:
                    embeddings = self._get_sentence_transformers_embeddings(text, model_key)
                elif model_key == 'star':
                    embeddings = self._get_star_embeddings(text)
                elif model_key == 'roberta':
                    embeddings = self._get_roberta_embeddings(text)
                else:
                    continue

                if embeddings:
                    logger.debug(f"Using fallback {model_key.upper()} embeddings: {len(embeddings)}D")
                    if len(embeddings) > 50000:  # Something is very wrong
                        logger.error(f"CRITICAL: Embedding dimension too large ({len(embeddings)}), using fallback")
                        return [0.0] * TARGET_EMBEDDING_DIM   
                    return embeddings

            except Exception as e:
                logger.warning(f"{model_key} embeddings failed: {e}")
                continue

        logger.warning("No embedding models available - using zeros")
        return [0.0] * TARGET_EMBEDDING_DIM
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Basic sentiment analysis"""
        # Simple word-based sentiment (could be enhanced with proper sentiment models)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst', 'disgusting']
        
        words = text.lower().split()
        pos_count = sum(words.count(word) for word in positive_words)
        neg_count = sum(words.count(word) for word in negative_words)
        total_words = len(words)
        
        return {
            'positive_ratio': pos_count / total_words if total_words > 0 else 0,
            'negative_ratio': neg_count / total_words if total_words > 0 else 0,
            'sentiment_polarity': (pos_count - neg_count) / total_words if total_words > 0 else 0
        }
    
    def _analyze_function_words(self, words: List[str]) -> Dict[str, float]:
        """Analyze function word usage patterns"""
        total_words = len(words)
        if total_words == 0:
            return {}
        
        ratios = {}
        for category, word_list in self.function_words.items():
            count = sum(words.count(word) for word in word_list)
            ratios[f'{category}_ratio'] = count / total_words
        
        return ratios
    
    # Similarity calculation methods
    
    def _calculate_function_word_similarity(self, target: StyleVector, corpus: StyleVector) -> float:
        """
        Calculate function word similarity - THE MOST RELIABLE metric for authorship.
        Function words are unconscious patterns that are very hard to fake.
        """
        scores = []
        
        # Function word frequencies (triple weight - most reliable)
        func_word_sim = self._calculate_distribution_similarity(
            target.function_word_frequencies, 
            corpus.function_word_frequencies
        )
        scores.extend([func_word_sim] * 3)
        
        # Function word bigrams (how function words combine)
        func_bigram_sim = self._calculate_distribution_similarity(
            target.function_word_bigrams,
            corpus.function_word_bigrams
        )
        scores.extend([func_bigram_sim] * 2)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_character_ngram_similarity(self, target: StyleVector, corpus: StyleVector) -> float:
        """
        Calculate character n-gram similarity - captures unconscious typing/writing patterns.
        Very reliable as these are subconscious habits.
        """
        scores = []
        
        # Character trigrams (most informative)
        # Compare top 50 most common trigrams
        common_trigrams_target = dict(sorted(target.char_trigrams.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:50])
        common_trigrams_corpus = dict(sorted(corpus.char_trigrams.items(), 
                                             key=lambda x: x[1], 
                                             reverse=True)[:50])
        trigram_sim = self._calculate_distribution_similarity(
            common_trigrams_target,
            common_trigrams_corpus
        )
        scores.extend([trigram_sim] * 2)  # Double weight
        
        # Character bigrams
        common_bigrams_target = dict(sorted(target.char_bigrams.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True)[:50])
        common_bigrams_corpus = dict(sorted(corpus.char_bigrams.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:50])
        bigram_sim = self._calculate_distribution_similarity(
            common_bigrams_target,
            common_bigrams_corpus
        )
        scores.append(bigram_sim)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_punctuation_similarity(self, target: StyleVector, corpus: StyleVector) -> float:
        """
        Calculate punctuation and formatting similarity.
        Punctuation habits are moderately reliable for authorship.
        """
        scores = []
        
        # Punctuation ratios (how often each punctuation mark is used)
        punct_ratio_sim = self._calculate_distribution_similarity(
            target.punctuation_ratios, 
            corpus.punctuation_ratios
        )
        scores.extend([punct_ratio_sim] * 2)  # Double weight
        
        # Punctuation sequences ("..", "!?", etc.)
        punct_seq_sim = self._calculate_distribution_similarity(
            target.punctuation_sequences,
            corpus.punctuation_sequences
        )
        scores.append(punct_seq_sim)
        
        # Formatting habits
        simple_features = [
            'capitalization_ratio', 'all_caps_ratio', 'contraction_ratio',
            'question_ratio', 'exclamation_ratio'
        ]
        for feature in simple_features:
            target_val = getattr(target, feature, 0.0)
            corpus_val = getattr(corpus, feature, 0.0)
            if max(target_val, corpus_val) > 0:
                score = 1.0 - abs(target_val - corpus_val) / max(target_val, corpus_val)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0

    def _calculate_lexical_similarity(self, target: StyleVector, corpus: StyleVector) -> float:
        """
        OVERHAULED: Calculate lexical similarity focusing on vocabulary complexity patterns.
        This measures writing sophistication, not topic.
        """
        scores = []
        
        # Basic lexical measures
        simple_features = ['avg_word_length', 'avg_sentence_length', 
                          'vocab_richness', 'hapax_legomena_ratio', 'yule_k']
        for feature in simple_features:
            target_val = getattr(target, feature, 0.0)
            corpus_val = getattr(corpus, feature, 0.0)
            if max(target_val, corpus_val) > 0:
                score = 1.0 - abs(target_val - corpus_val) / max(target_val, corpus_val)
                scores.append(score)
        
        # Word length distribution (more nuanced than just average)
        word_len_sim = self._calculate_distribution_similarity(
            target.word_length_distribution,
            corpus.word_length_distribution
        )
        scores.extend([word_len_sim] * 2)  # Double weight - very informative
        
        return np.mean(scores) if scores else 0.0

    def _calculate_syntactic_similarity(self, target: StyleVector, corpus: StyleVector) -> float:
        """
        OVERHAULED: Calculate syntactic similarity focusing on sentence structure patterns.
        Captures HOW someone structures sentences, not what they say.
        """
        scores = []
        
        # POS distribution (basic patterns)
        pos_sim = self._calculate_distribution_similarity(
            target.pos_distribution, 
            corpus.pos_distribution
        )
        scores.append(pos_sim)
        
        # POS bigrams (more specific structural patterns) - MOST RELIABLE
        pos_bigram_sim = self._calculate_distribution_similarity(
            target.pos_bigrams,
            corpus.pos_bigrams
        )
        scores.extend([pos_bigram_sim] * 2)  # Double weight - very reliable
        
        # POS trigrams (very specific patterns)
        pos_trigram_sim = self._calculate_distribution_similarity(
            target.pos_trigrams,
            corpus.pos_trigrams
        )
        scores.append(pos_trigram_sim)
        
        # Dependency patterns
        dep_sim = self._calculate_distribution_similarity(
            target.dependency_patterns,
            corpus.dependency_patterns
        )
        scores.append(dep_sim)
        
        # Clause patterns (subordinate vs coordinate)
        clause_sim = self._calculate_distribution_similarity(
            target.clause_patterns,
            corpus.clause_patterns
        )
        scores.append(clause_sim)
        
        # Sentence complexity
        if max(target.sentence_complexity, corpus.sentence_complexity) > 0:
            complexity_score = 1.0 - abs(target.sentence_complexity - corpus.sentence_complexity) / max(target.sentence_complexity, corpus.sentence_complexity)
            scores.append(complexity_score)
        
        # Sentence length variance (writing consistency)
        if max(target.sentence_length_variance, corpus.sentence_length_variance) > 0:
            variance_score = 1.0 - abs(target.sentence_length_variance - corpus.sentence_length_variance) / max(target.sentence_length_variance, corpus.sentence_length_variance)
            scores.append(variance_score)
            
        return np.mean(scores) if scores else 0.0


    
    def _calculate_semantic_similarity(self, vec1: List[float], vec2: List[float], dim1: int, dim2: int) -> float:
        """
        Calculates a stable, normalized similarity score based on the Euclidean distance
        between the original, unpadded dimensions of the vectors.
        This is more robust for stylometry than cosine similarity.
        """
        # If the original dimensions are different or zero, they are not comparable.
        if dim1 != dim2 or dim1 == 0:
            return 0.0

        try:
            import numpy as np

            # Slice the vectors to their original, meaningful length before comparison
            vec1_sliced = np.array(vec1[:dim1], dtype=np.float32)
            vec2_sliced = np.array(vec2[:dim1], dtype=np.float32)

            # --- NEW: Euclidean Distance Calculation ---
            # Calculate the straight-line distance between the two points in the embedding space.
            euclidean_distance = np.linalg.norm(vec1_sliced - vec2_sliced)

            # Normalize the distance to a 0-1 similarity score.
            # The "1 +" term prevents division by zero and ensures the score is bounded.
            # A smaller distance results in a higher similarity score.
            similarity = 1 / (1 + euclidean_distance)
            
            return float(similarity)

        except Exception as e:
            logger.error(f"Semantic similarity calculation error: {e}")
            return 0.0
    
 
    
    def _calculate_distribution_similarity(self, dist1: Dict[str, float], dist2: Dict[str, float]) -> float:
        """Calculate similarity between two distributions using a more stable intersection method."""
        if not dist1 or not dist2:
            return 0.0
        
        all_keys = set(dist1.keys()) | set(dist2.keys())
        if not all_keys:
            return 1.0 # If both are empty, they are perfectly similar
            
        score_list = []
        for key in all_keys:
            p = dist1.get(key, 0.0)
            q = dist2.get(key, 0.0)
            # Use max to avoid division by zero and provide stable normalization
            if max(p, q) > 0:
                score = 1.0 - (abs(p - q) / max(p, q))
                score_list.append(score)
            elif p == 0 and q == 0:
                score_list.append(1.0) # Both zero, perfect match for this feature
            
        return np.mean(score_list) if score_list else 0.0
    
    # Analysis and interpretation methods
    
    def _interpret_similarity_score(self, score: float) -> str:
        """Interpret similarity score"""
        if score >= 0.8:
            return "Very high similarity - likely same author"
        elif score >= 0.6:
            return "High similarity - probably same author"
        elif score >= 0.4:
            return "Moderate similarity - uncertain authorship"
        elif score >= 0.2:
            return "Low similarity - probably different author"
        else:
            return "Very low similarity - likely different author"
    
    async def _analyze_by_platform(self, statement: str, corpus: List[TextDocument]) -> Dict[str, float]:
        """
        Analyze similarity by platform. Now an async function to correctly call compare_styles.
        """
        platform_scores = {}
        platform_docs = defaultdict(list)
        
        for doc in corpus:
            platform_docs[doc.platform].append(doc)
        
        for platform, docs in platform_docs.items():
            if len(docs) < 3:
                continue
                
            platform_vectors = await asyncio.gather(*[
                self.extract_style_vector(doc.content, doc.platform) for doc in docs
            ])
            
            # --- FIX: Await the async compare_styles call ---
            # We don't need progress tracking here, so we don't pass a task_id.
            similarity = await self.compare_styles(statement, platform_vectors)
            platform_scores[platform] = similarity.overall_score
        
        return platform_scores
    
    def _temporal_consistency_analysis(self, corpus_vectors: List[StyleVector]) -> Dict[str, float]:
        """Analyze temporal consistency of writing style"""
        # This would require date information to be properly implemented
        # For now, return basic consistency metrics
        
        if len(corpus_vectors) < 5:
            return {'consistency': 0.5, 'note': 'Insufficient data for temporal analysis'}
        
        # Calculate variance in key features across corpus
        lexical_variance = np.var([v.avg_word_length for v in corpus_vectors])
        syntactic_variance = np.var([v.avg_sentence_length for v in corpus_vectors])
        
        # Low variance indicates consistency
        consistency = 1.0 / (1.0 + lexical_variance + syntactic_variance)
        
        return {
            'consistency': consistency,
            'lexical_variance': lexical_variance,
            'syntactic_variance': syntactic_variance
        }
    
    def _generate_recommendations(self, similarity: SimilarityScore) -> List[str]:
        """OVERHAULED: Generate recommendations based on stylometric analysis"""
        recommendations = []
        
        # Overall assessment
        if similarity.overall_score > 0.7 and similarity.confidence > 0.75:
            recommendations.append("✅ STRONG MATCH: High confidence authorship match across multiple stylometric features")
        elif similarity.overall_score > 0.5 and similarity.confidence > 0.6:
            recommendations.append("✓ PROBABLE MATCH: Moderate to high confidence - stylistic patterns align")
        elif similarity.overall_score < 0.3:
            recommendations.append("⚠️ WEAK MATCH: Low similarity - likely different author")
        
        # Confidence warnings
        if similarity.confidence < 0.5:
            recommendations.append("⚠️ Low confidence - consider building a larger corpus for more reliable results")
        
        # Function words (most reliable indicator)
        if similarity.function_word_score > 0.8:
            recommendations.append("💪 Strong function word similarity - highly reliable indicator of same author")
        elif similarity.function_word_score < 0.4:
            recommendations.append("⚠️ Function word patterns differ significantly - strong indicator of different author")
        
        # Syntactic patterns
        if similarity.syntactic_score > 0.7:
            recommendations.append("✓ Strong syntactic similarity - sentence structure patterns match")
        elif similarity.syntactic_score < 0.4:
            recommendations.append("⚠️ Sentence structure patterns differ - possible different author")
        
        # Character patterns
        if similarity.character_ngram_score > 0.75:
            recommendations.append("✓ Character-level patterns match - unconscious writing habits align")
        
        # Punctuation habits
        if similarity.punctuation_score > 0.7:
            recommendations.append("✓ Punctuation and formatting habits match")
        
        # Topic mismatch warning
        if similarity.topic_mismatch_warning:
            recommendations.append(f"ℹ️ {similarity.topic_mismatch_warning}")
            recommendations.append("Note: Low topic similarity is expected for cross-domain analysis")
        
        # Discrepancy warnings
        if similarity.function_word_score > 0.7 and similarity.syntactic_score < 0.4:
            recommendations.append("⚠️ MIXED SIGNALS: Function words match but syntax differs - investigate further")
        
        if similarity.overall_score < 0.5 and similarity.function_word_score > 0.6:
            recommendations.append("⚠️ Function words match despite low overall score - possible same author with different context")
        
        return recommendations
    
    # Caching methods
    
    def _cache_corpus(self, person_name: str, corpus: List[TextDocument]):
        """Cache corpus to disk"""
        cache_file = self.cache_dir / f"{person_name.lower().replace(' ', '_')}_corpus.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(corpus, f)
            logger.info(f"Cached corpus for {person_name} ({len(corpus)} documents)")
        except Exception as e:
            logger.error(f"Error caching corpus: {e}")
    
    def _load_cached_corpus(self, person_name: str) -> Optional[List[TextDocument]]:
        """Load cached corpus from disk"""
        cache_file = self.cache_dir / f"{person_name.lower().replace(' ', '_')}_corpus.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                corpus = pickle.load(f)
            logger.info(f"Loaded cached corpus for {person_name} ({len(corpus)} documents)")
            return corpus
        except Exception as e:
            logger.error(f"Error loading cached corpus: {e}")
            return None
    
    def _empty_style_vector(self) -> StyleVector:
        """Return an empty style vector for error cases"""
        return StyleVector(
            avg_word_length=0.0,
            avg_sentence_length=0.0,
            vocab_richness=0.0,
            hapax_legomena_ratio=0.0,
            yule_k=0.0,
            pos_distribution={},
            dependency_patterns={},
            sentence_complexity=0.0,
            punctuation_ratios={},
            capitalization_ratio=0.0,
            avg_paragraph_length=0.0,
            semantic_embeddings=[],
            sentiment_scores={},
            function_word_ratios={},
            modal_verb_usage=0.0,
            passive_voice_ratio=0.0,
            question_ratio=0.0,
            exclamation_ratio=0.0
        )

    def _get_model_manager_embeddings(self, text: str) -> List[float]:
        """Get embeddings from a model loaded via the ModelManager service."""
        try:
            model_info = self.model_manager.model_purposes.get('forensic_embeddings')
            if not model_info:
                logger.warning("No forensic_embeddings model is set in model_manager.")
                return []

            model_proxy = self.model_manager.get_model(model_info['name'], model_info['gpu_id'])

            # Use the .embed() method on the remote model proxy
            result = model_proxy.embed(text[:4096]) # Increased context for larger models
            
            # Handle both dict response (remote) and list response (direct GGUF)
            if isinstance(result, list):
                # Direct list response from GGUF model
                embeddings = result
            elif result and isinstance(result, dict) and result.get("status") == "success":
                # Dict response from remote model
                embeddings = result.get("embedding", [])
            else:
                # Error case
                error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else 'Invalid response'
                logger.warning(f"ModelManager remote embedding failed for {model_info['name']}: {error_msg}")
                return []
            
            # Flattening Logic: Handle cases where the model returns a list of lists
            if embeddings and len(embeddings) > 0 and isinstance(embeddings[0], list):
                embeddings = [item for sublist in embeddings for item in sublist]
            
            # Normalize embeddings for qwen3 models (critical for similarity tasks)
            if embeddings and model_info['name'] and 'qwen3' in model_info['name'].lower():
                import numpy as np
                embeddings_array = np.array(embeddings)
                norm = np.linalg.norm(embeddings_array)
                if norm > 0:
                    embeddings = (embeddings_array / norm).tolist()
                    logger.debug(f"Normalized qwen3 embedding vector (norm was {norm:.4f})")
            
            return embeddings

        except Exception as e:
            model_name = self.model_manager.model_purposes.get('forensic_embeddings', {}).get('name', 'unknown model')
            logger.warning(f"ModelManager embedding generation failed for {model_name}: {e}", exc_info=True)
            return []

    def set_active_embedding_model(self, model_key: str) -> bool:
        """Set the active embedding model to be used for analysis."""
        if model_key in self.embedding_models and self.embedding_models[model_key].get('enabled'):
            self.active_embedding_model = model_key
            logger.info(f"Active embedding model set to: {model_key}")
            return True
        logger.error(f"Cannot set active model to {model_key}, it is not enabled or does not exist.")
        return False

    def get_embedding_status(self) -> dict:
        """Get status of all available embedding models"""
        # Determine which model is currently active
        active_model = "None"
        if self.active_embedding_model and self.embedding_models.get(self.active_embedding_model, {}).get('enabled'):
            active_model = self.active_embedding_model.upper()
        else:
            # Fallback to priority if active is not set or enabled
            sorted_models = sorted(
                self.embedding_models.items(),
                key=lambda x: x[1].get('priority', 99)
            )
            for name, info in sorted_models:
                if info.get('enabled', False):
                    active_model = name.upper()
                    break

        return {
            "models": self.embedding_models,
            "active_model": active_model,
            "gme_enabled": self.embedding_models.get('gme', {}).get('enabled', False),
            "model_manager_available": self.model_manager is not None
        }



