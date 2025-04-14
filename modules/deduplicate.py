
import configparser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from modules.database import connect_db
from modules.logging_config import get_logger
from huggingface_hub import hf_hub_download

# Load configuration
config = configparser.ConfigParser()
config.read("config/config.ini")

logger = get_logger(__name__)

# Load BERT model
try:
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    logger.info("Successfully loaded SentenceTransformer model")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {e}")
    model = None

def get_existing_headlines():
    """Fetch all existing headlines from the database"""
    conn = connect_db()
    if not conn:
        return []
        
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT headline FROM {config['DATABASE']['table_name']}")
        headlines = [row[0] for row in cur.fetchall()]
        conn.close()
        return headlines
    except Exception as e:
        logger.error(f"Error fetching headlines: {e}")
        if conn:
            conn.close()
        return []

def get_existing_embeddings():
    """Fetch headlines and compute their embeddings"""
    headlines = get_existing_headlines()
    
    if not headlines:
        return [], []
    
    if model is None:
        logger.error("SentenceTransformer model not available")
        return headlines, []
    
    try:
        embeddings = model.encode(headlines)
        return headlines, embeddings
    except Exception as e:
        logger.error(f"Error computing embeddings: {e}")
        return headlines, []

def check_similar_news(new_headline, threshold=0.72):
    """Check if a similar news headline exists using BERT embeddings"""
    if not new_headline:
        return False
    
    if model is None:
        logger.warning("SentenceTransformer model not available, skipping similarity check")
        return False
    
    headlines, embeddings = get_existing_embeddings()
    
    if not headlines or len(embeddings) == 0:
        logger.info("No existing headlines to compare against")
        return False
    
    try:
        new_embedding = model.encode([new_headline])
        similarity_scores = cosine_similarity(new_embedding, embeddings)[0]
        
        is_similar = any(score >= threshold for score in similarity_scores)
        
        if is_similar:
            logger.info(f"Found similar headline to: {new_headline}")
        
        return is_similar
    except Exception as e:
        logger.error(f"Error in similarity check: {e}")
        return False