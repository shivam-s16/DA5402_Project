import logging
import hashlib
from sentence_transformers import SentenceTransformer, util
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from modules.logging_config import setup_logging
logger = setup_logging()

class Deduplicator:
    def __init__(self, db_handler, similarity_threshold: float = 0.8, 
                 enable_content_check: bool = True,
                 min_headline_length: int = 15):
        """
        Deduplicator that checks both in-memory and database for duplicates.

        Args:
            db_handler: Instance of DatabaseHandler, already connected.
            similarity_threshold: Cosine similarity threshold for semantic deduplication.
            enable_content_check: Whether to check content similarity in addition to headline.
            min_headline_length: Minimum length for headlines to consider for similarity.
        """
        self.db_handler = db_handler
        self.similarity_threshold = similarity_threshold
        self.enable_content_check = enable_content_check
        self.min_headline_length = min_headline_length
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info(f"Deduplicator initialized (threshold={similarity_threshold}, content_check={enable_content_check})")

    def generate_content_hash(self, article: Dict) -> str:
        """Generate a SHA-256 hash from headline, url, content, and date."""
        headline = article.get('headline', '').strip().lower()
        url = article.get('article_url', '').strip().lower()
        date = article.get('pub_date', '')
        hash_content = f"{headline}|{url}|{date}"
        return hashlib.sha256(hash_content.encode()).hexdigest()

    def clean_text(self, text: str) -> str:
        """Normalize text for similarity comparison."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts using transformer embeddings."""
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(emb1, emb2)[0][0])

    def fetch_db_articles_in_window(self, pub_date: datetime.date) -> List[Dict]:
        """
        Fetch articles from the DB within ±3 days of pub_date.
        Returns a list of dicts with at least: headline, content_hash, pub_date.
        """
        date_from = (pub_date - timedelta(days=3)).strftime('%Y-%m-%d')
        date_to = (pub_date + timedelta(days=3)).strftime('%Y-%m-%d')
        query = """
            SELECT headline, content_hash, pub_date
            FROM news_articles
            WHERE pub_date BETWEEN %s AND %s
        """
        self.db_handler.cursor.execute(query, (date_from, date_to))
        return self.db_handler.cursor.fetchall()

    def is_duplicate(self, article: Dict, batch_articles: List[Dict]) -> bool:
        """
        Check if article is a duplicate in batch (in-memory) or database (persistent).
        """
        article_hash = self.generate_content_hash(article)
        headline = article.get('headline', '')
        pub_date = article.get('pub_date')
        if isinstance(pub_date, str):
            try:
                pub_date = datetime.strptime(pub_date, '%Y-%m-%d').date()
            except Exception:
                pub_date = datetime.now().date()
        elif not pub_date:
            pub_date = datetime.now().date()

        # 1. Check against batch articles (in-memory)
        for existing in batch_articles:
            if existing.get('content_hash') == article_hash:
                logger.debug(f"Exact hash match (batch) for '{headline[:30]}...'")
                return True

            existing_headline = existing.get('headline', '')
            if len(headline) >= self.min_headline_length and len(existing_headline) >= self.min_headline_length:
                sim = self.semantic_similarity(headline, existing_headline)
                if sim >= self.similarity_threshold:
                    logger.info(f"Batch headline similarity {sim:.2f} for '{headline[:30]}...' and '{existing_headline[:30]}...'")
                    return True


        # 2. Check against DB articles in ±3 day window
        db_articles = self.fetch_db_articles_in_window(pub_date)
        for existing in db_articles:
            if existing.get('content_hash') == article_hash:
                logger.debug(f"Exact hash match (DB) for '{headline[:30]}...'")
                return True

            existing_headline = existing.get('headline', '')
            if len(headline) >= self.min_headline_length and len(existing_headline) >= self.min_headline_length:
                sim = self.semantic_similarity(headline, existing_headline)
                if sim >= self.similarity_threshold:
                    logger.info(f"DB headline similarity {sim:.2f} for '{headline[:30]}...' and '{existing_headline[:30]}...'")
                    return True

        return False

    def deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Deduplicate a batch of articles against both the batch and the database.
        Returns a list of unique articles (with content_hash added).
        """
        unique_articles = []
        total_duplicates = 0

        for article in articles:
            article['content_hash'] = self.generate_content_hash(article)
            if not self.is_duplicate(article, unique_articles):
                unique_articles.append(article)
            else:
                total_duplicates += 1

        logger.info(f"Deduplication complete: {len(unique_articles)} unique articles from {len(articles)} total ({total_duplicates} duplicates)")
        return unique_articles
