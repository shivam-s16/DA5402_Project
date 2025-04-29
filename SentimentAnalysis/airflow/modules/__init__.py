# Import key modules for easier access
from .database import DatabaseHandler
from .scrape_top_stories import get_top_stories_link
from .sentiment_analyzer import SentimentAnalyzer
from .deduplicate import Deduplicator
from .extract_data import extract_data
from .logging_config import setup_logging

# Version info
__version__ = '0.1.0'