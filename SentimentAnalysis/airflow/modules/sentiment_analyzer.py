import random

from modules.logging_config import setup_logging
logger = setup_logging()

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the dummy sentiment analyzer."""
        self.sentiments = ['positive', 'neutral', 'negative']
        logger.info("Dummy SentimentAnalyzer initialized (random output).")

    def get_sentiment(self, text):
        """Randomly pick a sentiment and assign a random score."""
        sentiment = random.choice(self.sentiments)
        return {
            'sentiment': sentiment,
            'score':round(random.uniform(0.4, 1.0), 2)
        }

    