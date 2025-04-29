import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from configparser import ConfigParser
from typing import Dict, Optional

from modules.logging_config import setup_logging
logger = setup_logging()

class DatabaseHandler:
    def __init__(self, config_path='config/config.ini'):
        self.config = ConfigParser()
        self.config.read(config_path)
        self.connection = None
        self.cursor = None

    def connect(self) -> bool:
        """Establish connection to PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(
                host=self.config.get('DATABASE', 'host'),
                database=self.config.get('DATABASE', 'dbname'),
                user=self.config.get('DATABASE', 'user'),
                password=self.config.get('DATABASE', 'password'),
                port=self.config.get('DATABASE', 'port')
            )
            self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            logger.info("Database connection established successfully.")
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            return False

    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed.")

    def create_table(self) -> bool:
        """Create the news_articles table if it doesn't exist."""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS news_articles (
                id SERIAL PRIMARY KEY,
                headline TEXT NOT NULL,
                pub_date DATE,
                thumbnail_url TEXT,
                article_url TEXT UNIQUE,
                content_hash TEXT UNIQUE,
                sentiment TEXT,
                sentiment_score FLOAT,
                positive_feedback INTEGER DEFAULT 0,
                neutral_feedback INTEGER DEFAULT 0,
                negative_feedback INTEGER DEFAULT 0,
                categories TEXT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_content_hash ON news_articles(content_hash);
            CREATE INDEX IF NOT EXISTS idx_sentiment ON news_articles(sentiment);
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("news_articles table created or verified successfully.")
            return True
        except Exception as e:
            logger.error(f"Error creating table: {str(e)}")
            self.connection.rollback()
            return False

    def insert_article(self, article_data: Dict) -> Optional[int]:
        """
        Insert a news article into the database.
        Returns article id or None if duplicate.
        """
        try:
            insert_query = """
            INSERT INTO news_articles 
            (headline, pub_date, thumbnail_url, article_url, content_hash, sentiment, sentiment_score, categories)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (content_hash) DO NOTHING
            RETURNING id;
            """
            self.cursor.execute(insert_query, (
                article_data['headline'],
                article_data.get('pub_date'),
                article_data.get('thumbnail_url'),
                article_data.get('article_url'),
                article_data['content_hash'],
                article_data.get('sentiment'),
                article_data.get('sentiment_score', 0.0),
                article_data.get('categories', [])
            ))
            result = self.cursor.fetchone()
            self.connection.commit()

            if result:
                logger.info(f"Inserted article with ID: {result['id']}")
                return result['id']
            else:
                logger.info("Article already exists (duplicate detected).")
                return None

        except Exception as e:
            logger.error(f"Error inserting article: {str(e)}")
            self.connection.rollback()
            return None

    def update_feedback(self, article_id: int, feedback_type: str) -> bool:
        """Update feedback counters for an article."""
        if feedback_type not in ['positive', 'neutral', 'negative']:
            logger.error(f"Invalid feedback type: {feedback_type}")
            return False
        try:
            update_query = sql.SQL("""
            UPDATE news_articles
            SET {feedback_column} = {feedback_column} + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """).format(
                feedback_column=sql.Identifier(f"{feedback_type}_feedback")
            )
            self.cursor.execute(update_query, (article_id,))
            self.connection.commit()
            logger.info(f"Updated {feedback_type} feedback for article {article_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating feedback: {str(e)}")
            self.connection.rollback()
            return False

    def check_article_exists(self, content_hash: str) -> bool:
        """Check if an article with the given hash already exists."""
        try:
            self.cursor.execute(
                "SELECT id FROM news_articles WHERE content_hash = %s",
                (content_hash,)
            )
            return self.cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking article existence: {str(e)}")
            return False

    def get_health_metrics(self) -> Dict:
        """Get database health metrics for monitoring."""
        metrics = {}
        try:
            # Count total articles
            self.cursor.execute("SELECT COUNT(*) as total FROM news_articles")
            metrics['total_articles'] = self.cursor.fetchone()['total']

            # Count articles by sentiment
            self.cursor.execute("""
                SELECT sentiment, COUNT(*) as count 
                FROM news_articles 
                GROUP BY sentiment
            """)
            metrics['sentiment_distribution'] = {row['sentiment']: row['count'] 
                                               for row in self.cursor.fetchall()}

            # Get latest article date
            self.cursor.execute("""
                SELECT MAX(created_at) as latest_article 
                FROM news_articles
            """)
            metrics['latest_article'] = self.cursor.fetchone()['latest_article']

            return metrics
        except Exception as e:
            logger.error(f"Error getting health metrics: {str(e)}")
            return {'error': str(e)}
