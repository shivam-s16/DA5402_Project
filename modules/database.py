import psycopg2
import configparser
import logging
from datetime import datetime

# Load configuration
config = configparser.ConfigParser()
config.read("config/config.ini")

logger = logging.getLogger(__name__)

def connect_db():
    """Connect to PostgreSQL database"""
    try:
        return psycopg2.connect(
            host=config["DATABASE"]["host"],
            port=config["DATABASE"]["port"],
            user=config["DATABASE"]["user"],
            password=config["DATABASE"]["password"],
            dbname=config["DATABASE"]["dbname"]
        )
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        return None

def create_table():
    """Create necessary tables if they don't exist"""
    conn = connect_db()
    if not conn:
        return False
        
    cur = conn.cursor()
    
    try:
        # Create main news articles table
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {config["DATABASE"]["table_name"]} (
                id SERIAL PRIMARY KEY,
                headline TEXT UNIQUE,
                image_url TEXT,
                article_url TEXT,
                article_content TEXT,
                image_filename TEXT,
                article_date DATE,    
                article_time TIME,
                sentiment_score FLOAT,
                sentiment_label VARCHAR(10),
                scrape_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        
        # Verify table exists
        cur.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{config["DATABASE"]["table_name"].lower()}'
            )
        """)
        table_exists = cur.fetchone()[0]
        logger.info(f"Table '{config['DATABASE']['table_name']}' exists: {table_exists}")
        
        return table_exists
    
    except psycopg2.Error as e:
        logger.error(f"Failed to create table: {e}")
        return False
    
    finally:
        cur.close()
        conn.close()

def insert_data(headline, image_url, article_url, article_content, image_filename, 
                article_date, article_time, sentiment_score=None, sentiment_label=None):
    """Insert data into the database"""
    conn = connect_db()
    if not conn:
        return False
        
    try:
        cur = conn.cursor()
        table_name = config["DATABASE"]["table_name"]
        
        # Insert data with sentiment information
        cur.execute(f"""
            INSERT INTO {table_name} 
            (headline, image_url, article_url, article_content, image_filename, 
             article_date, article_time, sentiment_score, sentiment_label)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) 
            ON CONFLICT (headline) DO NOTHING
        """, (headline, image_url, article_url, article_content, image_filename, 
              article_date, article_time, sentiment_score, sentiment_label))
        
        conn.commit()
        rows_affected = cur.rowcount
        
        return rows_affected > 0
        
    except psycopg2.Error as e:
        logger.error(f"Failed to insert data: {e}")
        return False
        
    finally:
        conn.close()

def update_sentiment(headline, sentiment_score, sentiment_label):
    """Update sentiment information for an existing article"""
    conn = connect_db()
    if not conn:
        return False
        
    try:
        cur = conn.cursor()
        table_name = config["DATABASE"]["table_name"]
        
        cur.execute(f"""
            UPDATE {table_name}
            SET sentiment_score = %s, sentiment_label = %s
            WHERE headline = %s
        """, (sentiment_score, sentiment_label, headline))
        
        conn.commit()
        rows_affected = cur.rowcount
        
        return rows_affected > 0
        
    except psycopg2.Error as e:
        logger.error(f"Failed to update sentiment: {e}")
        return False
        
    finally:
        conn.close()

def get_unsentimented_articles(limit=100):
    """Get articles without sentiment analysis"""
    conn = connect_db()
    if not conn:
        return []
        
    try:
        cur = conn.cursor()
        table_name = config["DATABASE"]["table_name"]
        
        cur.execute(f"""
            SELECT id, headline, article_content
            FROM {table_name}
            WHERE sentiment_score IS NULL
            LIMIT %s
        """, (limit,))
        
        articles = cur.fetchall()
        return articles
        
    except psycopg2.Error as e:
        logger.error(f"Failed to fetch unsentimented articles: {e}")
        return []
        
    finally:
        conn.close()

def get_all_news_for_training():
    """Get all news articles for model training"""
    conn = connect_db()
    if not conn:
        return []
        
    try:
        cur = conn.cursor()
        table_name = config["DATABASE"]["table_name"]
        
        cur.execute(f"""
            SELECT id, headline, article_content, sentiment_label
            FROM {table_name}
            WHERE sentiment_label IS NOT NULL
        """)
        
        articles = cur.fetchall()
        return articles
        
    except psycopg2.Error as e:
        logger.error(f"Failed to fetch training articles: {e}")
        return []
        
    finally:
        conn.close()

def get_recent_news(limit=25):
    """Get recent news with sentiment information"""
    conn = connect_db()
    if not conn:
        return []
        
    try:
        cur = conn.cursor()
        table_name = config["DATABASE"]["table_name"]
        
        cur.execute(f"""
            SELECT id, headline, image_url, article_url, article_content, 
                   image_filename, article_date, article_time, 
                   sentiment_score, sentiment_label, scrape_timestamp
            FROM {table_name}
            ORDER BY scrape_timestamp DESC
            LIMIT %s
        """, (limit,))
        
        articles = cur.fetchall()
        return articles
        
    except psycopg2.Error as e:
        logger.error(f"Failed to fetch recent news: {e}")
        return []
        
    finally:
        conn.close()