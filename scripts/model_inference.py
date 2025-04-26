import os
import sys
import pandas as pd
import configparser
from datetime import datetime
from pathlib import Path

# Add the project root to the path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from modules.sentiment_analyzer import SentimentAnalyzer
from modules.database import connect_db, create_table
from modules.logging_config import get_logger

# Load configuration
config = configparser.ConfigParser()
config.read(os.path.join(project_root, "config/config.ini"))

logger = get_logger(__name__)

def get_all_articles_from_db():
    """Get all articles from the database"""
    conn = connect_db()
    if not conn:
        logger.error("Failed to connect to database")
        return None, None
        
    try:
        cur = conn.cursor()
        table_name = config["DATABASE"]["table_name"]
        
        # Get all columns except image_filename (which we'll discard)
        cur.execute(f"""
            SELECT id, headline, image_url, article_url, article_content, 
                   article_date, article_time, sentiment_score, sentiment_label, scrape_timestamp
            FROM {table_name}
        """)
        
        # Fetch all data
        articles = cur.fetchall()
        logger.info(f"Retrieved {len(articles)} articles from database")
        
        # Get column names
        column_names = [desc[0] for desc in cur.description]
        
        return articles, column_names

    except Exception as e:
        logger.error(f"Error retrieving articles: {e}")
        return None, None
    finally:
        conn.close()

def analyze_and_update_missing_sentiment():
    """Analyze sentiment for articles that don't have sentiment data and update in database"""
    # Get articles from database
    articles_data, column_names = get_all_articles_from_db()
    if not articles_data or not column_names:
        logger.error("No articles retrieved from database")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(articles_data, columns=column_names)
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Skip if model not available
    if analyzer.model is None or analyzer.tokenizer is None:
        logger.error("Sentiment analyzer model not available")
        return
    
    # Find rows without sentiment
    missing_sentiment = df['sentiment_label'].isna()
    missing_count = missing_sentiment.sum()
    
    if missing_count > 0:
        logger.info(f"Analyzing sentiment for {missing_count} articles")
        
        # Connect to database for updating
        conn = connect_db()
        if not conn:
            logger.error("Failed to connect to database for updates")
            return
            
        try:
            cur = conn.cursor()
            table_name = config["DATABASE"]["table_name"]
            
            for idx in df[missing_sentiment].index:
                article_id = df.loc[idx, 'id']
                headline = df.loc[idx, 'headline']
                content = df.loc[idx, 'article_content']
                
                # Use headline and content for sentiment analysis
                text_to_analyze = headline
                if content and not pd.isna(content) and len(content) > 0:
                    text_to_analyze = headline + " " + content
                    
                sentiment_label, sentiment_score = analyzer.predict_sentiment(text_to_analyze)
                sentiment_score = float(sentiment_score)
                article_id = int(df.loc[idx, 'id'])
                if sentiment_label is not None and sentiment_score is not None:
                    # Update the database with the new sentiment values
                    cur.execute(f"""
                        UPDATE {table_name}
                        SET sentiment_label = %s, sentiment_score = %s
                        WHERE id = %s
                    """, (sentiment_label, sentiment_score, article_id))
                    
                    conn.commit()
                    logger.info(f"Updated sentiment for article ID {article_id}: {headline}")
            
            logger.info(f"Successfully updated sentiment for {missing_count} articles")
            
        except Exception as e:
            logger.error(f"Error updating sentiment in database: {e}")
        finally:
            conn.close()
    else:
        logger.info("No articles found with missing sentiment data")

if __name__ == "__main__":
    logger.info("Starting sentiment analysis process")
    analyze_and_update_missing_sentiment()
    logger.info("Sentiment analysis process completed")
