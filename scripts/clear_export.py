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

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

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



def clear_database():
    """Clear the current database table"""
    conn = connect_db()
    if not conn:
        logger.error("Failed to connect to database")
        return False
    
    try:
        cur = conn.cursor()
        table_name = config["DATABASE"]["table_name"]
        
        # Drop the table
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        logger.info(f"Dropped table {table_name}")
        
        
        # Save updated config
        with open(os.path.join(project_root, "config/config.ini"), 'w') as configfile:
            config.write(configfile)
        
        
        
        # Create new table with updated name
        create_table()
        
        return True
    
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        return False
    finally:
        conn.close()

def export_and_clear_database():
    """Export all data from database to CSV and clear the database"""
    # Create exports directory if it doesn't exist
    exports_dir = os.path.join(project_root, "data/exports")
    create_directory_if_not_exists(exports_dir)
    
    # Get current timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(exports_dir, f"news_data_export_{timestamp}.csv")
    
    # Get all articles from database
    articles, column_names = get_all_articles_from_db()
    
    if articles is None or len(articles) == 0:
        logger.warning("No articles found in database or error occurred")
        return False
    
    # Create DataFrame
    df = pd.DataFrame(articles, columns=column_names)
    
    # Analyze missing sentiment if any
    #df = analyze_missing_sentiment(df)
    
    # Save to CSV
    df.to_csv(csv_file, index=False)
    logger.info(f"Exported {len(df)} articles to {csv_file}")
    
    # Clear database and update table name
    if clear_database():
        logger.info("Database cleared and table name updated")
        return True
    else:
        logger.error("Failed to clear database")
        return False

if __name__ == "__main__":
    logger.info("Starting database export and clear process")
    
    if export_and_clear_database():
        logger.info("Database export and clear completed successfully")
    else:
        logger.error("Database export and clear process failed")