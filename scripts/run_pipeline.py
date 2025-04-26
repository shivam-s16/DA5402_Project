import os
import sys
import configparser
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to the path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from modules.scrape_top_stories import get_top_stories_link
from modules.extract_data import extract_headlines_images
from modules.database import create_table, insert_data, get_recent_news
from modules.deduplicate import check_similar_news
from modules.sentiment_analyzer import SentimentAnalyzer
from modules.logging_config import get_logger

# Load configuration
config = configparser.ConfigParser()
config.read(os.path.join(project_root, "config/config.ini"))

logger = get_logger(__name__)

def run_scraping_pipeline(save_images=False):
    """Run news scraping pipeline with sentiment analysis"""
    logger.info("Starting news scraping pipeline")
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Create database tables if they don't exist
    if not create_table():
        logger.error("Failed to create database tables")
        return False
    
    logger.info("Database tables created/verified")
    
    # Get the Top Stories link
    top_stories_url = get_top_stories_link()
    if not top_stories_url:
        logger.error("Failed to retrieve Top Stories link")
        return False
    
    logger.info(f"Scraping Top Stories from: {top_stories_url}")
    
    # Extract headlines and articles
    stories = extract_headlines_images(top_stories_url, image_save=save_images)
    
    if not stories:
        logger.error("No stories extracted")
        return False
    
    logger.info(f"Extracted {len(stories)} stories")
    
    # Process and insert stories into database
    processed_count = 0
    article_limit = int(config["SCRAPER"]["article_limit"])
    
    for headline, image_url, article_url, article_content, image_filename, article_date, article_time in stories:
        # Check for duplicates using semantic similarity
        if check_similar_news(headline):
            logger.info(f"Duplicate skipped: {headline}")
            continue
        
        # Analyze sentiment if model is available
        sentiment_label = None
        sentiment_score = None
        
        if analyzer.model is not None and analyzer.tokenizer is not None:
            # Use headline and content for sentiment analysis
            text_to_analyze = headline
            if article_content and len(article_content) > 0:
                text_to_analyze = headline + " " + article_content
                
            sentiment_label, sentiment_score = analyzer.predict_sentiment(text_to_analyze)
        
        # Insert into database
        if insert_data(headline, image_url, article_url, article_content, image_filename, 
                      article_date, article_time, sentiment_score, sentiment_label):
            processed_count += 1
            logger.info(f"Inserted: {headline} - Sentiment: {sentiment_label}")
        else:
            logger.error(f"Failed to insert: {headline}")
        
        # Stop after reaching the limit
        if processed_count >= article_limit:
            logger.info(f"Reached article limit of {article_limit}")
            break
    
    logger.info(f"Pipeline completed. Processed {processed_count} new articles")
    return processed_count > 0

def print_recent_news():
    """Print recent news with sentiment information"""
    articles = get_recent_news(limit=10)
    
    if not articles:
        print("No recent news articles found")
        return
    
    print("\nRECENT NEWS ARTICLES")
    print("=====================")
    
    for article in articles:
        id, headline, _, _, _, _, date, time, sentiment_score, sentiment_label, _ = article
        
        print(f"\nID: {id}")
        print(f"Headline: {headline}")
        print(f"Date/Time: {date} {time}")
        
        if sentiment_label:
            print(f"Sentiment: {sentiment_label} ({sentiment_score:.4f})")
        else:
            print("Sentiment: Not analyzed")
        
        print("---------------------")

if __name__ == "__main__":
    logger.info("=== NEWS SCRAPER PIPELINE STARTED ===")
    
    # Parse arguments
    save_images = "--save-images" in sys.argv
    show_recent = "--show-recent" in sys.argv
    
    # Run the pipeline
    success = run_scraping_pipeline(save_images=save_images)
    
    if success:
        logger.info("Pipeline executed successfully")
        
        if show_recent:
            print_recent_news()
    else:
        logger.error("Pipeline execution failed")
    
    logger.info("=== NEWS SCRAPER PIPELINE COMPLETED ===")