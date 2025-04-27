from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# --- Setup path so modules can be imported ---
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

from modules.scrape_top_stories import get_top_stories_link
from modules.extract_data import extract_headlines_images
from modules.database import create_table, insert_data
from modules.deduplicate import check_similar_news
from modules.sentiment_analyzer import SentimentAnalyzer
from modules.logging_config import get_logger

import configparser

# --- Load config ---
config = configparser.ConfigParser()
config.read(os.path.join(project_root, "config/config.ini"))

logger = get_logger(__name__)

# --- Default DAG args ---
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# --- Define DAG ---
with DAG(
    'news_scraper_pipeline',
    default_args=default_args,
    description='Pipeline to scrape news and store with sentiment analysis',
    schedule_interval='@hourly',  # or change to '@daily'
    start_date=datetime(2025, 4, 27),
    catchup=False,
    tags=['scraping', 'news'],
) as dag:

    def init_database():
        logger.info("Creating/verifying database tables...")
        if not create_table():
            raise Exception("Failed to create database tables")

    def scrape_and_store_news():
        logger.info("Running scraping logic...")
        analyzer = SentimentAnalyzer()

        top_stories_url = get_top_stories_link()
        if not top_stories_url:
            raise Exception("Failed to retrieve Top Stories link")

        stories = extract_headlines_images(top_stories_url, image_save=False)

        if not stories:
            raise Exception("No stories extracted")

        processed_count = 0
        article_limit = int(config["SCRAPER"]["article_limit"])

        for headline, image_url, article_url, article_content, image_filename, article_date, article_time in stories:
            if check_similar_news(headline):
                logger.info(f"Duplicate skipped: {headline}")
                continue

            sentiment_label = None
            sentiment_score = None

            if analyzer.model and analyzer.tokenizer:
                text_to_analyze = headline
                if article_content:
                    text_to_analyze = headline + " " + article_content

                sentiment_label, sentiment_score = analyzer.predict_sentiment(text_to_analyze)

            if insert_data(headline, image_url, article_url, article_content, image_filename,
                           article_date, article_time, sentiment_score, sentiment_label):
                processed_count += 1
                logger.info(f"Inserted: {headline} - Sentiment: {sentiment_label}")
            else:
                logger.error(f"Failed to insert: {headline}")

            if processed_count >= article_limit:
                logger.info(f"Reached article limit of {article_limit}")
                break

        logger.info(f"Processed {processed_count} articles.")

    # Define tasks
    task_init_db = PythonOperator(
        task_id='init_database',
        python_callable=init_database,
    )

    task_scrape_news = PythonOperator(
        task_id='scrape_and_store_news',
        python_callable=scrape_and_store_news,
    )

    # Task dependency
    task_init_db >> task_scrape_news
