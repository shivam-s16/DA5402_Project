from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os
from pathlib import Path

# Setup path for module imports
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from modules.database import DatabaseHandler
from modules.scrape_top_stories import get_top_stories_link
from modules.extract_data import extract_data
from modules.deduplicate import Deduplicator
from modules.sentiment_analyzer import SentimentAnalyzer
from modules.logging_config import setup_logging

import configparser

logger = setup_logging()

config = configparser.ConfigParser()
config.read(os.path.join(project_root, "config/config.ini"))

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'news_scraper_pipeline_modular',
    default_args=default_args,
    description='Pipeline to scrape news and store with sentiment analysis, modular nodes',
    schedule_interval='@hourly',
    start_date=datetime(2025, 4, 29),
    catchup=False,
    tags=['scraping', 'news'],
) as dag:

    def init_database(**kwargs):
        db = DatabaseHandler()
        if not db.connect():
            raise Exception("Failed to connect to database")
        if not db.create_table():
            raise Exception("Failed to create/verify database table")
        db.close()
        logger.info("Database initialized and table checked.")

    def get_news_url(**kwargs):
        url = get_top_stories_link()
        if not url:
            raise Exception("Failed to fetch top stories URL")
        kwargs['ti'].xcom_push(key='news_url', value=url)
        logger.info(f"Fetched news URL: {url}")

    def extract_articles(**kwargs):
        ti = kwargs['ti']
        url = ti.xcom_pull(task_ids='get_news_url', key='news_url')
        articles = extract_data(url, image_save=False)
        if not articles:
            logger.warning("No articles extracted.")
        ti.xcom_push(key='extracted_articles', value=articles)
        logger.info(f"Extracted {len(articles)} articles.")

    def deduplicate_articles(**kwargs):
        ti = kwargs['ti']
        articles = ti.xcom_pull(task_ids='extract_articles', key='extracted_articles')
        db = DatabaseHandler()
        if not db.connect():
            raise Exception("Failed to connect to database for deduplication")
        deduplicator = Deduplicator(db)
        unique_articles = deduplicator.deduplicate_articles(articles)
        db.close()
        ti.xcom_push(key='unique_articles', value=unique_articles)
        logger.info(f"Deduplicated to {len(unique_articles)} unique articles.")

    def analyze_sentiment(**kwargs):
        ti = kwargs['ti']
        articles = ti.xcom_pull(task_ids='deduplicate_articles', key='unique_articles')
        analyzer = SentimentAnalyzer()
        for article in articles:
            # Use dummy get_sentiment for now
            sentiment_result = analyzer.get_sentiment(article.get('headline', '') + ' ' + article.get('description', ''))
            article['sentiment'] = sentiment_result['sentiment']
            article['sentiment_score'] = sentiment_result['score']
        ti.xcom_push(key='scored_articles', value=articles)
        logger.info(f"Sentiment analyzed for {len(articles)} articles.")

    def store_articles(**kwargs):
        ti = kwargs['ti']
        articles = ti.xcom_pull(task_ids='analyze_sentiment', key='scored_articles')
        db = DatabaseHandler()
        if not db.connect():
            raise Exception("Failed to connect to database for storing articles")
        inserted = 0
        for article in articles:
            data = {
                'headline': article.get('headline'),
                'pub_date': article.get('pub_date'),
                'thumbnail_url': article.get('thumbnail'),
                'article_url': article.get('link'),
                'content_hash': article.get('content_hash'),
                'sentiment': article.get('sentiment'),
                'sentiment_score': article.get('sentiment_score'),
                'categories': article.get('categories', []),
            }
            if db.insert_article(data):
                inserted += 1
        db.close()
        logger.info(f"Inserted {inserted} articles into the database.")

    # Node tasks
    task_init_db = PythonOperator(
        task_id='init_database',
        python_callable=init_database
    )
    task_get_url = PythonOperator(
        task_id='get_news_url',
        python_callable=get_news_url
    )
    task_extract = PythonOperator(
        task_id='extract_articles',
        python_callable=extract_articles
    )
    task_deduplicate = PythonOperator(
        task_id='deduplicate_articles',
        python_callable=deduplicate_articles
    )
    task_sentiment = PythonOperator(
        task_id='analyze_sentiment',
        python_callable=analyze_sentiment
    )
    task_store = PythonOperator(
        task_id='store_articles',
        python_callable=store_articles
    )

    # DAG dependencies
    task_init_db >> task_get_url >> task_extract >> task_deduplicate >> task_sentiment >> task_store