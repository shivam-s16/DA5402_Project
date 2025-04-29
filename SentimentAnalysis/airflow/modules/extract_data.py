"""
Module for extracting article data from Google News
"""
from configparser import ConfigParser
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from datetime import datetime

from modules.logging_config import setup_logging
logger = setup_logging()

def extract_data(url, image_save=False):
    """
    Extract article data from Google News page
    Args:
        url (str): URL of news page to scrape
        image_save (bool): Whether to download images
    Returns:
        list: List of tuples containing article data
    """
    config = ConfigParser()
    config.read('config/config.ini')
    
    try:
        logger.info(f"Starting extraction from {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        response = requests.get(url, timeout=100, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = []
        
        # Load selectors from config
        article_class = config.get('SELECTORS', 'article_class')
        article_elements = soup.find_all('article', class_=article_class)
        
        if not article_elements:
            logger.warning("No articles found with the specified selectors")
            return articles
            
        logger.info(f"Found {len(article_elements)} articles")
        
        for article in article_elements:
            try:
                headline = article.find(config.get('SELECTORS', 'headline_path'), class_=config.get('SELECTORS', 'headline_class')).text.strip()
                relative_link = article.select_one(config.get('SELECTORS', 'link_path'))['href']
                link = urljoin(config.get('SCRAPER', 'base_url'), relative_link)
                relative_thumbnail = article.select_one(config.get('SELECTORS', 'thumbnail_path'))['src'] if article.select_one(config.get('SELECTORS', 'thumbnail_path')) else None
                thumbnail = urljoin(config.get('SCRAPER', 'base_url'), relative_thumbnail)
                pub_date_str = article.select_one(config.get('SELECTORS', 'pub_date_path'))['datetime']
                pub_date = datetime.strptime(pub_date_str, '%Y-%m-%dT%H:%M:%SZ')

                articles.append({
                    'headline': headline,
                    'link': link,
                    'thumbnail': thumbnail,
                    'pub_date': pub_date
                })
                
            except Exception as e:
                logger.error(f"Error processing article: {e}")
                continue
                
        logger.info(f"Successfully processed {len(articles)} articles")
        return articles
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise