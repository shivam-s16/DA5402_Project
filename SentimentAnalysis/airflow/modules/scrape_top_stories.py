from configparser import ConfigParser
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from modules.logging_config import setup_logging
logger = setup_logging()

def get_top_stories_link():
    """
    Fetch Google News Top Stories URL
    Returns:
        str: URL of Google News Top Stories section
    """
    config = ConfigParser()
    config.read('config/config.ini')
    base_url = config.get('SCRAPER', 'base_url')
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        response = requests.get(base_url, headers=headers, timeout=100)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        relative_top_stories_url = soup.find('a', id=config.get('SELECTORS', 'top_stories_link_id'))['href']
        top_stories_url = urljoin(config.get('SCRAPER', 'base_url'), relative_top_stories_url)
        logger.info(f"Constructed Top Stories URL: {top_stories_url}")
        # Verify URL accessibility
        response = requests.head(top_stories_url, timeout=10)
        response.raise_for_status()
        return top_stories_url
    except requests.RequestException as e:
        logger.error(f"Failed to access Top Stories URL: {e}")
        raise