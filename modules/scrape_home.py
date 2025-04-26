
import requests
from bs4 import BeautifulSoup
import configparser
from modules.logging_config import get_logger

# Load configuration
config = configparser.ConfigParser()
config.read("config/config.ini")

logger = get_logger(__name__)

def scrape_home():
    """Scrape the Google News homepage"""
    url = config["SCRAPER"]["base_url"]
    logger.info(f"Scraping homepage: {url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        logger.info("Successfully scraped homepage")
        return soup
        
    except requests.RequestException as e:
        logger.error(f"Failed to scrape homepage: {e}")
        return None

