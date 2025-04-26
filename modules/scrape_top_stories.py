import configparser
from modules.scrape_home import scrape_home
from modules.logging_config import get_logger

# Load configuration
config = configparser.ConfigParser()
config.read("config/config.ini")

logger = get_logger(__name__)

def get_top_stories_link():
    """Extracts the dynamic 'Top Stories' link from the homepage"""
    soup = scrape_home()
    if not soup:
        logger.error("Failed to load Google News homepage.")
        return None
    
    logger.info("Extracting 'Top Stories' link...")
    
    selector = config["SCRAPER"]["top_stories_selector"]
    top_story_link = soup.select_one(selector)
    
    if not top_story_link:
        logger.error(f"Could not find 'Top Stories' link using selector: {selector}")
        return None
    
    if "href" in top_story_link.attrs:
        full_link = "https://news.google.com" + top_story_link["href"]
        logger.info(f"Extracted 'Top Stories' link: {full_link}")
        return full_link
    else:
        logger.error("Extracted element has no 'href' attribute.")
        return None