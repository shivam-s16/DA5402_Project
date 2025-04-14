import requests
from bs4 import BeautifulSoup
import logging
import os
import re
from datetime import datetime
import configparser
from modules.logging_config import get_logger

# Load configuration
config = configparser.ConfigParser()
config.read("config/config.ini")

logger = get_logger(__name__)

def sanitize_filename(filename):
    """Sanitize a filename by removing invalid characters"""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', filename)

def extract_article_content(article_url):
    """Extract the main content from an article URL"""
    try:
        response = requests.get(article_url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find the main article content
        # This is a simplified approach - real-world extraction may need more complex logic
        content = ""
        
        # Look for common article containers
        article_element = soup.find('article') or soup.find('div', class_=re.compile(r'article|content|story'))
        
        if article_element:
            # Extract paragraphs
            paragraphs = article_element.find_all('p')
            content = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        if not content:
            # Fallback: get all paragraphs from the page
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        return content
        
    except Exception as e:
        logger.error(f"Failed to extract article content: {e}")
        return ""

def extract_headlines_images(url, image_save=False, save_dir="images"):
    """Extract headlines and image URLs from the top stories page"""
    stories = []
    article_limit = int(config["SCRAPER"]["article_limit"])
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        base_url = soup.find('base').get('href') if soup.find('base') else url
        
        if image_save and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Find all article elements
        article_count = 0
        for item in soup.find_all('article'):
            if article_count >= article_limit:
                break
                
            # Skip articles without figures if we need images
            if image_save and not item.find('figure'):
                continue
            
            # Extract image URL
            image_url = None
            image_filename = "No image"
            
            if item.find('figure'):
                img_tag = item.find('figure').find('img')
                image_url = base_url + img_tag['src'][1:] if img_tag and 'src' in img_tag.attrs else None
            
            # Extract headline and article URL
            links = item.find_all('a')
            headline = links[1].get_text(strip=True) if len(links) > 1 else "No headline"
            article_url = base_url + links[1]['href'] if len(links) > 1 and 'href' in links[1].attrs else "No URL"
            
            # Extract article date and time
            date_tag = item.find('time')
            article_date = date_tag['datetime'].split('T')[0] if date_tag and 'datetime' in date_tag.attrs else datetime.now().strftime('%Y-%m-%d')
            article_time = date_tag['datetime'].split('T')[1] if date_tag and 'datetime' in date_tag.attrs else datetime.now().strftime('%H:%M:%S')
            
            # Extract article content
            article_content = extract_article_content(article_url) if article_url != "No URL" else ""
            
            # Process image if needed
            if image_save and image_url:
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                sanitized_headline = sanitize_filename(headline)
                image_filename = f"{save_dir}/{sanitized_headline}_{timestamp}.jpg"
                
                try:
                    # Save image
                    img_response = requests.get(image_url)
                    if img_response.status_code == 200:
                        with open(image_filename, 'wb') as f:
                            f.write(img_response.content)
                except Exception as e:
                    logger.error(f"Failed to save image: {e}")
                    image_filename = "No image"
            
            # Add to stories list
            stories.append((headline, image_url, article_url, article_content, image_filename, article_date, article_time))
            article_count += 1
    
    except requests.RequestException as e:
        logger.error(f"Failed to fetch page: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in extract_headlines_images: {e}")
    
    return stories