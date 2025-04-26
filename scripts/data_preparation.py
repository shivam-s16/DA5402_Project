import os
import pandas as pd
import configparser
import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from modules.database import get_all_news_for_training
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

def merge_csv_files(csv_directory, output_file):
    """Merge multiple CSV files into one"""
    logger.info(f"Looking for CSV files in: {csv_directory}")
    
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    
    if not csv_files:
        logger.warning(f"No CSV files found in {csv_directory}")
        return False
    
    logger.info(f"Found {len(csv_files)} CSV files to merge")
    
    # Read and combine all CSV files
    all_dfs = []
    for file in csv_files:
        file_path = os.path.join(csv_directory, file)
        try:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
            logger.info(f"Read file: {file} with {len(df)} rows")
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
    
    if not all_dfs:
        logger.error("No valid CSV data found")
        return False
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save to output file
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Saved merged data to {output_file} with {len(combined_df)} rows")
    
    return True

def get_data_from_database():
    """Get training data from database"""
    logger.info("Fetching news articles from database for training")
    
    articles = get_all_news_for_training()
    
    if not articles:
        logger.warning("No articles found in database with sentiment labels")
        return None
    
    # Convert to dataframe
    df = pd.DataFrame(articles, columns=['article_id', 'headline', 'content', 'sentiment_label'])
    
    logger.info(f"Retrieved {len(df)} articles from database")
    return df

def prepare_training_data():
    """Prepare data for model training"""
    # Create necessary directories
    processed_dir = os.path.join(project_root, "data/processed")
    create_directory_if_not_exists(processed_dir)
    
    # Output file path
    output_file = os.path.join(processed_dir, "training_data.csv")
    
    # Try to get data from database first
    df = get_data_from_database()
    
    # If no data in database, try to merge CSV files
    if df is None or len(df) == 0:
        csv_directory = os.path.join(project_root, "data")
        if not merge_csv_files(csv_directory, output_file):
            logger.error("Failed to prepare training data")
            return None
        
        # Read the merged file
        df = pd.read_csv(output_file)
    else:
        # Save database data to CSV
        df.to_csv(output_file, index=False)
    
    logger.info(f"Training data prepared with {len(df)} samples")
    return output_file

if __name__ == "__main__":
    logger.info("Starting data preparation process")
    prepared_file = prepare_training_data()
    
    if prepared_file:
        logger.info(f"Data preparation complete. File saved to: {prepared_file}")
    else:
        logger.error("Data preparation failed")