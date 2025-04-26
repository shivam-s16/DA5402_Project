<<<<<<< HEAD
import logging
import configparser
import os

# Load configuration
config = configparser.ConfigParser()
config.read("config/config.ini")

# Ensure log directory exists
log_dir = os.path.dirname(config["LOGGER"]["log_file"])
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(
    filename=config["LOGGER"]["log_file"],
    level=getattr(logging, config["LOGGER"]["log_level"]),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_logger(name):
    """Returns a logger instance with the given name"""
=======
import logging
import configparser
import os

# Load configuration
config = configparser.ConfigParser()
config.read("config/config.ini")

# Ensure log directory exists
log_dir = os.path.dirname(config["LOGGER"]["log_file"])
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(
    filename=config["LOGGER"]["log_file"],
    level=getattr(logging, config["LOGGER"]["log_level"]),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_logger(name):
    """Returns a logger instance with the given name"""
>>>>>>> 865e3dee350745261eab842079e5aca439e51963
    return logging.getLogger(name)