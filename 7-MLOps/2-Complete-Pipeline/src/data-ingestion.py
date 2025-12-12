import os
import yaml
import logging  # Setting up logging for the data ingestion process
import pandas as pd
from sklearn.model_selection import train_test_split


log_dir = 'logs'  # Directory to store log files
os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists

# making object for logging configuration
logger = logging.getLogger('data_ingestion')
# Setting the logging level to DEBUG to capture all types of log messages
logger.setLevel('DEBUG')

# Handler to output logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')  # Setting console handler to DEBUG level

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')  # Setting file handler to DEBUG level

formatter = logging.Formatter(
    # Defining the log message format
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Applying the format to console handler
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)  # Applying the format to file handler

logger.addHandler(console_handler)  # Adding console handler to the logger
logger.addHandler(file_handler)  # Adding file handler to the logger
