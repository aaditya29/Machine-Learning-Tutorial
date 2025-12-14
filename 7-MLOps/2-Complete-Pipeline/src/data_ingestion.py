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


# loading data from csv file
def load_data(data_url: str) -> pd.DataFrame:
    try:
        # Loading data from the specified CSV file URL
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Error parsing CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3',
                'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


# splitting data into train and test sets
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')  # Path to save raw data
        # Ensure the raw data directory exists
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(
            raw_data_path, "train.csv"), index=False)  # Saving training data to CSV
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)


def main():
    try:
        test_size = 0.2  # Defining the test size for train-test split
        data_path = 'https://raw.githubusercontent.com/aaditya29/Machine-Learning-Tutorial/refs/heads/main/7-MLOps/2-Complete-Pipeline/experiments/spam.csv'  # URL to the dataset
        df = load_data(data_url=data_path)  # Loading the data
        final_df = preprocess_data(df)  # Preprocessing the data
        train_data, test_data = train_test_split(
            # Splitting the data into training and testing sets
            final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data,
                  # Saving the split data
                  data_path='/Users/adityamishra/Documents/Machine-Learning-Tutorial/7-MLOps/2-Complete-Pipeline')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
