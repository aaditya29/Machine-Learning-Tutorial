import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Ensure the logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')  # Setting the logging level to DEBUG

# Handler to output logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')  # Setting console handler to DEBUG level
# File handler to write logs to a file
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
# Setting file handler to DEBUG level
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')  # Setting file handler to DEBUG level

formatter = logging.Formatter(
    # Defining the log message format
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Applying the format to console handler
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)  # Applying the format to file handler

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words(
        'english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)


def preprocess_df(df, text_column='text', target_column='target'):
    try:
        logger.debug('Starting preprocessing for DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        # Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df

    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise


def main(text_column='text', target_column='target'):
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv(
            '/Users/adityamishra/Documents/Machine-Learning-Tutorial/7-MLOps/2-Complete-Pipeline/raw/train.csv')
        test_data = pd.read_csv(
            '/Users/adityamishra/Documents/Machine-Learning-Tutorial/7-MLOps/2-Complete-Pipeline/raw/test.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_df(
            train_data, text_column, target_column)
        test_processed_data = preprocess_df(
            test_data, text_column, target_column)  # Processing train and test data

        # Store the data inside interim folder
        data_path = os.path.join(
            '/Users/adityamishra/Documents/Machine-Learning-Tutorial/7-MLOps/2-Complete-Pipeline', "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(
            data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(
            data_path, "test_processed.csv"), index=False)

        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error(
            'Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
