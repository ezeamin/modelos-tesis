import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

def preprocess_text(text):
    # Tokenization and Lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def load_or_preprocess_data():
    # Load or preprocess the dataset if needed
    data_file = 'preprocessed_data.csv'

    try:
        # Try loading preprocessed data if it exists
        data = pd.read_csv(data_file)
        print("Preprocessed data loaded.")
    except FileNotFoundError:
        # If preprocessed data doesn't exist, load raw data and preprocess it
        print("Preprocessed data not found. Preprocessing data...")

        # Load the raw dataset
        raw_data = pd.read_csv('datasets/CEAS_08.csv')

        # Data preprocessing
        # Remove duplicates
        raw_data = raw_data.drop_duplicates()

        # Handle missing values (if any)
        raw_data = raw_data.dropna()

        tqdm.pandas(desc="Processing text...")
        raw_data['processed_body'] = raw_data['body'].progress_apply(preprocess_text)
        raw_data['subject'] = raw_data['subject'].progress_apply(preprocess_text)
        
        # Remove rows that have empty or NaN 'processed_body' or 'subject'
        raw_data = raw_data[(raw_data['processed_body'] != '') & (raw_data['subject'] != '')]

        # Save preprocessed data to a file
        raw_data.to_csv(data_file, index=False)
        print("Preprocessed data saved.")

        # Assign preprocessed data to 'data' variable
        data = raw_data

    return data