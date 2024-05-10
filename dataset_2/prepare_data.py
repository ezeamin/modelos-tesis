import pandas as pd
from nltk.corpus import stopwords
import re

def prepareData():
    df = pd.read_csv('datasets/Phishing_Email.csv')
    
    # Remove the 'Unnamed: 0' column
    df = df.drop(columns=['Unnamed: 0'])

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove empty or null values
    df = df.dropna()

    # From Email Text element, retrieve only text and ignore any mail structure (html for example, but remove the tags and any special symbol)
    df['Email Text'] = df['Email Text'].astype(str).str.replace(r'<[^>]*>', '').apply(lambda x: re.sub(r'[^\w\s]', '', x)).str.replace('_', '').str.lower()

    # Remove stopwords and numbers
    stop_words = stopwords.words('english')
    df['Email Text'] = df['Email Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words and not word.isdigit()]))

    return df