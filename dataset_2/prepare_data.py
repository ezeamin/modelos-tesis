import pandas as pd
from nltk.corpus import stopwords
import re

stop_words = stopwords.words('english')

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
    df['Email Text'] = df['Email Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words and not word.isdigit()]))

    # Oversample the dataset to match amount between phishing and non-phishing emails
    phishing_emails = df[df['Email Type'] == "Phishing Email"]
    non_phishing_emails = df[df['Email Type'] == "Safe Email"]
    
    phishing_emails = phishing_emails.sample(len(non_phishing_emails), replace=True)
    
    df = pd.concat([phishing_emails, non_phishing_emails])

    return df

def prepareTestData(emailBody):
    # Prepare the email body
    emailBody = re.sub(r'<[^>]*>', '', emailBody).replace('_', '').lower()
    emailBody = ' '.join([word for word in emailBody.split() if word not in stopwords.words('english') and not word.isdigit()])
    
    return emailBody