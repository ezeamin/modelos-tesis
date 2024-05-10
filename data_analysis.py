import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import re

# Read the CSV file
df = pd.read_csv('datasets/Phishing_Email.csv')

# Log the first row's body with label 1
# print(df.loc[df['label'] == 1, 'body'].iloc[7])

print(df.head())

# Remove duplicates
df = df.drop_duplicates()

# From body element, retrieve only text and ignore any mail structure (html for example, but remove the tags and any special symbol)
df['body'] = df['body'].str.replace(r'<[^>]*>', '').apply(lambda x: re.sub(r'[^\w\s]', '', x)).str.lower()

# Remove stopwords and numbers
stop_words = stopwords.words('english')
df['body'] = df['body'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words and not word.isdigit()]))

# Plot the distribution of words in the 'body' column
# Calculate the frequency of each word in the 'body' column
word_frequency = df['body'].str.split().explode().value_counts().head(10)

# Plot the frequency of words
word_frequency.plot(kind='bar')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Frequency of Top 10 Words in Body')
plt.show()

# Plot the frequency of words based on the 'label' column
spam_word_frequency = df[df['label'] == 1]['body'].str.split().explode().value_counts().head(10)
non_spam_word_frequency = df[df['label'] == 0]['body'].str.split().explode().value_counts().head(10)

plt.subplot(1, 2, 1)
spam_word_frequency.plot(kind='bar')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Frequency of Top 10 Words in Spam Mails')

plt.subplot(1, 2, 2)
non_spam_word_frequency.plot(kind='bar')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Frequency of Top 10 Words in Non-Spam Mails')

plt.tight_layout()
plt.show()

# Plot the distribution of information in the 'label' column
df['label'].value_counts().plot(kind='bar')
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.xticks([0, 1], ['Not Phishing', 'Phishing'], rotation=0)  # Modify the x-axis labels
plt.show()

# Set the background color to white
wordcloud = WordCloud(background_color='white')

# Create a word cloud for spam mails
spam_wordcloud = wordcloud.generate(' '.join(df[df['label'] == 1]['body']))
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Spam Mails')
plt.show()

# Create a word cloud for non-spam mails
non_spam_wordcloud = wordcloud.generate(' '.join(df[df['label'] == 0]['body']))
plt.imshow(non_spam_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Non-Spam Mails')
plt.show()