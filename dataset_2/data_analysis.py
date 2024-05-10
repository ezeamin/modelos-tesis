import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from prepare_data import prepareData

# Read the CSV file
df = prepareData()

# Show dataset info
print(df.info())
    
# Plot the frequency of words based on the 'label' column
spam_word_frequency = df[df['Email Type'] == "Phishing Email"]['Email Text'].str.split().explode().value_counts().head(10)
non_spam_word_frequency = df[df['Email Type'] == "Safe Email"]['Email Text'].str.split().explode().value_counts().head(10)

plt.subplot(1, 2, 1)
spam_word_frequency.plot(kind='bar')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Frequency of Top 10 Words in Phishing Mails')

plt.subplot(1, 2, 2)
non_spam_word_frequency.plot(kind='bar')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Frequency of Top 10 Words in Non-Phishing Mails')

plt.tight_layout()
plt.show()

# Plot the distribution of information in the 'label' column in pie chart
df['Email Type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Email Types')
plt.show()

# Create a chart that correlates the frequency of "urgent, information, free" etc keywords with email type
# Create a list of words to search for in the email text
words = ['urgent', 'information', 'free', 'click', 'account', 'password', 'bank', 'money', 'offer', 'transaction']
word_counts = {word: [df[(df['Email Type'] == "Phishing Email") & (df['Email Text'].str.contains(word))]['Email Text'].count(), 
                      df[(df['Email Type'] == "Safe Email") & (df['Email Text'].str.contains(word))]['Email Text'].count()] 
               for word in words}

# Create a DataFrame from the word_counts dictionary
word_counts_df = pd.DataFrame(word_counts, index=['Phishing Email', 'Safe Email'])

# Plot the word counts
word_counts_df.plot(kind='bar')
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.title('Frequency of Words in Phishing and Non-Phishing Emails')
plt.show()

# Set the background color to white
wordcloud = WordCloud(width=800, height=800, background_color='white', colormap='hot')

# Create a word cloud for spam mails
spam_wordcloud = wordcloud.generate(' '.join(df[df['Email Type'] == "Phishing Email"]['Email Text']))
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Phishing Mails')
plt.show()

# Create a word cloud for non-spam mails
non_spam_wordcloud = wordcloud.generate(' '.join(df[df['Email Type'] == "Safe Email"]['Email Text']))
plt.imshow(non_spam_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Non-Phishing Mails')
plt.show()