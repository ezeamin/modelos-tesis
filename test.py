from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

from prepare_data import prepareTestData,prepareData,prepareTestRawData

test_data = prepareTestData()
test_data['Email Type'] = test_data['Email Type'].replace({'Safe Email': 1, 'Phishing Email': -1})

# Convert feature with TF-ID
# convert_feature = TfidfVectorizer()
convert_feature = load('tfidf/tfidf_vectorizer.joblib')

# training_data = prepareData()
# training_data['Email Type'] = training_data['Email Type'].replace({'Safe Email': 1, 'Phishing Email': -1})

# model = ComplementNB()
model = load('models/ComplementNB.joblib')

# training_X = convert_feature.fit_transform(training_data['Email Text'])
# training_Y = training_data['Email Type']
# model.fit(training_X, training_Y)

X = convert_feature.transform(test_data['Email Text'])
Y = test_data['Email Type']

print(Y)

pred = model.predict(X)

matrix = confusion_matrix(Y, pred)
sns.heatmap(matrix, annot=True, cmap='Blues', fmt='g', xticklabels=['Safe Email', 'Phishing Email'], yticklabels=['Safe Email', 'Phishing Email'])
plt.xlabel('Predicted labels')
plt.title('ComplementNB')
plt.show()

# ------------------------------------------------------------

emailBody = "Congratulations! you have won a brand new car. Claim it now at https://honda.xyz/login."
print(f'\nTest data: {emailBody}\n')

new_processed_data = [prepareTestRawData(emailBody)]
print(f'Processed test data: {new_processed_data[0]}\n')
    
pred = model.predict(convert_feature.transform(new_processed_data))
if pred[0] == -1:
    print(f'Predicted: Phishing text')
else:
    print(f'Predicted: Safe text')