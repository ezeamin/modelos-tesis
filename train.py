from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion, Pipeline
from joblib import dump, load

from data import load_or_preprocess_data 
from itemSelector import ItemSelector
# from senderReputation import SenderReputationExtractor

# Load or preprocess the dataset
data = load_or_preprocess_data()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['processed_body', 'sender', 'subject', 'urls']], data['label'], test_size=0.2, random_state=42)

# Normalize values? Should be done in load_or_preprocess_data()

# Build a pipeline for feature extraction and model training
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_features', Pipeline([
            ('selector', ItemSelector(key='processed_body')),
            ('vectorizer', CountVectorizer())
        ])),
        # ('sender_reputation', SenderReputationExtractor())
    ])),
    ('classifier', MultinomialNB())
])

# Try to load the model from file
model = load('model.joblib')
hasLoadedModel = False
if(model):
    print("Loaded model training from file")
    hasLoadedModel = True
    pipeline = model;

# If not loaded, train the model
if(not hasLoadedModel):
    pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model to file if it was not loaded
if(not hasLoadedModel):
    dump(pipeline, 'model.joblib')
