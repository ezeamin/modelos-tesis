from sklearn.naive_bayes import ComplementNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from joblib import dump, load

from prepare_data import prepareData,prepareTestData

# Prepare data
loaded_dataset = prepareData()

# Convert the label
loaded_dataset['Email Type'] = loaded_dataset['Email Type'].replace({'Safe Email': 1, 'Phishing Email': -1})

models = [ComplementNB(), BernoulliNB(), MultinomialNB(),RandomForestClassifier(),GradientBoostingClassifier()]
model_names = ['ComplementNB', 'BernoulliNB', 'MultinomialNB', 'RandomForestClassifier', 'GradientBoostingClassifier']

accuracies = []
precisions = []
recalls = []
f1 = []

# Convert feature with TF-ID
convert_feature = TfidfVectorizer()

X = convert_feature.fit_transform(loaded_dataset['Email Text'])
Y = loaded_dataset['Email Type']

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=42)

# Calculate the performance
for model in models:
    # Try to load the model from file
    loadedFromFile = False
    try:
        model = load(f'dataset_2/models/{type(model).__name__}.joblib')
        print(f'Loaded model training from file for {type(model).__name__}')
        loadedFromFile = True
    except:
        print(f'No model found for {type(model).__name__}')
        model.fit(X_train, y_train)
            
    pred = model.predict(X_test)
    
    # Save the training to avoid refit
    if not loadedFromFile:
        dump(model, f'dataset_2/models/{type(model).__name__}.joblib')
    
    accuracies.append(accuracy_score(y_test, pred))
    precisions.append(precision_score(y_test, pred))
    recalls.append(recall_score(y_test, pred))
    f1.append(f1_score(y_test, pred))
    
    matrix = confusion_matrix(y_test, pred)
    sns.heatmap(matrix, annot=True, cmap='Blues', fmt='g', xticklabels=['Safe Email', 'Phishing Email'], yticklabels=['Safe Email', 'Phishing Email'])
    plt.xlabel('Predicted labels')
    plt.title(f'{type(model).__name__}')
    plt.show()

axis = np.arange(len(model_names))

plt.figure(figsize=(10, 6))
plt.bar(axis - 0.3, accuracies, 0.2, label = 'Accuracy') 
plt.bar(axis - 0.1, precisions, 0.2, label = 'Precisions')
plt.bar(axis + 0.1, recalls, 0.2, label = 'Recalls')
plt.bar(axis + 0.3, f1, 0.2, label = 'F1')

plt.xticks(axis, model_names) 
plt.xlabel("Model") 
plt.ylabel("Scores") 
plt.title("Models performance") 
plt.legend(title="Evaluation") 
plt.show() 
    
# Predict new data
emailBody = "Congratulations! you have won a brand new car. Claim it now at https://honda.xyz/login."
print(f'\nTest data: {emailBody}\n')

new_processed_data = [prepareTestData(emailBody)]
print(f'Processed test data: {new_processed_data[0]}\n')
    
for model in models:
    convert_feature = TfidfVectorizer()

    X = convert_feature.fit_transform(loaded_dataset['Email Text'])
    Y = loaded_dataset['Email Type']
    
    # TODO: This is loading a falsy model, fix it
    # Try to load trained model
    # try:
    #     model = load(f'dataset_2/models/{type(model).__name__}.joblib')
    # except:
    model.fit(X, Y)

    pred = model.predict(convert_feature.transform(new_processed_data))

    if pred[0] == -1:
       print(f'Predicted by {type(model).__name__}: Phishing text')
    else:
       print(f'Predicted by {type(model).__name__}: Safe text')