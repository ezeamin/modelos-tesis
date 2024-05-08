import pandas as pd
from data import preprocess_text 
from joblib import load

# Load model from file
model = load('model.joblib')

# Predict over 1 email stored as csv in example.csv and print output
example_sender = "Gaddafi Aisha <gaddafiaisha6643@gmail.com>"
example_subject = "HELLO DEAR"
example_body = """Dear Friend,

How are you doing today, I came across your e-mail contact prior a private search while in need of your assistance.

Please May i use this medium to open a mutual communication with you, and seeking your acceptance towards investing in your country under your management as my partner.

My name is Aisha Gaddafi am presently living in Oman as a refugee, i am a Widow and single Mother with One Daughter, the only biological Daughter of late Libyan President (Late Colonel Muammar Gaddafi) i am presently  under political asylum protection by the Omani Government in oman.

I have funds worth "Twentyseven Million Five Hundred Thousand United state dollar" ($27.500.000.00 USD) which i want to entrust on you for investment project in your country and  i shall compensate you 50% of the total sum after the funds have been transfered into your account in your country.
If you are willing to handle this project on my behalf, kindly reply urgent to enable me provide you more details to start the transfer process.

I shall appreciate your urgent response through my private email address below:
(aisha.gaddaffi28@mail.com)

Thanks
Yours Truly
Aisha Gaddaf"""

# Preprocess the example email
example_processed_body = preprocess_text(example_body)
example_processed_subject = preprocess_text(example_subject)
example_processed_sender = preprocess_text(example_sender)

# Create a DataFrame from the example email
example_email = pd.DataFrame({
    'processed_body': [example_processed_body],
    'sender': [example_processed_sender],
    'subject': [example_processed_subject],
    'urls': [1]
})

# Predict the label of the example email
example_prediction = model.predict(example_email)

# Print the prediction
print("Prediction for the example email:")
print("PHISHING" if example_prediction[0] == 1 else "Not Phishing")