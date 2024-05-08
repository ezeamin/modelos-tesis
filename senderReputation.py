import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import re

# Custom transformer for extracting sender reputation
class SenderReputationExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        sender_reputation = []
        for sender in tqdm(X['sender'], desc="Extracting sender reputation...", total=len(X)):
            # Example checks: 
            # 1. If sender domain is suspicious
            if re.search(r'(?:\b(?:phish|spoof)\b)|(?:\b(?:example|test)\b)', sender):
                sender_reputation.append(1)
            else:
                sender_reputation.append(0)
            # 2. If sender address contains random alphanumeric characters
            if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', sender):
                sender_reputation.append(0)
            else:
                sender_reputation.append(1)
        # Ensure that the length of sender_reputation matches the length of X
        assert len(sender_reputation) == len(X), "Length of sender_reputation does not match the length of X"
        return pd.DataFrame(sender_reputation, columns=['sender_reputation'])
