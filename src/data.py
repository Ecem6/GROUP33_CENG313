import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(train_path, test_path):
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def clean_text(text):
    """Converts text to lowercase and removes URLs and punctuation."""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_and_vectorize(train_df, test_df=None, max_features=5000):
     """Cleans, combines, and converts texts into TF-IDF vectors."""
    
    train_df['text_combined'] = train_df['body'].apply(clean_text) + " " + train_df['rule'].apply(clean_text)
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(train_df['text_combined'])
    y_train = train_df['rule_violation']
    
    
    if test_df is not None:
        test_df['text_combined'] = test_df['body'].apply(clean_text) + " " + test_df['rule'].apply(clean_text)
        X_test_vec = vectorizer.transform(test_df['text_combined'])
        return X_train_vec, y_train, X_test_vec, vectorizer
        
    return X_train_vec, y_train, vectorizer
