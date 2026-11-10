import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_and_split_data(df, test_size=0.2, random_state=42):
    # Modelin anlaması için temizlenmiş kural ve yorumu birleştiriyoruz
    df['combined_text'] = df['clean_rule'] + " " + df['clean_body']
    X = df['combined_text']
    y = df['rule_violation']
    
    # Eğitim ve doğrulama setlerini ayırma
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_val, y_train, y_val
