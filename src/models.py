from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def train_baseline(X_train, y_train, max_features=5000):
    # Metni vektörleştirme
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Modeli kurma ve eğitme
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    return vectorizer, model

def evaluate_baseline(vectorizer, model, X_val, y_val):
    # Doğrulama seti üzerinde tahmin yapma
    X_val_tfidf = vectorizer.transform(X_val)
    y_pred = model.predict(X_val_tfidf)
    y_pred_proba = model.predict_proba(X_val_tfidf)[:, 1]

  
    print("--- Sınıflandırma Raporu ---")
    print(classification_report(y_val, y_pred))
    print(f"ROC-AUC Skoru: {roc_auc_score(y_val, y_pred_proba):.4f}")

