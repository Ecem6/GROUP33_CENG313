from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def train_baseline_model(X_train, y_train):
    """Trains the Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    """Calculates and prints the model's ROC AUC score."""
    y_val_proba = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_val_proba)
    print(f"Validation ROC AUC Score: {auc_score:.4f}")
    return auc_score

