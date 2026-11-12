import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_balance(df, target_col='rule_violation'):
    """Hedef değişkenin dağılımını çizer."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=target_col, hue=target_col, palette='viridis', legend=False)
    plt.title('Target Variable Balance')
    plt.xlabel('Rule Violation (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.show()

def save_submission(test_df, predictions, filename='submission.csv'):
    """Kaggle için submission CSV dosyasını oluşturur ve kaydeder."""
    submission_df = pd.DataFrame({
        'row_id': test_df['row_id'],
        'rule_violation': predictions
    })
    submission_df.to_csv(filename, index=False)
    print(f"Submission başarıyla '{filename}' olarak kaydedildi.")
