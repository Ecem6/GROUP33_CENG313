# Jigsaw - Agile Community Rules Classification

**Course:** CENG313  
**Group:** 33

This repository contains our group's work for the **Jigsaw - Agile Community Rules Classification** Kaggle competition as part of the CENG313 course assignments.

## Team Members
*   **Member 1:** Ayşe Sena Nasır | Student ID: 23118080083 | GitHub: senanasir
*   **Member 2:** Ecem Uçlusoy | Student ID: 23118080067 | GitHub: Ecem6

---

## 1. Project Overview & Kaggle Competition
The goal of this project is to build a binary classifier that predicts whether a Reddit comment broke a specific subreddit rule. Unlike standard toxic comment classification, this task requires a context-aware approach, as community norms and specific rules vary heavily across different subreddits. 
* **Competition Link:** [Jigsaw - Agile Community Rules Classification](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview)

## 2. Dataset
The dataset consists of a small labeled set (`train.csv` - ~2029 records) and a larger unlabeled test set. 
*   **Key Features:** `body` (the text of the comment), `rule` (the text of the rule being evaluated), `subreddit` (the community).
*   **Target Variable:** `rule_violation` (0: No violation, 1: Violation).
*   **Characteristics:** The dataset is noisy (contains URLs, markdown, newlines) and has an imbalanced class distribution, necessitating rigorous text preprocessing and class-weighting strategies.

## 3. Our Approach
We follow a structured, modular pipeline divided into two main assignments:

**Assignment 1: EDA and Baseline Implementation**
*   **Data Cleaning:** Applied Regex to remove URLs, HTML tags, markdown, and special characters.
*   **Feature Engineering:** Concatenated the `clean_rule` and `clean_body` columns to create a context-aware textual feature (`combined_text`).
*   **Vectorization:** TF-IDF Vectorizer with `max_features=5000`.
*   **Baseline Model:** Logistic Regression using `class_weight='balanced'` to penalize majority class bias.

**Assignment 2: Proposed Solution**
*   *(To be updated in Assignment 2)*

## 4. Results
*   **Baseline Model (Logistic Regression + TF-IDF):** 
    *   **ROC-AUC Score:** 0.8241
    *   *(Assignment 2 results will be appended here).*

## 5. Repository Structure
Following the CENG313 guidelines, reusable codes are modularized inside the `src/` directory.

## 6. How to Run the Notebooks

### Local Execution
1. **Clone the repository:** Run `git clone https://github.com/Ecem6/GROUP33_CENG313.git` and navigate into the project folder.
2. **Environment setup:** Create a virtual environment using your preferred method (e.g., `python -m venv venv`) and activate it.
3. **Install dependencies:** Execute `pip install -r requirements.txt` to install required libraries like `pandas`, `scikit-learn`, and `matplotlib`.
4. **Data preparation:** Accept the competition rules on Kaggle and download the dataset. Extract the files and place them in a dedicated `data/` folder at the root of the project (ensuring paths like `data/train.csv` and `data/test.csv` are valid).
5. **Run the code:** Open `assignment1_eda_baseline.ipynb` in Jupyter Notebook/Lab, ensure your virtual environment is selected as the kernel, and execute the cells sequentially.

### Google Colab Execution
To run the project in the cloud without local setup, you can pull the repository directly into a Colab instance.
1. Open Google Colab and create a new notebook (or open the existing one).
2. In the very first cell, run the following commands to fetch the source code and change the working directory:
   ```python
   !git clone https://github.com/Ecem6/GROUP33_CENG313.git /content/GROUP33_CENG313
   %cd /content/GROUP33_CENG313

## 8. Kaggle Submission History
*   **Submission 1 (Baseline):** *(Pending - Test predictions to be uploaded).*
