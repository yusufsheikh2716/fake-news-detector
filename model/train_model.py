"""
train_model.py
==============
This script trains a Fake News Detection model using:
  - TF-IDF (Term Frequency - Inverse Document Frequency) for text vectorization
  - Logistic Regression for binary classification (Real=1, Fake=0)

Run this script ONCE before starting the Flask app:
    python model/train_model.py

Outputs:
    model/model.pkl       - Trained Logistic Regression classifier
    model/vectorizer.pkl  - Fitted TF-IDF vectorizer (must use same one at prediction time)
"""

import os
import re
import string
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# ─────────────────────────────────────────────
# STEP 1: LOAD THE DATASET
# ─────────────────────────────────────────────

def load_dataset(filepath):
    """
    Load the labeled news dataset from a CSV file.

    The CSV must have two columns:
        - 'text'  : the news headline or article text
        - 'label' : 1 = Real news, 0 = Fake news

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe with 'text' and 'label' columns.
    """
    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)

    # Strip leading/trailing whitespace from column names (common CSV issue)
    df.columns = df.columns.str.strip()

    print(f"[INFO] Dataset loaded: {len(df)} rows")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts()}\n")

    return df


# ─────────────────────────────────────────────
# STEP 2: TEXT PREPROCESSING
# ─────────────────────────────────────────────

# Stopwords are very common words (like "the", "is", "and") that appear in almost
# every sentence but carry NO useful information for classification.
# Removing them helps TF-IDF focus on meaningful words.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "each", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "just", "as", "if",
    "up", "out", "about", "into", "through", "during", "before", "after",
    "that", "this", "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "we", "you", "i", "my", "your", "his", "her", "our",
    "who", "which", "what", "when", "where", "how", "all", "any", "also"
}


def preprocess_text(text):
    """
    Clean and normalize a piece of text before feeding it to the model.

    Steps performed (in order):
        1. Lowercase  - so 'News' and 'news' are treated as the same word
        2. Remove URLs - URLs don't help classify real vs fake
        3. Remove punctuation - commas, periods etc. add noise
        4. Remove digits - numbers rarely help in fake news detection
        5. Remove extra whitespace - keep text tidy
        6. Remove stopwords - drop common filler words

    Args:
        text (str): Raw input text string.

    Returns:
        str: Cleaned text string.
    """

    # 1. Convert to lowercase so the model treats 'SHOCKING' same as 'shocking'
    text = text.lower()

    # 2. Remove URLs (http://... or https://...) — they appear in articles but are not useful signals
    text = re.sub(r'http\S+|www\S+', '', text)

    # 3. Remove punctuation using Python's string.punctuation set
    #    e.g., removes: ! ? . , : ; ' " ( ) etc.
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 4. Remove standalone numbers — "123" doesn't mean real or fake
    text = re.sub(r'\b\d+\b', '', text)

    # 5. Remove extra spaces that may have appeared after above removals
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Remove stopwords — split into individual words, filter, then rejoin
    words = text.split()
    filtered_words = [word for word in words if word not in STOPWORDS]
    cleaned_text = ' '.join(filtered_words)

    return cleaned_text


def preprocess_dataframe(df):
    """
    Apply text preprocessing to every row in the dataset.

    Args:
        df (pd.DataFrame): DataFrame with 'text' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'cleaned_text' column.
    """
    print("[INFO] Preprocessing text data...")

    # Apply our cleaning function to every row in the 'text' column
    df['cleaned_text'] = df['text'].astype(str).apply(preprocess_text)

    # Drop rows where cleaning left us with empty strings
    df = df[df['cleaned_text'].str.strip() != '']

    print(f"[INFO] Preprocessing complete. {len(df)} rows remaining.\n")
    return df


# ─────────────────────────────────────────────
# STEP 3: SPLIT DATA
# ─────────────────────────────────────────────

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    We use 80% of data to train the model and hold out 20% to evaluate it.
    random_state=42 ensures results are reproducible (same split every time).

    Args:
        df (pd.DataFrame): DataFrame with 'cleaned_text' and 'label' columns.
        test_size (float): Fraction of data to use for testing (default 20%).
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
               X = feature (text), y = label (0 or 1)
    """
    # X = the input features (news text), y = the target (real or fake label)
    X = df['cleaned_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y   # stratify ensures both train and test have same Real/Fake ratio
    )

    print(f"[INFO] Training set size : {len(X_train)} samples")
    print(f"[INFO] Test set size     : {len(X_test)} samples\n")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# STEP 4: VECTORIZE TEXT (TF-IDF)
# ─────────────────────────────────────────────

def vectorize_text(X_train, X_test, max_features=5000):
    """
    Convert raw text into numerical feature vectors using TF-IDF.

    TF-IDF stands for Term Frequency - Inverse Document Frequency:
        - TF  (Term Frequency)        : How often a word appears in THIS document
        - IDF (Inverse Doc Frequency) : How rare the word is across ALL documents

    Words that appear often in one article but rarely overall get HIGH scores.
    Common words like 'the', 'is' that appear everywhere get LOW scores.

    We fit the vectorizer ONLY on training data (to avoid data leakage),
    then transform both train and test data using the same vocabulary.

    Args:
        X_train (pd.Series): Training text data.
        X_test  (pd.Series): Test text data.
        max_features (int): Limit vocabulary to top N words by TF-IDF score.

    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, vectorizer)
    """
    print(f"[INFO] Vectorizing text using TF-IDF (max_features={max_features})...")

    # Initialize TF-IDF vectorizer
    # ngram_range=(1,2) means we capture both single words AND pairs of words (bigrams)
    # e.g., "mind control" as a phrase is more meaningful than just "mind" or "control"
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),        # use unigrams and bigrams
        sublinear_tf=True          # apply log scaling to TF (reduces effect of very frequent terms)
    )

    # fit_transform on training data: learns vocabulary AND converts to numbers
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # transform on test data: converts to numbers using SAME vocabulary as training
    # (we do NOT fit again on test data — that would leak test info into training)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"[INFO] Vocabulary size: {len(vectorizer.vocabulary_)} unique terms\n")

    return X_train_tfidf, X_test_tfidf, vectorizer


# ─────────────────────────────────────────────
# STEP 5: TRAIN THE MODEL
# ─────────────────────────────────────────────

def train_model(X_train_tfidf, y_train):
    """
    Train a Logistic Regression classifier.

    Why Logistic Regression?
        - Simple, fast, and interpretable — great for a viva explanation
        - Works well for text classification with TF-IDF features
        - Produces probability scores (predict_proba) for confidence display
        - The math: P(Real) = sigmoid(w1*x1 + w2*x2 + ... + wn*xn + b)
          where each xi is the TF-IDF score of a word

    Args:
        X_train_tfidf: TF-IDF feature matrix for training data.
        y_train (pd.Series): Training labels (0 or 1).

    Returns:
        LogisticRegression: Trained classifier model.
    """
    print("[INFO] Training Logistic Regression model...")

    # max_iter=1000 ensures the optimizer has enough iterations to converge
    # C=1.0 is the regularization strength (higher C = less regularization)
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',       # L-BFGS is efficient for small-to-medium datasets
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    print("[INFO] Model training complete!\n")
    return model


# ─────────────────────────────────────────────
# STEP 6: EVALUATE THE MODEL
# ─────────────────────────────────────────────

def evaluate_model(model, X_test_tfidf, y_test):
    """
    Evaluate the trained model on the held-out test set.

    Metrics explained:
        - Accuracy   : Percentage of all predictions that were correct
        - Precision  : Of all news predicted as Fake, how many were actually Fake?
        - Recall     : Of all actually Fake news, how many did we correctly identify?

    Args:
        model: Trained LogisticRegression model.
        X_test_tfidf: TF-IDF matrix of test data.
        y_test (pd.Series): True labels for test data.
    """
    print("[INFO] Evaluating model on test set...")

    # Generate predictions on the test set
    y_pred = model.predict(X_test_tfidf)

    # Calculate individual metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)

    print("=" * 50)
    print("         MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {accuracy  * 100:.2f}%")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  Recall    : {recall    * 100:.2f}%")
    print("=" * 50)
    print("\n[INFO] Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake News", "Real News"]))


# ─────────────────────────────────────────────
# STEP 7: SAVE THE MODEL AND VECTORIZER
# ─────────────────────────────────────────────

def save_artifacts(model, vectorizer, model_dir):
    """
    Save the trained model and vectorizer to disk using joblib.

    WHY save both?
        - The vectorizer must convert input text the EXACT same way as training.
        - If we only saved the model, we couldn't convert new text to numbers correctly.
        - Both must be loaded together at prediction time.

    Args:
        model: Trained LogisticRegression model.
        vectorizer: Fitted TfidfVectorizer.
        model_dir (str): Directory path to save the files.
    """
    os.makedirs(model_dir, exist_ok=True)

    model_path      = os.path.join(model_dir, 'model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')

    # joblib is more efficient than pickle for numpy arrays (TF-IDF matrices are numpy)
    joblib.dump(model,      model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"[INFO] Model saved to      : {model_path}")
    print(f"[INFO] Vectorizer saved to : {vectorizer_path}")
    print("\n[SUCCESS] Training complete! You can now run: python app.py\n")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Compute absolute paths regardless of where this script is run from
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    DATASET_PATH = os.path.join(project_root, 'data', 'news_dataset.csv')
    MODEL_DIR    = script_dir   # Save model.pkl and vectorizer.pkl in the /model folder

    # Execute the full training pipeline
    df                              = load_dataset(DATASET_PATH)
    df                              = preprocess_dataframe(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
    model                           = train_model(X_train_tfidf, y_train)

    evaluate_model(model, X_test_tfidf, y_test)
    save_artifacts(model, vectorizer, MODEL_DIR)
