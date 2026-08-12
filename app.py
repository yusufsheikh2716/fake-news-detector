"""
app.py
======
Flask web application for the Fake News Detection System.

Routes:
    GET  /         → Renders the main form (index.html)
    POST /predict  → Accepts text, runs ML prediction, logs to SQLite, returns result
    GET  /history  → Shows last 10 predictions from the database

How it works:
    1. User types a news headline/article into the form
    2. Flask receives the text via POST request
    3. We preprocess the text (same steps as training)
    4. We load the saved TF-IDF vectorizer to convert text to numbers
    5. We load the saved Logistic Regression model to predict Real/Fake
    6. The confidence score (%) comes from predict_proba (probability of each class)
    7. The result + timestamp is saved to SQLite database
    8. The result is displayed back on the same page

Run with:
    python app.py
"""

import os
import re
import string
import sqlite3
import datetime
import joblib

from flask import Flask, render_template, request

# ─────────────────────────────────────────────
# APP CONFIGURATION
# ─────────────────────────────────────────────

app = Flask(__name__)

# Build paths relative to this file so the app works from any directory
# Using os.path.abspath(__file__) ensures paths are correct regardless of
# where the process is started from (locally, via gunicorn on Render, etc.)
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'model', 'model.pkl')
VECT_PATH   = os.path.join(BASE_DIR, 'model', 'vectorizer.pkl')
DB_PATH     = os.path.join(BASE_DIR, 'database.db')

# Maximum number of characters allowed in a single prediction request
MAX_INPUT_LENGTH = 5000



# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────

def init_db():
    """
    Create the SQLite database and predictions table if they don't already exist.

    This is called once when the app starts. SQLite automatically creates
    the database.db file if it doesn't exist.

    Table schema:
        id         - Auto-increment primary key
        input_text - The raw news text entered by the user
        result     - "Real" or "Fake"
        confidence - Confidence percentage as a float (e.g., 87.34)
        timestamp  - Date and time of the prediction (ISO format string)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT    NOT NULL,
            result     TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            timestamp  TEXT    NOT NULL
        )
    ''')

    conn.commit()
    conn.close()


def log_prediction(input_text, result, confidence):
    """
    Save a single prediction record to the SQLite database.

    Args:
        input_text (str): The original text entered by the user.
        result     (str): "Real" or "Fake".
        confidence (float): Confidence percentage (0.0 to 100.0).
    """
    # Get current timestamp in human-readable ISO format
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        'INSERT INTO predictions (input_text, result, confidence, timestamp) VALUES (?, ?, ?, ?)',
        (input_text, result, round(confidence, 2), timestamp)
    )

    conn.commit()
    conn.close()


def get_last_predictions(limit=10):
    """
    Retrieve the most recent predictions from the database for the history page.

    Args:
        limit (int): Maximum number of records to fetch (default: 10).

    Returns:
        list of tuples: Each tuple is (id, input_text, result, confidence, timestamp).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ORDER BY id DESC gives us newest first
    cursor.execute(
        'SELECT id, input_text, result, confidence, timestamp FROM predictions ORDER BY id DESC LIMIT ?',
        (limit,)
    )

    rows = cursor.fetchall()
    conn.close()

    return rows


# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────

# Must use the EXACT same stopwords list as in train_model.py
# This is crucial — if training removed stopwords and prediction doesn't,
# the model will see different input patterns and give wrong results.
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
    Apply the same text cleaning pipeline used during model training.

    IMPORTANT: This function must be identical to the one in train_model.py.
    Any difference here would mean the model receives data in a format it wasn't
    trained on, causing incorrect predictions.

    Args:
        text (str): Raw user-input text.

    Returns:
        str: Cleaned, normalized text ready for TF-IDF vectorization.
    """
    # Step 1: Lowercase all characters
    text = text.lower()

    # Step 2: Remove URLs (http:// or https:// links)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 3: Remove all punctuation characters (.,!?:; etc.)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 4: Remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)

    # Step 5: Collapse multiple spaces into one and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 6: Filter out stopwords (common words with no classification value)
    words = text.split()
    filtered = [w for w in words if w not in STOPWORDS]

    return ' '.join(filtered)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

def load_model_artifacts():
    """
    Load the pre-trained model and TF-IDF vectorizer from disk.

    We load fresh from disk on each prediction to keep things simple and safe.
    For a production app you'd cache these in memory, but for a mini-project
    this approach is cleaner and easier to understand.

    Returns:
        tuple: (model, vectorizer) or raises FileNotFoundError if not found.

    Raises:
        FileNotFoundError: If model.pkl or vectorizer.pkl don't exist yet.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
        raise FileNotFoundError(
            "Model files not found. Please run: python model/train_model.py"
        )

    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)

    return model, vectorizer


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    """
    Home page route — renders the main form.

    On a GET request (user visits the page), render index.html with no results.
    The template checks for 'result' and 'error' variables to decide what to show.
    """
    return render_template('index.html', result=None, error=None, input_text='')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction route — processes the form submission and returns a result.

    Workflow:
        1. Extract the text from the form
        2. Validate it (not empty, not too long)
        3. Preprocess it the same way as training data
        4. Vectorize it using the saved TF-IDF vectorizer
        5. Predict using the saved Logistic Regression model
        6. Extract confidence score from predict_proba output
        7. Log the prediction to SQLite database
        8. Render the page again with the result
    """
    # ── 1. Get text from the submitted form ──────────────────────────────────
    raw_text = request.form.get('news_text', '')

    # ── 2. Input validation ───────────────────────────────────────────────────
    # Reject if empty or just whitespace
    if not raw_text or not raw_text.strip():
        return render_template(
            'index.html',
            result=None,
            error="⚠️ Please enter some news text before submitting.",
            input_text=''
        )

    # Reject if input is unreasonably long
    if len(raw_text) > MAX_INPUT_LENGTH:
        return render_template(
            'index.html',
            result=None,
            error=f"⚠️ Input is too long. Please limit to {MAX_INPUT_LENGTH} characters.",
            input_text=raw_text[:200] + '...'
        )

    # ── 3. Load model and vectorizer ─────────────────────────────────────────
    try:
        model, vectorizer = load_model_artifacts()
    except FileNotFoundError as e:
        return render_template(
            'index.html',
            result=None,
            error=f"⚠️ {str(e)}",
            input_text=raw_text
        )

    # ── 4. Preprocess the input ───────────────────────────────────────────────
    cleaned = preprocess_text(raw_text)

    # Edge case: after cleaning, text might be empty (e.g., user typed only numbers/symbols)
    if not cleaned.strip():
        return render_template(
            'index.html',
            result=None,
            error="⚠️ After cleaning, no meaningful text was found. Please enter a real news headline.",
            input_text=raw_text
        )

    # ── 5. Vectorize using the SAME TF-IDF vectorizer fitted during training ──
    # transform() converts our text to a numerical matrix matching the training vocabulary
    input_vector = vectorizer.transform([cleaned])

    # ── 6. Make prediction and get confidence score ───────────────────────────
    # predict()      → returns [0] for Fake or [1] for Real
    # predict_proba() → returns [[prob_fake, prob_real]] as decimals
    prediction_label = model.predict(input_vector)[0]
    probabilities    = model.predict_proba(input_vector)[0]

    # The predicted class index matches the probability array index
    # e.g., if prediction_label=1 (Real), probabilities[1] is the confidence for "Real"
    confidence_score = probabilities[prediction_label] * 100  # convert to percentage

    # Map numeric label to human-readable string
    result_text = "Real" if prediction_label == 1 else "Fake"

    # ── 7. Log to database ────────────────────────────────────────────────────
    log_prediction(raw_text, result_text, confidence_score)

    # ── 8. Render result page ─────────────────────────────────────────────────
    return render_template(
        'index.html',
        result=result_text,
        confidence=round(confidence_score, 2),
        error=None,
        input_text=raw_text
    )


@app.route('/history', methods=['GET'])
def history():
    """
    History page route — displays the last 10 predictions.

    This is useful for a demo: the faculty can see that predictions are being
    logged with timestamps, showing the SQLite integration works correctly.
    """
    # Fetch last 10 predictions from the database, newest first
    predictions = get_last_predictions(limit=10)

    return render_template('history.html', predictions=predictions)


# ─────────────────────────────────────────────
# MODULE-LEVEL DB INIT (for gunicorn / Render)
# ─────────────────────────────────────────────
# Gunicorn imports app.py as a Python module and never runs __main__.
# We must initialise the database here so it exists before any request hits.
# The 'try' guard means this is safe to call multiple times (CREATE IF NOT EXISTS).
try:
    init_db()
except Exception as _db_err:
    print(f"[WARN] DB init at module level failed (will retry): {_db_err}")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT (local dev server only)
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Initialize the database (creates tables if they don't exist)
    init_db()
    print("[INFO] Database initialized.")
    print("[INFO] Starting Flask development server...")

    # Read PORT from environment variable — Render assigns a dynamic port.
    # Falls back to 5000 for local development.
    port = int(os.environ.get('PORT', 5000))

    # host='0.0.0.0' means 'listen on all network interfaces'.
    # Required for Render and any container/cloud environment.
    # Locally this still works fine — access via http://127.0.0.1:<port>
    print(f"[INFO] Open your browser and go to: http://127.0.0.1:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)
