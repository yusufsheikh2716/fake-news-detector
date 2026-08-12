"""
app.py
======
Flask web application for the Fake News Detection System.

Routes:
    GET  /         -> Renders the main form (index.html)
    POST /predict  -> Accepts text, runs ML prediction, logs to SQLite, returns result
    GET  /history  -> Shows last 10 predictions from the database

Security Hardening Applied (8 Layers):
    1.  Rate Limiting       -- Max 10 predictions/minute per IP (blocks bots & DoS)
    2.  Security Headers    -- CSP, X-Frame-Options, HSTS (blocks XSS, clickjacking)
    3.  Input Sanitization  -- Strips HTML/script tags before processing
    4.  Content Length Cap  -- Max 16 KB request body (blocks large payload attacks)
    5.  Secret Key          -- Loaded from environment variable, never hardcoded
    6.  Parameterized SQL   -- Prevents SQL injection in all DB queries
    7.  Server Fingerprint  -- Removes Flask/Werkzeug version from HTTP headers
    8.  Error Handlers      -- 404/500/429/413 never expose stack traces to users

Run with:
    python app.py
"""

import os
import re
import string
import sqlite3
import secrets
import datetime
import joblib
import logging

from flask import Flask, render_template, request, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
# Log to console only — never log raw user input to files (privacy best practice)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s -- %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# APP CONFIGURATION
# ─────────────────────────────────────────────

app = Flask(__name__)

# ── SECRET KEY ────────────────────────────────────────────────────────────────
# Flask uses the secret key to sign cookies/sessions cryptographically.
# SECURITY RULE: Never hardcode a secret key in source code — anyone reading
# the code on GitHub could forge session cookies.
# We read it from an environment variable (set on Render dashboard).
# Locally, we fall back to a fresh 256-bit random value per process startup.
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)

# ── FILE PATHS ────────────────────────────────────────────────────────────────
# All paths are relative to this file — works on any machine/server.
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
VECT_PATH  = os.path.join(BASE_DIR, 'model', 'vectorizer.pkl')
DB_PATH    = os.path.join(BASE_DIR, 'database.db')

# ── REQUEST SIZE LIMIT ────────────────────────────────────────────────────────
# SECURITY: Reject any HTTP body larger than 16 KB automatically.
# Prevents "Large Payload Attacks" where an attacker sends gigabytes of data
# to exhaust server memory. Flask returns HTTP 413 if this is exceeded.
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024   # 16 KB

# Secondary Python-level character limit on the text field
MAX_INPUT_LENGTH = 5000


# ─────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────
# SECURITY: Rate limiting prevents:
#   - Brute-force attacks (thousands of requests per second)
#   - Denial-of-Service (DoS) attacks (flooding the server)
#   - Automated scraping bots
#
# We limit by remote IP address.
# Global defaults: 200 requests/day, 50/hour per IP.
# /predict route gets a stricter limit: 10/minute (see decorator on the route).
# storage_uri="memory://" stores counters in RAM — works for single gunicorn
# worker (Render free tier). For multi-worker, use Redis.
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    strategy="fixed-window"
)


# ─────────────────────────────────────────────
# SECURITY HEADERS
# ─────────────────────────────────────────────

@app.after_request
def set_security_headers(response):
    """
    Inject HTTP security headers into EVERY response the server sends.

    Browsers obey these headers and refuse to perform dangerous operations
    even if an attacker somehow injects content into our page.

    Header-by-header explanation:
        X-Content-Type-Options      : Browser must not guess content type
                                      (prevents MIME-sniffing attacks)
        X-Frame-Options             : Page cannot be loaded in an <iframe>
                                      (prevents clickjacking attacks)
        X-XSS-Protection            : Legacy XSS filter for old browsers
        Referrer-Policy             : Limits how much URL info leaks to other sites
        Permissions-Policy          : Disables camera/mic/geolocation for this page
        Content-Security-Policy     : Whitelist of where JS/CSS/fonts may load from
                                      (most powerful XSS mitigation header)
        Strict-Transport-Security   : Force HTTPS connections for 1 year
        form-action 'self'          : Forms can only POST to our own domain
    """
    response.headers['X-Content-Type-Options']  = 'nosniff'
    response.headers['X-Frame-Options']         = 'DENY'
    response.headers['X-XSS-Protection']        = '1; mode=block'
    response.headers['Referrer-Policy']         = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy']      = (
        'geolocation=(), microphone=(), camera=(), payment=(), usb=()'
    )
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )
    response.headers['Strict-Transport-Security'] = (
        'max-age=31536000; includeSubDomains'
    )

    # ── REMOVE SERVER FINGERPRINTING ─────────────────────────────────────────
    # Flask/Werkzeug adds "Server: Werkzeug/x.x.x Python/x.x.x" by default.
    # This reveals exactly which software + version we run, helping attackers
    # look up CVEs (known vulnerabilities). We replace it with a generic string.
    response.headers['Server'] = 'WebServer'
    response.headers.pop('X-Powered-By', None)

    return response


# ─────────────────────────────────────────────
# INPUT SANITIZATION
# ─────────────────────────────────────────────

def sanitize_input(text):
    """
    Strip potentially malicious content from raw user input.

    SECURITY (Defence-in-Depth):
        Jinja2 auto-escapes template variables, preventing reflected XSS in
        the browser. This function adds a SECOND layer by removing dangerous
        patterns before they ever reach our Python logic or database.

    What we remove:
        - HTML/XML tags        : <script>, <img onerror=...>, <a href=...>
        - JS/data URI schemes  : javascript:alert(1), data:text/html,...
        - Null bytes           : \x00 (can confuse some parsers)

    Args:
        text (str): Raw user input string.

    Returns:
        str: Sanitized input safe for ML processing and database storage.
    """
    if not isinstance(text, str):
        return ''

    # Remove all HTML/XML tags including <script>, <iframe>, etc.
    text = re.sub(r'<[^>]*>', '', text)

    # Remove dangerous URI schemes used in XSS payloads
    text = re.sub(r'(?i)(javascript|data|vbscript)\s*:', '', text)

    # Remove null bytes that can bypass filters in some parsers
    text = text.replace('\x00', '')

    # Normalize line endings and collapse excessive whitespace
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]{3,}', '  ', text)

    return text.strip()


# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────

def init_db():
    """
    Create the SQLite database and predictions table if they don't exist.

    SECURITY NOTES:
        - CHECK constraints enforce that result is only 'Real' or 'Fake'
          and confidence is between 0 and 100 — bad data is rejected at DB level
        - All INSERT/SELECT queries use parameterized statements (? placeholders)
          which completely prevent SQL Injection attacks
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT    NOT NULL,
            result     TEXT    NOT NULL CHECK(result IN ("Real", "Fake")),
            confidence REAL    NOT NULL CHECK(confidence >= 0 AND confidence <= 100),
            timestamp  TEXT    NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def log_prediction(input_text, result, confidence):
    """
    Save a prediction record to the SQLite database.

    SECURITY: Parameterized query (? placeholders) ensures user input is
    treated as DATA only, never executed as SQL code.
    Even if someone types "'; DROP TABLE predictions; --", it is stored
    as a literal string, not executed.

    Args:
        input_text (str): Sanitized text from the user.
        result     (str): "Real" or "Fake".
        confidence (float): Confidence percentage.
    """
    # Validate result before inserting — belt-and-suspenders check
    if result not in ('Real', 'Fake'):
        logger.warning("Blocked attempt to insert invalid result value.")
        return

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            # Parameterized query — NEVER use f-string or .format() for SQL
            'INSERT INTO predictions (input_text, result, confidence, timestamp) VALUES (?, ?, ?, ?)',
            (input_text[:MAX_INPUT_LENGTH], result, round(confidence, 2), timestamp)
        )
        conn.commit()
    except sqlite3.Error as db_err:
        # Log internally — never expose DB error details to the user
        logger.error("Database write error: %s", db_err)
    finally:
        conn.close()


def get_last_predictions(limit=10):
    """
    Retrieve the most recent predictions from the database.

    SECURITY: The limit parameter is cast to int and capped at 100
    to prevent an internal caller from accidentally fetching all rows.
    """
    limit = min(int(limit), 100)   # never fetch more than 100 rows

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, input_text, result, confidence, timestamp '
            'FROM predictions ORDER BY id DESC LIMIT ?',
            (limit,)
        )
        return cursor.fetchall()
    except sqlite3.Error as db_err:
        logger.error("Database read error: %s", db_err)
        return []
    finally:
        conn.close()


# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────

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
    Apply the same cleaning pipeline used during model training.
    Must be identical to train_model.py to get correct predictions.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    return ' '.join(w for w in words if w not in STOPWORDS)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

def load_model_artifacts():
    """
    Load the pre-trained ML model and TF-IDF vectorizer from disk.

    SECURITY: Files are loaded only from the fixed, known paths defined
    at startup. User input cannot influence which file gets loaded.

    Raises:
        FileNotFoundError: If model.pkl or vectorizer.pkl do not exist.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
        raise FileNotFoundError(
            "Model files not found. Please run: python model/train_model.py"
        )
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    return model, vectorizer


# ─────────────────────────────────────────────
# CUSTOM ERROR HANDLERS
# ─────────────────────────────────────────────
# SECURITY: Without custom error handlers, Flask shows a default error page
# that includes the Python version, library name, and sometimes file paths.
# Custom handlers show only a user-friendly message and log the real error
# internally — attackers learn nothing useful.

@app.errorhandler(400)
def bad_request(e):
    """Malformed HTTP request."""
    return render_template('index.html', result=None,
        error="Bad request. Please use the form correctly.", input_text=''), 400

@app.errorhandler(404)
def not_found(e):
    """Route or page does not exist."""
    return render_template('index.html', result=None,
        error="Page not found. You have been redirected home.", input_text=''), 404

@app.errorhandler(405)
def method_not_allowed(e):
    """Wrong HTTP method used on a route."""
    return render_template('index.html', result=None,
        error="Method not allowed.", input_text=''), 405

@app.errorhandler(413)
def request_too_large(e):
    """Request body exceeded 16 KB MAX_CONTENT_LENGTH limit."""
    return render_template('index.html', result=None,
        error="Request too large. Maximum allowed size is 16 KB.", input_text=''), 413

@app.errorhandler(429)
def too_many_requests(e):
    """Rate limit exceeded — IP sent too many requests."""
    return render_template('index.html', result=None,
        error="Too many requests. Please wait a moment before trying again.", input_text=''), 429

@app.errorhandler(500)
def internal_error(e):
    """
    Unexpected server error.
    SECURITY: Log the real error internally; show only a generic message to user.
    Stack traces contain file paths, variable names, and library versions —
    all useful to an attacker.
    """
    logger.exception("Internal server error: %s", e)
    return render_template('index.html', result=None,
        error="An unexpected error occurred. Please try again later.", input_text=''), 500


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    """Home page — renders the input form. No rate limit on GET (harmless)."""
    return render_template('index.html', result=None, error=None, input_text='')


@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")   # SECURITY: Max 10 ML predictions per IP per minute
def predict():
    """
    Prediction route — ML inference endpoint.

    Security layers active on this route:
        [1] Rate limit: 10 requests/minute per IP (decorator above)
        [2] Content length: max 16 KB enforced by Flask config
        [3] Input sanitized: HTML/script tags stripped
        [4] Length validated: max 5000 chars enforced in Python
        [5] ML errors caught: never expose model internals to user
        [6] DB write: parameterized query prevents SQL injection
    """
    # Step 1: Get and sanitize input
    raw_text = sanitize_input(request.form.get('news_text', ''))

    # Step 2: Validate — reject empty input
    if not raw_text or not raw_text.strip():
        return render_template('index.html', result=None,
            error="Please enter some news text before submitting.", input_text='')

    # Step 3: Validate — reject overly long input
    if len(raw_text) > MAX_INPUT_LENGTH:
        return render_template('index.html', result=None,
            error=f"Input is too long. Please limit to {MAX_INPUT_LENGTH} characters.",
            input_text=raw_text[:200] + '...')

    # Step 4: Load model
    try:
        model, vectorizer = load_model_artifacts()
    except FileNotFoundError:
        # Don't reveal the internal error message — just log it
        logger.error("Model artifacts not found when serving /predict")
        return render_template('index.html', result=None,
            error="Model not available. Please contact the administrator.",
            input_text=raw_text)

    # Step 5: Preprocess and check result
    cleaned = preprocess_text(raw_text)
    if not cleaned.strip():
        return render_template('index.html', result=None,
            error="No meaningful text found. Please enter a real news headline.",
            input_text=raw_text)

    # Step 6: Run ML prediction (wrapped in try/except for safety)
    try:
        input_vector     = vectorizer.transform([cleaned])
        prediction_label = model.predict(input_vector)[0]
        probabilities    = model.predict_proba(input_vector)[0]
        confidence_score = probabilities[prediction_label] * 100
    except Exception as ml_err:
        logger.exception("ML prediction failed: %s", ml_err)
        return render_template('index.html', result=None,
            error="Prediction failed. Please try again.", input_text=raw_text)

    result_text = "Real" if prediction_label == 1 else "Fake"

    # Step 7: Log to database (parameterized — SQL injection proof)
    log_prediction(raw_text, result_text, confidence_score)

    # Step 8: Return result
    return render_template('index.html',
        result=result_text,
        confidence=round(confidence_score, 2),
        error=None,
        input_text=raw_text)


@app.route('/history', methods=['GET'])
@limiter.limit("30 per minute")   # SECURITY: slightly looser than /predict
def history():
    """History page — shows last 10 logged predictions."""
    predictions = get_last_predictions(limit=10)
    return render_template('history.html', predictions=predictions)


# ─────────────────────────────────────────────
# BLOCK COMMON ATTACK SCANNER PATHS
# ─────────────────────────────────────────────
# Automated vulnerability scanners (Nikto, Shodan, etc.) always probe these
# paths first. Returning 404 immediately gives them zero information and
# wastes their time.

@app.route('/admin',        methods=['GET', 'POST'])
@app.route('/wp-admin',     methods=['GET', 'POST'])
@app.route('/phpmyadmin',   methods=['GET', 'POST'])
@app.route('/.env',         methods=['GET'])
@app.route('/config',       methods=['GET'])
@app.route('/shell',        methods=['GET', 'POST'])
@app.route('/backup',       methods=['GET'])
def block_scanners():
    """Return 404 for well-known attacker probe paths."""
    abort(404)


# ─────────────────────────────────────────────
# MODULE-LEVEL DB INIT (required for gunicorn)
# ─────────────────────────────────────────────
# Gunicorn imports this file as a module — __main__ is never executed.
# We call init_db() here so the table exists before the first request.
try:
    init_db()
    logger.info("Database initialized successfully.")
except Exception as _db_err:
    logger.warning("DB init at module level failed: %s", _db_err)


# ─────────────────────────────────────────────
# MAIN ENTRY POINT (local development only)
# ─────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    logger.info("Starting development server on http://127.0.0.1:%s", port)
    # SECURITY: debug=False always — debug mode exposes an interactive Python
    # console in the browser that any visitor could use to run arbitrary code.
    app.run(debug=False, host='0.0.0.0', port=port)
