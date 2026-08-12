<div align="center">

# 🔍 Fake News Detection System

### *Classify news headlines as Real or Fake using Machine Learning — instantly, in your browser.*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)

**[🚀 Live Demo](https://fake-news-detector-oqtf.onrender.com)** &nbsp;|&nbsp; **[📂 Repository](https://github.com/yusufsheikh2716/fake-news-detector)** &nbsp;|&nbsp; **[👤 Author](https://github.com/yusufsheikh2716)**

</div>

---

## 📖 Problem Statement

The rapid spread of misinformation online poses a serious threat to public discourse. Manually fact-checking every article is impossible at scale. This project demonstrates how classical Machine Learning (TF-IDF + Logistic Regression) can be applied to automatically classify news text as **Real** or **Fake** — with a quantified confidence score — making it a practical, explainable, and lightweight solution.

---

## ⚙️ How It Works

```
User Input (news text)
        │
        ▼
┌──────────────────────┐
│  Text Preprocessing  │  lowercase → strip URLs → remove punctuation
│                      │  → remove digits → remove stopwords
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  TF-IDF Vectorizer   │  Converts text to numerical feature matrix
│  (5000 features,     │  TF × IDF score per word — rare, distinctive
│   unigrams+bigrams)  │  words score highest
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Logistic Regression  │  Binary classifier: P(Real) = sigmoid(wᵀx)
│  Classifier          │  Outputs class label + probability (confidence)
└──────────┬───────────┘
           │
           ▼
   ✅ Real  /  🚫 Fake
   + Confidence % bar
           │
           ▼
   Logged to SQLite DB
   (text, result, confidence, timestamp)
```

### ML Pipeline Summary

| Step | Detail |
|---|---|
| Dataset | 425 labeled news rows (Real=1, Fake=0), balanced ~50/50 |
| Preprocessing | Lowercase, URL/punct/digit removal, custom stopword filter |
| Vectorization | `TfidfVectorizer(max_features=5000, ngram_range=(1,2))` |
| Model | `LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')` |
| Train/Test Split | 80% train, 20% test (`stratify=y`, `random_state=42`) |
| **Accuracy** | **87.06%** on held-out test set |
| Recall (Real) | **100%** — never misclassifies Real news as Fake |

---

## 🖥️ Screenshots

> *Add screenshots after deployment*

| Home — Input Form | Result — Real News | Result — Fake News | History Log |
|---|---|---|---|
| *(screenshot)* | *(screenshot)* | *(screenshot)* | *(screenshot)* |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.9+, Flask 3.0, Gunicorn |
| **ML** | scikit-learn (TfidfVectorizer + LogisticRegression) |
| **Data Handling** | pandas |
| **Model Persistence** | joblib |
| **Database** | SQLite (via Python's built-in `sqlite3`) |
| **Frontend** | Vanilla HTML5 + Custom CSS (glassmorphism dark theme) |
| **Hosting** | Render (free tier) |

---

## 📂 Project Structure

```
fake-news-detector/
├── data/
│   └── news_dataset.csv        ← 425-row labeled training dataset
├── model/
│   ├── train_model.py          ← Full ML training pipeline
│   ├── model.pkl               ← Trained Logistic Regression classifier
│   └── vectorizer.pkl          ← Fitted TF-IDF vectorizer
├── templates/
│   ├── index.html              ← Main analysis form + result display
│   └── history.html            ← Prediction history table
├── static/
│   └── style.css               ← Custom dark glassmorphism CSS (no frameworks)
├── app.py                      ← Flask application (routes, prediction, DB logging)
├── Procfile                    ← Gunicorn startup command (Render/Heroku)
├── render.yaml                 ← One-click Render deployment config
├── requirements.txt            ← Python dependencies
└── README.md
```

---

## 🚀 Local Setup & Installation

### Prerequisites
- Python 3.9 or newer
- pip / pip3

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yusufsheikh2716/fake-news-detector.git
cd fake-news-detector

# 2. (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the ML model (generates model.pkl + vectorizer.pkl)
python model/train_model.py

# 5. Run the Flask development server
python app.py

# 6. Open in browser
# → http://127.0.0.1:5000
```

### Expected Training Output
```
[INFO] Dataset loaded: 425 rows
[INFO] Training set size : 340 samples
[INFO] Test set size     :  85 samples
==================================================
         MODEL EVALUATION RESULTS
==================================================
  Accuracy  : 87.06%
  Precision : 81.67%
  Recall    : 100.00%
==================================================
[SUCCESS] Training complete! You can now run: python app.py
```

---

## 🌐 Live Demo

> 🔗 **[https://fake-news-detector-oqtf.onrender.com](https://fake-news-detector-oqtf.onrender.com)**

The live demo is hosted on **Render (free tier)**. It may take 30–60 seconds to wake up on first visit if the server has been idle (this is normal for Render's free plan).

---

## 🧠 Viva / Interview Q&A

| Question | Answer |
|---|---|
| What is TF-IDF? | Term Frequency × Inverse Document Frequency. Words that appear often in one document but rarely across all get the highest score — they're distinctive and meaningful. |
| Why Logistic Regression? | Fast, interpretable, mathematically explainable (`P = sigmoid(wᵀx)`), and natively outputs probabilities for confidence scores. |
| What does `predict_proba` return? | `[P(Fake), P(Real)]` as decimals. We use `probabilities[predicted_class] × 100` as the confidence %. |
| Why remove stopwords? | "the", "is", "and" appear in every article — they carry zero classification signal. Removing them lets TF-IDF focus on semantically meaningful words. |
| Why save both model.pkl AND vectorizer.pkl? | The vectorizer's vocabulary must be identical at training and prediction time. Loading a fresh vectorizer would produce a different feature space and wrong predictions. |
| Why 80/20 split with `stratify=y`? | Stratification ensures both sets maintain the same Real/Fake ratio, preventing skewed evaluation. |
| What is SQLite used for here? | Logging every prediction (input, result, confidence, timestamp) without a separate DB server — the file lives alongside the app. |

---

## ⚠️ Limitations & Future Work

- **Dataset size:** 425 synthetic rows. For production-grade accuracy, replace with [LIAR dataset](https://paperswithcode.com/dataset/liar) (12,836 statements) or [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet).
- **Context:** The model analyses text patterns only — it cannot access external URLs or cross-reference facts.
- **Language:** English-only; multi-language support would require language-specific preprocessing.
- **Future:** BERT/RoBERTa fine-tuning, real-time URL scraping, fact-check API integration.

---

## 👤 Author

**Mohd. Yusuf Sheikh**
B.Tech Information Technology — BBDNIIT, Lucknow (AKTU)

[![GitHub](https://img.shields.io/badge/GitHub-yusufsheikh2716-181717?style=flat-square&logo=github)](https://github.com/yusufsheikh2716)

---

<div align="center">
<sub>Built with ❤️ as a B.Tech IT Mini Project | Fake News Detection using Machine Learning</sub>
</div>
