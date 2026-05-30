# E-Commerce Review Sentiment Analysis

**Author:** Raja Rashid  
**Project:** M.Sc. Final Year Project (2024–2025)  
**Goal:** Classify customer product reviews as Positive, Neutral, or Negative using NLP and Machine Learning.

---

## Overview

This project builds a complete NLP pipeline to automatically classify e-commerce product reviews into three sentiment categories: **Positive**, **Neutral**, and **Negative**. Two machine learning models — Support Vector Machine (SVM) and Multinomial Naive Bayes — are trained, evaluated, and compared. The best model is then deployed via a Streamlit web app for live predictions.

---

## Pipeline

```
Raw Reviews → Preprocessing → TF-IDF Features → SMOTE Balancing → Train SVM & NB → Evaluate → Save & Deploy
```

1. **Load & Explore Data** — Load CSV, inspect shape, missing values, class distribution
2. **Text Preprocessing** — Remove emojis, URLs, HTML, punctuation; lowercase; tokenize; remove stopwords; lemmatize
3. **Feature Extraction** — TF-IDF vectorization (unigrams + bigrams, top 5000 features)
4. **Handle Class Imbalance** — SMOTE applied on training set only (prevents data leakage)
5. **Train Models** — SVM (`kernel=linear`, `C=1`) and Multinomial Naive Bayes (`alpha=1`)
6. **Evaluate & Compare** — Accuracy, classification report, confusion matrices, 5-fold cross-validation
7. **Save Models** — Serialized with `joblib` for deployment
8. **Streamlit App** — Interactive web UI for real-time sentiment prediction

---

## Project Structure

```
├── ecommerce_sentiment_analysis.py   # Main pipeline script
├── byakh.csv                         # Dataset (reviews, ratings, sentiment labels)
├── app.py                            # Streamlit deployment app
├── svm_model.pkl                     # Saved SVM model
├── naive_bayes_model.pkl             # Saved Naive Bayes model
├── tfidf_vectorizer.pkl              # Saved TF-IDF vectorizer
├── label_encoder.pkl                 # Saved label encoder
├── sentiment_distribution.png        # Pie chart + rating bar chart
├── wordclouds.png                    # Word clouds per sentiment class
├── confusion_matrices.png            # Side-by-side confusion matrices
├── model_comparison.png              # Test vs CV accuracy bar chart
└── requirements.txt                  # Python dependencies
```

---

## Setup & Installation

**Python 3.8+ required.**

```bash
# 1. Clone or download the project
git clone <your-repo-url>
cd ecommerce-sentiment-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK resources (done automatically in the script)
python -c "import nltk; nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet'])"
```

---

## Usage

### Train the Models

```bash
python ecommerce_sentiment_analysis.py
```

This will:
- Preprocess the dataset (`byakh.csv`)
- Train SVM and Naive Bayes models
- Output evaluation metrics and save charts
- Save all model files (`.pkl`)

### Run the Streamlit App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, paste any product review, and click **Predict** to get the sentiment.

### Predict from Python

```python
import joblib
from ecommerce_sentiment_analysis import preprocess_text

model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

review = "Absolutely loved this product, great quality!"
cleaned = preprocess_text(review)
X = vectorizer.transform([cleaned]).toarray()
prediction = label_encoder.inverse_transform(model.predict(X))[0]
print(f"Sentiment: {prediction}")
```

---

## Dataset

The dataset (`byakh.csv`) contains e-commerce product reviews with the following columns:

| Column | Description |
|---|---|
| `review` | Full review text written by the customer |
| `summary` | Short review title/summary |
| `rating` | Star rating (1–5) |
| `sentiment` | Target label: `Positive`, `Neutral`, or `Negative` |

The combined `summary + review` text is used as model input for richer feature representation.

---

## Text Preprocessing Steps

Each review goes through the following pipeline:

1. Convert to lowercase
2. Remove emojis (via `emoji` library)
3. Remove URLs, HTML tags, and special characters
4. Remove digits
5. Tokenize using NLTK `word_tokenize`
6. Remove English stopwords
7. Lemmatize tokens using `WordNetLemmatizer`

---

## Models & Configuration

| Model | Key Hyperparameters |
|---|---|
| SVM | `kernel=linear`, `C=1`, `gamma=scale` |
| Multinomial Naive Bayes | `alpha=1` (Laplace smoothing) |
| TF-IDF Vectorizer | `max_features=5000`, `ngram_range=(1,2)`, `stop_words=english` |

Both models are trained on SMOTE-balanced data and evaluated on the original (unbalanced) test set to reflect real-world performance.

---

## Outputs & Visualizations

| File | Description |
|---|---|
| `sentiment_distribution.png` | Pie chart of sentiment class proportions + star rating bar chart |
| `wordclouds.png` | Word clouds for Positive, Neutral, and Negative review classes |
| `confusion_matrices.png` | Confusion matrices for SVM and Naive Bayes side by side |
| `model_comparison.png` | Bar chart comparing test accuracy and 5-fold CV accuracy |

---

## Dependencies

See `requirements.txt` for the full list. Key libraries:

- `pandas`, `numpy` — data manipulation
- `nltk`, `emoji` — text preprocessing
- `scikit-learn` — TF-IDF, models, evaluation
- `imbalanced-learn` — SMOTE oversampling
- `joblib` — model serialization
- `matplotlib`, `seaborn`, `wordcloud` — visualization
- `streamlit` — web app deployment
- `tqdm` — progress bars

---

## Results

### SVM — Accuracy: **96.41%**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Negative | 0.98 | 0.96 | 0.97 | 280 |
| Neutral | 0.89 | 0.90 | 0.89 | 124 |
| Positive | 0.97 | 0.98 | 0.97 | 710 |
| **Weighted Avg** | **0.96** | **0.96** | **0.96** | **1114** |

### Naive Bayes — Accuracy: **92.73%**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Negative | 0.96 | 0.91 | 0.93 | 280 |
| Neutral | 0.71 | 0.81 | 0.76 | 124 |
| Positive | 0.96 | 0.95 | 0.96 | 710 |
| **Weighted Avg** | **0.93** | **0.93** | **0.93** | **1114** |

## Screenshots

### App Interface
![App Screenshot](screenshots/app.png)

### Sentiment Distribution
![Sentiment Distribution](screenshots/sentiment_distribution.png)

### Confusion Matrices
![Confusion Matrices](screenshots/confusion_matrices.png)

### Model Comparison
![Model Comparison](screenshots/model_comparison.png)

### Key Observations

- **SVM outperforms Naive Bayes** across all metrics — 3.68% higher accuracy overall.
- Both models struggle most with the **Neutral** class, which is expected given its semantic ambiguity and smaller support (124 vs 280/710).
- SVM's Neutral F1 of **0.89** is notably stronger than Naive Bayes's **0.76**, making SVM the clear choice for deployment.
- **Negative** and **Positive** classes are classified with high confidence by both models (F1 ≥ 0.93).
- Total test set: **1,114 samples** across 3 classes (Positive: 710, Negative: 280, Neutral: 124).

> **Recommended model for deployment: SVM** — higher accuracy, better Neutral class handling, and more consistent performance across all three sentiment classes.

---

## Notes

- SMOTE is applied **only on training data** to prevent data leakage into the test set.
- The `combined_text` feature merges `cleaned_summary` and `cleaned_review` for richer input.
- Cross-validation is run on the full (pre-SMOTE) dataset for an unbiased performance estimate.
- The Streamlit app uses the saved SVM model by default (swap to `naive_bayes_model.pkl` if preferred).
