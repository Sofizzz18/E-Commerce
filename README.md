# E-Commerce Review Sentiment Analysis

A machine learning project that classifies customer product reviews as **Positive**, **Neutral**, or **Negative** using NLP techniques and classical ML algorithms. Built as part of M.Sc. Final Year Project (2024–2025).

---

## Results

| Model | Accuracy |
|-------|----------|
| Support Vector Machine (SVM) | **93.81%** |
| Naive Bayes | **94.43%** |

---

## Project Overview

Online product reviews contain valuable insights about customer satisfaction. This project builds an end-to-end sentiment classification pipeline that:

- Cleans and preprocesses raw review text (removing URLs, emojis, special characters)
- Converts text to numerical features using **TF-IDF vectorization** (unigrams + bigrams)
- Handles class imbalance using **SMOTE** oversampling
- Trains and evaluates **SVM** and **Naive Bayes** classifiers
- Visualizes sentiment distribution and word clouds per sentiment category

---

## Dataset

- **Size:** 5,568 customer reviews
- **Source:** E-commerce product reviews (CSV format)
- **Features:** `review`, `summary`, `rating`
- **Labels:** Positive (rating ≥ 4), Neutral (rating = 3), Negative (rating ≤ 2)

---

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, imbalanced-learn, Matplotlib, Seaborn, WordCloud
- **ML Models:** Support Vector Machine (SVM), Multinomial Naive Bayes
- **NLP:** TF-IDF Vectorization, Tokenization, Lemmatization, Stopword Removal

---

## Project Structure

```
E-Commerce-Sentiment-Analysis/
│
├── E-Commerce_Project.ipynb   # Main notebook with full pipeline
├── byakh.csv                  # Dataset
└── README.md                  # Project documentation
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Sofizzz18/E-Commerce.git
   cd E-Commerce
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn nltk imbalanced-learn matplotlib seaborn wordcloud emoji tqdm
   ```

3. Download NLTK resources (runs automatically in the notebook):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

4. Open and run the notebook:
   ```bash
   jupyter notebook E-Commerce_Project.ipynb
   ```

---

## Key Steps in the Pipeline

1. **Data Loading & Exploration** — Load CSV, check for nulls, inspect distributions
2. **Sentiment Labeling** — Convert star ratings to Positive/Neutral/Negative labels
3. **Text Preprocessing** — Remove URLs, emojis, punctuation; tokenize; lemmatize; remove stopwords
4. **Feature Extraction** — TF-IDF with unigrams and bigrams (max 5,000 features)
5. **Class Imbalance Handling** — SMOTE oversampling on training data
6. **Model Training** — Train SVM and Naive Bayes classifiers
7. **Evaluation** — Accuracy score and classification report
8. **Visualization** — Sentiment distribution charts and word clouds per sentiment

---

## Author

**Raja Rashid**  
M.Sc. Information Technology — Cluster University, Srinagar  
[LinkedIn](https://linkedin.com/in/rajarashid18) | [GitHub](https://github.com/Sofizzz18)
