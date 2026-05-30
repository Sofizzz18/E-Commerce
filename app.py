import streamlit as st
import joblib
import re
import string
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources silently
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load models
@st.cache_resource
def load_models():
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_models()

# Preprocessing (must match training pipeline)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = emoji.replace_emoji(str(text), replace='')
    text = re.sub(r'https?://\S+|www\.\S+|ftp://\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower().strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# UI
st.set_page_config(page_title="Sentiment Analyser", page_icon="🛍️")
st.title("🛍️ E-Commerce Review Sentiment Analysis")
st.markdown("Enter a product review below to classify it as **Positive**, **Neutral**, or **Negative**.")

review = st.text_area("Product Review", placeholder="e.g. Great product, arrived quickly and works perfectly!")

if st.button("Predict Sentiment"):
    if review.strip():
        cleaned = preprocess_text(review)
        X = vectorizer.transform([cleaned]).toarray()
        pred = label_encoder.inverse_transform(model.predict(X))[0]

        color = {"Positive": "green", "Neutral": "orange", "Negative": "red"}.get(pred, "gray")
        icon  = {"Positive": "✅", "Neutral": "😐", "Negative": "❌"}.get(pred, "")

        st.markdown(f"### {icon} Sentiment: :{color}[**{pred}**]")
    else:
        st.warning("Please enter a review before predicting.")
