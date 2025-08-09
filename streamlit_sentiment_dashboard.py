"""
Streamlit Sentiment Analysis Dashboard
File: streamlit_sentiment_dashboard.py

Features:
- Load dataset (CSV) or upload custom file
- Cleans text, applies VADER sentiment analysis (NLTK)
- Optional Hugging Face transformer-based sentiment (if installed)
- Labels sentiment into Positive / Neutral / Negative
- Visualizations: pie/bar charts, sample texts, word clouds
- Interactive single-sentence sentiment checker

Usage:
pip install -r requirements.txt
streamlit run streamlit_sentiment_dashboard.py

Dataset expectation: CSV with a column named 'review' (IMDB dataset uses 'review').
"""

from typing import Optional
import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

# NLP tools
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Try importing transformers pipeline for optional advanced model
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# Ensure required NLTK data
nltk.download('vader_lexicon')

# -----------------------
# Utility / Processing
# -----------------------

VADER = SentimentIntensityAnalyzer()

def load_dataset_from_path(path: str) -> pd.DataFrame:
    """Load dataset from a path. Tries common CSV options."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    # try reading
    df = pd.read_csv(path)
    return df


def preprocess_text(text: str) -> str:
    """Basic cleaning (expandable). Keep simple to avoid changing meaning.
    - strip whitespace
    - convert to str and lower for some operations
    """
    if pd.isna(text):
        return ""
    s = str(text).strip()
    return s


def vader_sentiment_label(text: str) -> tuple:
    """Return vader scores and mapped label (Positive/Neutral/Negative).
    Uses standard VADER thresholds: compound >= 0.05 -> pos; <= -0.05 -> neg; else neutral.
    """
    scores = VADER.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        label = 'Positive'
    elif compound <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
    return scores, label


# Optional: Hugging Face pipeline
HF_PIPE = None
if HF_AVAILABLE:
    try:
        HF_PIPE = pipeline("sentiment-analysis")
    except Exception:
        HF_PIPE = None


def hf_sentiment_label(text: str) -> Optional[tuple]:
    """If HF pipeline is available, get label and score.
    Returns (label, score) where label typically 'POSITIVE' or 'NEGATIVE'.
    """
    if HF_PIPE is None:
        return None
    try:
        out = HF_PIPE(text[:512])  # limit length to first 512 chars
        if isinstance(out, list) and len(out) > 0:
            item = out[0]
            # map to readable labels
            label = item.get('label', '')
            score = item.get('score', None)
            # normalize to our three-class system
            if label.upper().startswith('POS'):
                mapped = 'Positive'
            elif label.upper().startswith('NEG'):
                mapped = 'Negative'
            else:
                mapped = 'Neutral'
            return mapped, score
    except Exception:
        return None


def textblob_polarity_label(text: str) -> tuple:
    """Fallback using TextBlob polarity (-1..1)"""
    tb = TextBlob(text)
    p = tb.sentiment.polarity
    if p > 0.05:
        lbl = 'Positive'
    elif p < -0.05:
        lbl = 'Negative'
    else:
        lbl = 'Neutral'
    return p, lbl


def generate_wordcloud(texts, max_words=150):
    """Create a WordCloud image from a list/series of texts."""
    joined = " ".join([str(t) for t in texts if t])
    if not joined:
        return None
    wc = WordCloud(width=800, height=400, background_color='white', max_words=max_words)
    img = wc.generate(joined)
    return img

# -----------------------
# Streamlit App
# -----------------------

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout='wide')

# Header
st.title("🎯 Sentiment Analysis Dashboard")
st.markdown(
    "Upload a dataset of text reviews (or use the example IMDB dataset), classify sentences as Positive/Neutral/Negative, and explore visualizations."
)

# Sidebar controls
st.sidebar.header("Data & Model Options")
use_example = st.sidebar.checkbox("Load example IMDB dataset (50k)", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload a CSV file (must contain a 'review' or 'text' column)")

model_choice = st.sidebar.selectbox("Sentiment method", options=['VADER (fast, explained)', 'TextBlob (fallback)', 'HuggingFace (optional)'])
if model_choice.startswith('HuggingFace') and not HF_AVAILABLE:
    st.sidebar.warning("Transformers not installed or pipeline unavailable. Install 'transformers' to enable.")

apply_wordcloud = st.sidebar.checkbox("Generate word clouds for each sentiment", value=True)
show_samples = st.sidebar.slider("Number of sample texts per class", min_value=1, max_value=10, value=3)

# Load data (example or uploaded)
@st.cache_data
def load_example_imdb() -> pd.DataFrame:
    """Load the IMDB example dataset. Expect file 'IMDB Dataset.csv' in working dir or fallback to remote path if provided externally."""
    # The example dataset typically has columns: 'review' and 'sentiment'
    local = 'IMDB Dataset.csv'
    if os.path.exists(local):
        df = pd.read_csv(local)
    else:
        # If not present, create an instruction DataFrame prompting the user to download
        raise FileNotFoundError("Example dataset not found locally. Please place 'IMDB Dataset.csv' in the app folder or upload your own CSV.")
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataset to contain a 'text' column for processing."""
    df = df.copy()
    # common column names
    if 'review' in df.columns:
        df.rename(columns={'review':'text'}, inplace=True)
    elif 'text' in df.columns:
        pass
    elif 'tweet' in df.columns:
        df.rename(columns={'tweet':'text'}, inplace=True)
    else:
        # fallback: take first text-like column
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            raise ValueError('No text-like column found in dataset.')
        df.rename(columns={text_cols[0]:'text'}, inplace=True)
    df['text'] = df['text'].astype(str).map(preprocess_text)
    return df


# Main pipeline: apply chosen model to dataframe
@st.cache_data
def annotate_sentiments(df: pd.DataFrame, method: str) -> pd.DataFrame:
    df = df.copy()
    # Prepare storage
    df['vader_compound'] = np.nan
    df['vader_neg'] = np.nan
    df['vader_neu'] = np.nan
    df['vader_pos'] = np.nan
    df['vader_label'] = None
    df['tb_polarity'] = np.nan
    df['tb_label'] = None
    df['hf_label'] = None
    df['hf_score'] = np.nan

    texts = df['text'].tolist()
    for i, t in enumerate(texts):
        # VADER
        scores, vlabel = vader_sentiment_label(t)
        df.at[i, 'vader_compound'] = scores['compound']
        df.at[i, 'vader_neg'] = scores['neg']
        df.at[i, 'vader_neu'] = scores['neu']
        df.at[i, 'vader_pos'] = scores['pos']
        df.at[i, 'vader_label'] = vlabel
        # TextBlob
        p, lbl_tb = textblob_polarity_label(t)
        df.at[i, 'tb_polarity'] = p
        df.at[i, 'tb_label'] = lbl_tb
        # HF (optional)
        if HF_PIPE is not None:
            hf_out = hf_sentiment_label(t)
            if hf_out:
                df.at[i, 'hf_label'] = hf_out[0]
                df.at[i, 'hf_score'] = hf_out[1]

    # Choose final label according to method preference
    if method.startswith('VADER'):
        df['sentiment'] = df['vader_label']
    elif method.startswith('TextBlob'):
        df['sentiment'] = df['tb_label']
    elif method.startswith('HuggingFace') and HF_PIPE is not None:
        # use HF where available, fallback to VADER
        df['sentiment'] = df['hf_label'].fillna(df['vader_label'])
    else:
        df['sentiment'] = df['vader_label']

    return df


# UI: Load or upload
data_loaded = None
if use_example:
    try:
        df_raw = load_example_imdb()
        data_loaded = prepare_dataframe(df_raw)
    except FileNotFoundError as e:
        st.warning(str(e))
        use_example = False

if uploaded_file is not None:
    try:
        uploaded_bytes = uploaded_file.read()
        df_up = pd.read_csv(io.BytesIO(uploaded_bytes))
        data_loaded = prepare_dataframe(df_up)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")

if data_loaded is None:
    st.info("No dataset loaded. Please upload a CSV or place 'IMDB Dataset.csv' in the app folder and enable the example dataset.")
    # still allow single-sentence input below
else:
    st.success(f"Dataset loaded: {len(data_loaded)} rows")
    # annotate
    with st.spinner('Annotating sentiments...'):
        df_annot = annotate_sentiments(data_loaded, model_choice)

    # Layout: two columns
    left, right = st.columns([2,1])

    # Left: Overview charts
    with left:
        st.subheader("Sentiment Distribution")
        counts = df_annot['sentiment'].value_counts().reindex(['Positive','Neutral','Negative']).fillna(0)
        fig_pie = px.pie(values=counts.values, names=counts.index, title='Sentiment Distribution')
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Sentiment Counts (Bar)")
        fig_bar = px.bar(x=counts.index, y=counts.values, labels={'x':'Sentiment','y':'Count'}, title='Counts by Sentiment')
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Samples")
        for s in ['Positive','Neutral','Negative']:
            st.markdown(f"**{s} samples**")
            sample_texts = df_annot[df_annot['sentiment']==s]['text'].head(show_samples).tolist()
            if not sample_texts:
                st.write("No samples for this class.")
            else:
                for i, t in enumerate(sample_texts, start=1):
                    st.write(f"{i}. {t}")

        # Optional wordclouds
        if apply_wordcloud:
            st.subheader('Word Clouds')
            wc_cols = st.columns(3)
            for idx, cls in enumerate(['Positive','Neutral','Negative']):
                with wc_cols[idx]:
                    st.markdown(f"**{cls}**")
                    texts = df_annot[df_annot['sentiment']==cls]['text']
                    img = generate_wordcloud(texts)
                    if img is None:
                        st.write('No words to display')
                    else:
                        fig, ax = plt.subplots(figsize=(6,3))
                        ax.imshow(img, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)

    # Right: Details & filters
    with right:
        st.subheader('Filters & Export')
        sentiment_filter = st.selectbox('Show only...', options=['All','Positive','Neutral','Negative'])
        if sentiment_filter!='All':
            filtered = df_annot[df_annot['sentiment']==sentiment_filter]
        else:
            filtered = df_annot
        st.write(f"Rows shown: {len(filtered)}")
        st.dataframe(filtered[['text','sentiment']].head(50))

        # Allow user to download annotated CSV
        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button('Download filtered CSV', data=csv, file_name='annotated_sentiments.csv', mime='text/csv')

# Real-time single sentence prediction
st.sidebar.header('Quick single-sentence check')
user_text = st.sidebar.text_area('Enter text to analyze', value='I love this movie!')
if st.sidebar.button('Analyze sentence'):
    st.sidebar.subheader('Result')
    if not user_text.strip():
        st.sidebar.write('Please enter a sentence.')
    else:
        scores, vlabel = vader_sentiment_label(user_text)
        st.sidebar.write(f"VADER → {vlabel} (compound={scores['compound']:.3f})")
        tb_p, tb_lbl = textblob_polarity_label(user_text)
        st.sidebar.write(f"TextBlob → {tb_lbl} (polarity={tb_p:.3f})")
        if HF_PIPE is not None:
            hf_out = hf_sentiment_label(user_text)
            if hf_out:
                st.sidebar.write(f"HuggingFace → {hf_out[0]} (score={hf_out[1]:.3f})")

# Footer / Notes
st.markdown('---')
st.caption('Notes: VADER is tuned for social media / short texts. For best performance on long reviews, consider transformer-based models (Hugging Face).')


# End of file