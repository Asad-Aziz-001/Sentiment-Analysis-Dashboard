# **🎯 Project: Sentiment Analysis Dashboard**

# **Link 🔗** : https://sentiment-analysis-dashboard-2rzkdja4wc6barszydoqqy.streamlit.app/ 

# **1. Overview**

This project is an interactive NLP-powered dashboard built with Streamlit that performs sentiment analysis on a dataset of text reviews (e.g., movie reviews, tweets, or comments).
The app classifies each text as Positive, Neutral, or Negative, visualizes the results, and allows real-time sentiment detection via user input.

The main goal is to combine Natural Language Processing (NLP) with data visualization to make sentiment analysis accessible for both technical and non-technical users.

# **2. Objectives**

Load and preprocess a dataset of text reviews.

Apply sentiment analysis using tools like VADER or TextBlob.

Classify sentiment into three categories: Positive, Neutral, Negative.

Visualize the sentiment distribution through interactive plots.

Provide real-time sentiment prediction for custom text input.

Deploy the dashboard online using Streamlit Cloud.

# **3. Tech Stack**

Languages & Libraries
Python → Core programming language.

pandas / numpy → Data manipulation and preprocessing.

nltk (VADER) → Rule-based sentiment analysis.

TextBlob → Polarity and subjectivity scoring.

matplotlib / plotly → Data visualization (pie chart, bar chart).

wordcloud → Generate word clouds for different sentiment classes.

transformers (Hugging Face) → (Optional) Advanced transformer-based sentiment models.

Streamlit → Web framework for interactive dashboards.

# **4. Dataset**

**Structure:**

Source: IMDB Dataset of 50K Movie Reviews

| review                      | sentiment |
| --------------------------- | --------- |
| "This movie was amazing..." | positive  |
| "Worst film ever..."        | negative  |

# **5. Features**

Data Upload & Cleaning
Upload CSV file or load default IMDB dataset.

Remove missing values, normalize text (optional).

Sentiment Analysis
VADER (nltk) → Quick, lightweight, and good for short texts.

Optionally integrate Hugging Face models for deeper accuracy.

Visualization
Pie chart → Percentage of positive, neutral, and negative reviews.

Bar chart → Sentiment counts.

Word clouds → Frequent words in each sentiment class.

User Interaction
Custom input box → Type a sentence, get instant sentiment feedback.

Optional filter to view only Positive/Negative/Neutral comments.

# **6. Deployment**

Local Run

streamlit run streamlit_sentiment_dashboard.py

Streamlit Cloud Deployment
Run requirements.txt with all dependencies

# **7. Example Output**

Pie Chart:
Positive → 60%
Neutral  → 25%
Negative → 15%

User Input Test:

"I absolutely loved this!" → Sentiment: Positive
"This was okay, nothing special." → Sentiment: Neutral
"Terrible experience, I regret watching it." → Sentiment: Negative

# **8. Learning Outcomes**

Through this project, you learn:

Data preprocessing in NLP.

Applying sentiment analysis using multiple approaches.

Visualizing results interactively with Plotly & Matplotlib.

Deploying Python apps with Streamlit Cloud.

Managing dependencies with requirements.txt.

# **9. Credits**

Developed by: ASAD AZIZ
Internship Task: Tech Horizon Internship — NLP & Data Visualization
Special Thanks: Kaggle dataset providers & Streamlit community.
