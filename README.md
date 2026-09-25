# 📊 Social Media Sentiment Analysis

An end-to-end Data Science & Natural Language Processing (NLP) project that cleans raw social media text data, performs exploratory data analysis (EDA), compares supervised machine learning classifiers, and evaluates rule-based sentiment scoring using NLTK's VADER analyzer.

---

## 📌 Project Overview

Social media platforms generate vast amounts of unstructured user content daily. The goal of this project is to clean and process raw social media posts, analyze keyword frequencies, and classify comments into **Positive**, **Neutral**, or **Negative** sentiment categories using both traditional Machine Learning models and Rule-Based Lexicon approaches.

---

## 🛠️ Tech Stack & Key Libraries

- **Language:** Python 3.x
- **Data Manipulation:** `pandas`, `numpy`
- **NLP & Text Processing:** `nltk`, `VADER` (`SentimentIntensityAnalyzer`), `TextBlob`
- **Data Visualization:** `matplotlib`, `seaborn`, `wordcloud`
- **Machine Learning:** `scikit-learn` (`TfidfVectorizer`, `LogisticRegression`, `MultinomialNB`, `RandomForestClassifier`)

---

## 🚀 Workflow & Pipeline

1. **Text Preprocessing & Cleaning**
   - Standardized raw text inputs and removed noise (URLs, special characters, numerical digits, and hashtags).
   - Filtered out standard English stop-words using `nltk`.
   - Tokenized text for downstream feature extraction.

2. **Exploratory Data Analysis (EDA)**
   - Visualized original sentiment distributions using Seaborn `countplot`.
   - Built sentiment-specific **Word Clouds** (Positive vs. Negative) to highlight high-frequency keywords.

3. **Supervised Machine Learning**
   - Converted clean text into numerical vectors using `TfidfVectorizer`.
   - Trained and compared multiple models:
     - **Logistic Regression**
     - **Multinomial Naive Bayes**
     - **Random Forest Classifier**
   - Evaluated models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

4. **Lexicon-Based Sentiment Scoring (VADER)**
   - Leveraged NLTK's `SentimentIntensityAnalyzer` (VADER) to derive polarity scores optimized for short social media posts.
   - Outperformed traditional baseline models on sample text data with **85.71% accuracy**.

5. **Inference & Export**
   - Built a custom `analyze_tweet()` function for real-time sentiment scoring.
   - Exported final predictions to `social_media_sentiment_final_results.csv`.

---

## 📈 Model Performance Comparison

| Model / Approach | Accuracy | Highlights |
| :--- | :---: | :--- |
| **VADER Sentiment Analyzer** | **85.71%** | Handles social media slang, emojis, and punctuation effectively. |
| **Logistic Regression (TF-IDF)** | ~50.00% | Baseline supervised model; limited by small training set size. |
| **Multinomial Naive Bayes** | ~50.00% | Fast baseline probabilistic classifier. |
| **Random Forest** | ~0.00% | Overfits on small feature sets without hyperparameter tuning. |

---

## 📂 Project Structure

```text
├── Social Media Sentiment Analysis.ipynb    # Main Jupyter Notebook with code & visualizations
├── social_media_sentiment_processed.csv      # Intermediate clean data
├── social_media_sentiment_final_results.csv  # Final output data with VADER predictions
└── README.md                                 # Project documentation

---

💻 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/pachareshreyash41-dev/social-media-sentiment-analysis.git](https://github.com/pachareshreyash41-dev/social-media-sentiment-analysis.git)
   cd social-media-sentiment-analysis
