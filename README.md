# Phishing Email Detection using NLP

AI-driven system that classifies emails as **phishing** or **legitimate** using Natural Language Processing and machine learning. Built as part of the IICT AI/ML Summer Internship Program.

## Overview

Phishing emails rely on deceptive language, urgency cues, and disguised links rather than technical exploits. This project builds an end-to-end ML pipeline that combines **text analysis (TF-IDF)** with **structural metadata** (URL count, sender domain suspicion, urgency-word frequency) to detect phishing emails automatically.

Since a large real-world labeled dataset wasn't available, a **synthetic dataset of 1,300 emails** was generated with realistic phishing/legitimate patterns — including deliberate class overlap (e.g. legitimate emails using urgency-like phrasing) to avoid an unrealistically easy classification problem.

## Dataset

- **1,300 emails** — 650 phishing, 650 legitimate
- Features: subject, body, sender, `num_urls`, `has_suspicious_domain`, `urgency_word_count`
- `phishing_email_dataset_raw.csv` — original generated data
- `phishing_email_dataset_cleaned.csv` — after text preprocessing

## Pipeline

1. **Data Cleaning** — lowercase, strip HTML, normalize URLs, remove punctuation/digits, remove stopwords (NLTK)
2. **Feature Engineering** — TF-IDF (unigrams + bigrams, 1,500 features) + scaled metadata features → 842-dim feature vector
3. **Model Training** — 75/25 stratified train-test split, 4 classifiers trained:
   - Logistic Regression
   - Random Forest (200 trees)
   - Multinomial Naive Bayes
   - Neural Network (MLP, 2 hidden layers)
4. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrices, Feature Importance

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Neural Network (MLP) | 0.9938 | 0.9878 | 1.0000 | 0.9939 |

Metadata features (`has_suspicious_domain`, `urgency_word_count`) ranked among the top predictors alongside key lexical terms like "verify" and "click" — confirming that combining structural and text signals is more effective than text alone.

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `NLTK` · `matplotlib` · `seaborn`

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk scipy

python3 generate_dataset.py          # generates the dataset
python3 phishing_detection_pipeline.py   # runs the full pipeline
```

Outputs: cleaned dataset, model comparison CSV, confusion matrices, ROC curves, and feature importance plots.

## Future Work

- Validate on real-world datasets (Kaggle/UCI phishing corpora)
- Explore word embeddings (Word2Vec/BERT) for richer text representation
- Deploy via a Flask/Streamlit interface for live email classification

## Author
**Monika G** — B.E. CSE With AI & ML, Sathyabama Institute of Science and Technology
IICT Summer Internship Program
