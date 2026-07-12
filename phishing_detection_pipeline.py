"""
AI-Driven Phishing Email Detection Using NLP
==============================================
Full pipeline: data cleaning -> feature engineering (TF-IDF + metadata)
-> model training (Logistic Regression, Random Forest, Naive Bayes, Neural
Network) -> evaluation (accuracy, precision, recall, F1) -> visualization
(confusion matrices, feature importance).

Author: Monika G (AI/ML, Sathyabama Institute of Science and Technology)
IICT Summer Internship Project
"""

import re
import string
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from scipy.sparse import hstack, csr_matrix

import nltk
from nltk.corpus import stopwords

for pkg in ["stopwords", "punkt"]:
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg == "stopwords" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))

OUT = "/home/claude/phishing_project"

# -----------------------------------------------------------------
# 1. DATA COLLECTION
# -----------------------------------------------------------------
df = pd.read_csv(f"{OUT}/phishing_email_dataset_raw.csv")
print(f"Loaded dataset: {df.shape[0]} emails, {df.shape[1]} columns")
print(df["label"].value_counts().rename({0: "Legitimate", 1: "Phishing"}))

# -----------------------------------------------------------------
# 2. DATA CLEANING
# -----------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)                 # remove HTML tags
    text = re.sub(r"http\S+|www\.\S+", " URLTOKEN ", text)  # normalize URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

df["full_text"] = df["subject"].astype(str) + " " + df["body"].astype(str)
df["clean_text"] = df["full_text"].apply(clean_text)

df.to_csv(f"{OUT}/phishing_email_dataset_cleaned.csv", index=False)
print("\nSample cleaned text:")
print(df[["full_text", "clean_text"]].head(2).to_string())

# -----------------------------------------------------------------
# 3. FEATURE ENGINEERING (TF-IDF + metadata)
# -----------------------------------------------------------------
X_train_text, X_test_text, y_train, y_test, train_idx, test_idx = train_test_split(
    df["clean_text"], df["label"], df.index,
    test_size=0.25, random_state=42, stratify=df["label"]
)

tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

meta_cols = ["num_urls", "has_suspicious_domain", "urgency_word_count"]
scaler = MinMaxScaler()
meta_train = scaler.fit_transform(df.loc[train_idx, meta_cols])
meta_test = scaler.transform(df.loc[test_idx, meta_cols])

X_train = hstack([X_train_tfidf, csr_matrix(meta_train)])
X_test = hstack([X_test_tfidf, csr_matrix(meta_test)])

feature_names = list(tfidf.get_feature_names_out()) + meta_cols
print(f"\nFeature matrix: {X_train.shape[1]} features "
      f"({len(tfidf.get_feature_names_out())} TF-IDF + {len(meta_cols)} metadata)")
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# -----------------------------------------------------------------
# 4. MODEL DEVELOPMENT
# -----------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                           random_state=42, early_stopping=True),
}

results = {}
predictions = {}
probs = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    if hasattr(model, "predict_proba"):
        probs[name] = model.predict_proba(X_test)[:, 1]
    else:
        probs[name] = y_pred

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

results_df = pd.DataFrame(results).T.round(4)
results_df.to_csv(f"{OUT}/model_comparison_results.csv")
print("\n=== Model Comparison Summary ===")
print(results_df)

# -----------------------------------------------------------------
# 5. EVALUATION VISUALIZATIONS
# -----------------------------------------------------------------
sns.set_style("whitegrid")

# --- 5a. Confusion matrices (2x2 grid) ---
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for ax, (name, y_pred) in zip(axes.flatten(), predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"])
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrices.png", dpi=150)
plt.close()

# --- 5b. Model comparison bar chart ---
fig, ax = plt.subplots(figsize=(10, 6))
results_df.plot(kind="bar", ax=ax, colormap="viridis")
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUT}/model_comparison_chart.png", dpi=150)
plt.close()

# --- 5c. Feature importance (Random Forest) ---
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
top_idx = np.argsort(importances)[-20:]
fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(range(len(top_idx)), importances[top_idx], color="#2C5F2D")
ax.set_yticks(range(len(top_idx)))
ax.set_yticklabels([feature_names[i] for i in top_idx])
ax.set_xlabel("Feature Importance")
ax.set_title("Top 20 Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=150)
plt.close()

# --- 5d. ROC curves ---
fig, ax = plt.subplots(figsize=(8, 7))
for name, y_prob in probs.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - Model Comparison", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/roc_curves.png", dpi=150)
plt.close()

print("\nAll visualizations saved to:", OUT)
print("Pipeline complete.")
