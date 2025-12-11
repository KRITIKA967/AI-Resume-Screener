# train_model.py
"""
Ready-to-run training script for your dataset:
- File expected: UpdatedResumeDataSet.csv (located in the same folder)
- Columns expected: 'Resume' (text) and 'Category' (label; 1/0 or words like Hired/Selected)
"""

import pandas as pd
import re
import spacy
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

CSV_FILE = "UpdatedResumeDataSet.csv"
TEXT_COLUMN = "Resume"
LABEL_COLUMN = "Category"

print("Loading CSV:", CSV_FILE)
df = pd.read_csv(CSV_FILE)

print("Columns found in CSV:", list(df.columns))

if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
    raise SystemExit(f"ERROR: Expected columns '{TEXT_COLUMN}' and '{LABEL_COLUMN}' not found. Update column names in this script.")

# Ensure types and no NaNs
X_raw = df[TEXT_COLUMN].astype(str).fillna("")
y_raw = df[LABEL_COLUMN].astype(str).fillna("")

print("\nSample resume (first row, truncated):")
print(X_raw.iloc[0][:500])
print("\nSample raw label (first row):", y_raw.iloc[0])

# Convert common label formats to 0/1
def label_to_int(label):
    label = str(label).strip()
    # numeric string
    if re.fullmatch(r"\d+", label):
        return int(label)
    low = label.lower()
    if low in {"1","true","yes","y","hired","selected","shortlisted","accept","accepted"}:
        return 1
    if low in {"0","false","no","n","not hired","not_selected","rejected","reject","not shortlisted"}:
        return 0
    # heuristic
    if "hire" in low or "select" in low or "shortlist" in low:
        return 1
    return 0

y = y_raw.apply(label_to_int)

print("\nLabel distribution after mapping to 0/1:")
print(y.value_counts())

# Load spaCy (disable heavy parts for speed)
print("\nLoading spaCy model (en_core_web_sm)...")
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

def clean_text(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        tokens.append(token.lemma_.lower())
    return " ".join(tokens)

print("Cleaning text (this may take some time depending on dataset size)...")
X = X_raw.apply(clean_text)

# Train/test split
stratify_arg = y if len(y.unique()) > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_arg
)

# Build pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

print("Training model...")
model.fit(X_train, y_train)

print("Evaluating on test set...")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Save model
MODEL_FILE = "resume_model.joblib"
joblib.dump(model, MODEL_FILE)
print(f"\nModel saved to {MODEL_FILE}.")
