# prepare_and_train.py
"""
Reads UpdatedResumeDataSet.csv (columns: 'Resume', 'Category'),
inspects Category values, and if there is no meaningful hired/not-hired label,
automatically creates 'hired_auto' using Data-Analyst skill matches,
then trains a TF-IDF + LogisticRegression model and saves it.
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
CATEGORY_COLUMN = "Category"
AUTO_LABEL = "hired_auto"

print("Loading CSV:", CSV_FILE)
df = pd.read_csv(CSV_FILE)
print("Columns:", list(df.columns))

# Show sample of Category values
print("\nTop values in 'Category' (up to 20):")
unique_vals = df[CATEGORY_COLUMN].astype(str).str.strip().value_counts()
print(unique_vals.head(20))

# If Category already appears to be a hired/label column (like contains 'Hired' or 'Selected'),
# you could map it directly. But here we will create an automatic label if there is only one class.
if set(df[CATEGORY_COLUMN].astype(str).str.strip().str.lower()).issubset({"0","1","0.0","1.0","hired","selected","not hired","rejected"}):
    print("\nCategory looks like a binary label. Mapping it to 0/1.")
    def label_map(x):
        s = str(x).strip().lower()
        if s in {"1","true","yes","y","hired","selected","shortlisted","accepted"}:
            return 1
        return 0
    df[AUTO_LABEL] = df[CATEGORY_COLUMN].apply(label_map)
else:
    print("\nCategory appears to contain job-role or non-binary values.")
    print("Creating automatic label 'hired_auto' using skill-based heuristic for Data Analyst role.")

    # Data Analyst skill keywords (adjustable)
    skills = [
        "python","pandas","numpy","matplotlib","seaborn","sql","excel",
        "tableau","power bi","powerbi","etl","data cleaning","data visualization",
        "statistics","dashboard","data analysis","data wrangling","scikit-learn",
        "machine learning","regression","analytics","plotly","powerpoint"
    ]

    def count_skills(text):
        t = str(text).lower()
        cnt = 0
        for kw in skills:
            if re.search(r"\b" + re.escape(kw) + r"\b", t):
                cnt += 1
        return cnt

    df["skill_count"] = df[TEXT_COLUMN].apply(count_skills)

    # CHANGE THIS THRESHOLD IF TOO MANY/FEW positives
    threshold = 5
    df[AUTO_LABEL] = (df["skill_count"] >= threshold).astype(int)

    print(f"\nDistribution of skill_count (threshold={threshold}):")
    print(df["skill_count"].describe())
    print("\nLabel counts for hired_auto:")
    print(df[AUTO_LABEL].value_counts())

    print("\nShow 6 example resumes (skill_count, hired_auto):")
    sample = df[[TEXT_COLUMN, "skill_count", AUTO_LABEL]].sort_values("skill_count", ascending=False).head(6)
    for i, row in sample.iterrows():
        print("\n--- Example (skill_count, label):", row["skill_count"], row[AUTO_LABEL])
        print(row[TEXT_COLUMN][:300])
        print("-----")

# Prepare training data
X_raw = df[TEXT_COLUMN].astype(str).fillna("")
y = df[AUTO_LABEL].astype(int)

print("\nFinal label distribution used for training:")
print(y.value_counts())

# If only one class, abort and advise lowering threshold
if len(y.unique()) == 1:
    raise SystemExit("Only one label class present after automatic labeling. Lower the threshold and try again.")

# Load spaCy and clean
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

print("Cleaning text (this can take some time)...")
X = X_raw.apply(clean_text)

# Train/test split & train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

print("Training model...")
pipeline.fit(X_train, y_train)

print("Evaluating...")
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

MODEL_FILE = "resume_model_auto.joblib"
joblib.dump(pipeline, MODEL_FILE)
print(f"\nSaved model to {MODEL_FILE}")
df.to_csv("UpdatedResumeDataSet_with_auto_labels.csv", index=False)
print("Saved dataset with auto labels to UpdatedResumeDataSet_with_auto_labels.csv")
