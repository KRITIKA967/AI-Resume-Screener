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

print("\nTop 'Category' values (sample):")
print(df[CATEGORY_COLUMN].astype(str).str.strip().value_counts().head(20))

# Create automatic label using Data Analyst skills
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

threshold = 3  # improved threshold
df[AUTO_LABEL] = (df["skill_count"] >= threshold).astype(int)

print(f"\nUsing threshold = {threshold}")
print("Label distribution (hired_auto):")
print(df[AUTO_LABEL].value_counts())

# Abort if only one class
if len(df[AUTO_LABEL].unique()) == 1:
    raise SystemExit("Only one label class present after auto-labeling. Lower threshold further.")

# Prepare data
X_raw = df[TEXT_COLUMN].astype(str).fillna("")
y = df[AUTO_LABEL].astype(int)

# Clean text using spaCy
print("\nLoading spaCy...")
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

def clean_text(text):
    doc = nlp(text)
    return " ".join(
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    )

print("Cleaning text... (this may take time)")
X = X_raw.apply(clean_text)

# Split + Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

print("\nTraining model...")
pipeline.fit(X_train, y_train)

print("\nEvaluating model...")
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
MODEL_FILE = "resume_model_improved.joblib"
joblib.dump(pipeline, MODEL_FILE)
print(f"\nSaved improved model to {MODEL_FILE}")

df.to_csv("UpdatedResumeDataSet_with_auto_labels.csv", index=False)
print("Saved updated dataset with auto labels.")
