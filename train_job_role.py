import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

df = pd.read_csv("UpdatedResumeDataSet.csv")
# assume Category column contains hard-coded job roles
df = df[df["Category"].notna()]
X = df["Resume"].astype(str)
y = df["Category"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])
pipe.fit(X_train, y_train)
print("Train done")
joblib.dump(pipe, "job_role_model.joblib")
