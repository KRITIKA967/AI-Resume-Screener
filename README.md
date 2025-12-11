# 🧠 AI Powered Resume Screening System (ATS Alternative)

An advanced **AI-driven resume screening web application** designed to evaluate resumes,
predict whether a candidate should be **shortlisted**, detect job roles, extract resume text,
and manage candidates using a modern analytics dashboard.

This system is built using **Machine Learning (TF-IDF + Logistic Regression)**, **NLP**, **Flask**, and **Bootstrap**—offering features similar to **Indeed Resume Screener, Naukri ATS, and LinkedIn Recruiter**.

---

# 🚀 Features

### 🤖 AI Resume Shortlisting
- Predicts whether a resume should be **Shortlisted / Not Shortlisted**
- Provides **probability score**

### 🧩 Job Role Detection
Automatically predicts likely job role (Data Analyst, Python Developer, etc.)

### 📄 Resume Processing
- Upload PDF / TXT files  
- Drag & drop upload UI  
- Resume text extraction via PyMuPDF  

### 🔐 Admin Login
- Accepts **any email + any password** (for demo purposes)

### 📊 Dashboard
- View all screened candidates  
- Sort by probability  
- One-click resume viewer  
- Delete candidate  
- Export CSV  

### 📈 Analytics
- Pie chart: Shortlisted vs Rejected  
- Real-time API endpoint for chart data  

---

# 🧠 Machine Learning Model Details

### Dataset
The dataset contained:
- `Resume` (text)
- `Category` (job role)

Since it did not contain “Hired / Not Hired”, an **automatic labeling strategy** was used:

### Labeling Logic
- Extract skills from resume  
- Count technical skills  
- If skill count ≥ threshold → **Shortlisted (1)**  
- Else → **Not Shortlisted (0)**  

### Train / Test Split
- **80% Training**
- **20% Validation**

### Model Used
- **TF-IDF Vectorizer** → Convert text to numeric form  
- **Logistic Regression** → Best for binary text classification  

### Model Accuracy
✔ **99.48% Overall Accuracy**  
✔ **High Precision & Recall**  

### Final Model Saved As:
