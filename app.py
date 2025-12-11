# app.py (LOGIN ACCEPTS ANY EMAIL + ANY PASSWORD)
from flask import Flask, render_template, request, redirect, jsonify, url_for, Response
import joblib, sqlite3, os, datetime, csv, io
from utils.pdf_reader import extract_text_from_pdf
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "change_me")

app = Flask(__name__)
app.secret_key = SECRET_KEY

# --------------------------
# FLASK LOGIN SETUP
# --------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class SimpleUser(UserMixin):
    def __init__(self, id, email):
        self.id = id
        self.email = email


@login_manager.user_loader
def load_user(user_id):
    return SimpleUser(id=1, email="demo@example.com")  # static user for login


# --------------------------
# LOAD MODEL
# --------------------------
model = joblib.load("resume_model_improved.joblib")

JOB_ROLE_MODEL_PATH = "job_role_model.joblib"
job_role_model = joblib.load(JOB_ROLE_MODEL_PATH) if os.path.exists(JOB_ROLE_MODEL_PATH) else None


# --------------------------
# DATABASE INIT
# --------------------------
def init_db():
    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            resume TEXT,
            prediction TEXT,
            probability REAL,
            job_role TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


# --------------------------
# LOGIN — ACCEPT ANY EMAIL/PASSWORD
# --------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # ACCEPT ANY EMAIL + PASSWORD
        user = SimpleUser(id=1, email=email)
        login_user(user)

        return redirect(url_for("home"))

    return render_template("login.html")


# LOGOUT
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# --------------------------
# HOME PAGE
# --------------------------
@app.route("/")
def home():
    return render_template("index.html")


# --------------------------
# PREDICT ROUTE
# --------------------------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    name = request.form.get("name", "Unknown")
    uploaded_file = request.files.get("resume_file")
    pasted_text = request.form.get("resume", "")
    resume_text = ""

    if uploaded_file and uploaded_file.filename != "":
        file_path = f"uploads/{datetime.datetime.now().timestamp()}_{uploaded_file.filename}"
        os.makedirs("uploads", exist_ok=True)
        uploaded_file.save(file_path)

        resume_text = extract_text_from_pdf(file_path) if uploaded_file.filename.lower().endswith(".pdf") \
            else open(file_path, encoding="utf8", errors="ignore").read()
    else:
        resume_text = pasted_text

    prob = model.predict_proba([resume_text])[0][1]
    label = "Shortlisted" if prob >= 0.5 else "Not Shortlisted"

    jobrole = None
    if job_role_model:
        try:
            jobrole = job_role_model.predict([resume_text])[0]
        except:
            jobrole = None

    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO candidates (name, resume, prediction, probability, job_role, created_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
    """, (name, resume_text, label, float(prob), jobrole))
    conn.commit()
    conn.close()

    return render_template("result.html",
                           prediction=label,
                           probability=round(prob, 3),
                           job_role=jobrole)


# --------------------------
# DASHBOARD
# --------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()
    cur.execute("SELECT id, name, prediction, probability, job_role, created_at FROM candidates ORDER BY probability DESC")
    candidates = cur.fetchall()
    conn.close()
    return render_template("dashboard.html", candidates=candidates)


@app.route("/view/<int:candidate_id>")
@login_required
def view_candidate(candidate_id):
    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()
    cur.execute("SELECT name, resume, prediction, probability, job_role, created_at FROM candidates WHERE id=?", (candidate_id,))
    row = cur.fetchone()
    conn.close()
    return render_template("view_resume.html", candidate=row)


@app.route("/delete/<int:candidate_id>")
@login_required
def delete_candidate(candidate_id):
    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM candidates WHERE id=?", (candidate_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("dashboard"))


@app.route("/export")
@login_required
def export_csv():
    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()
    cur.execute("SELECT id, name, prediction, probability, job_role, created_at FROM candidates")
    rows = cur.fetchall()
    conn.close()

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["ID", "Name", "Prediction", "Probability", "Job Role", "Date"])
    cw.writerows(rows)

    return Response(si.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=candidates.csv"})


@app.route("/chart_data")
@login_required
def chart_data():
    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM candidates WHERE prediction='Shortlisted'")
    shortlisted = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM candidates WHERE prediction='Not Shortlisted'")
    rejected = cur.fetchone()[0]

    return jsonify({"shortlisted": shortlisted, "not_shortlisted": rejected})


if __name__ == "__main__":
    app.run(debug=True)
