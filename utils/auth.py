# utils/auth.py
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DB = "candidates.db"

def init_admin(admin_username, admin_password, admin_email):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Create users table with email column
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password_hash TEXT
        )
    """)

    # Check if admin already exists
    cur.execute("SELECT id FROM users WHERE username=? OR email=?", (admin_username, admin_email))
    row = cur.fetchone()

    # If admin does not exist → create it
    if row is None:
        ph = generate_password_hash(admin_password)
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (admin_username, admin_email, ph)
        )
        print("✔ Admin user created")

    conn.commit()
    conn.close()


def verify_user(email, password):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("SELECT password_hash FROM users WHERE email=?", (email,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return False

    return check_password_hash(row[0], password)
