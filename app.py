"""
app.py — Flask entry point for ResumeRank (v2).

New in v2:
  - Auth routes: /login, /logout, /api/auth/me, /api/users (admin)
  - History: /api/history, /api/history/<session_id>
  - Export: /api/export/csv/<session_id>
  - Email shortlist: /api/email-shortlist
  - Candidate comparison: /api/compare
  - JD improvement: /api/improve-jd
  - Full insights pipeline (clustering, risk flags, hidden gems, etc.)
"""

import os
import uuid
import json
import csv
from io import StringIO
from datetime import timedelta
from flask import (
    Flask, request, jsonify, render_template,
    session, redirect, url_for, send_file, make_response
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

from utils import extract_text_from_pdf, validate_resume_text
from ai_engine import rank_resumes
from auth import (
    authenticate, login_user, logout_user, get_current_user,
    login_required, admin_required, is_admin,
    list_users, create_user, update_user, delete_user,
)
from history import (
    save_ranking_session, get_user_history, get_session_by_id,
    delete_session, compare_candidates, export_to_csv, send_shortlist_email,
)
from insights import enrich_candidates, improve_job_description
from utils import extract_skills

# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "resumerank_dev_secret_change_in_prod")
app.permanent_session_lifetime = timedelta(hours=8)
CORS(app, supports_credentials=True)

UPLOAD_FOLDER  = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"pdf"}
MAX_RESUMES        = 30
MAX_FILE_SIZE_MB   = 5

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024 * MAX_RESUMES
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────
# Page Routes
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    user = get_current_user()
    if not user:
        return redirect(url_for("login_page"))
    return render_template("index.html")


@app.route("/login", methods=["GET"])
def login_page():
    if get_current_user():
        return redirect(url_for("home"))
    return render_template("login.html")


# ─────────────────────────────────────────────
# Auth API Routes
# ─────────────────────────────────────────────
@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data     = request.get_json(force=True) or {}
    email    = data.get("email", "").strip()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = authenticate(email, password)
    if not user:
        return jsonify({"error": "Invalid email or password."}), 401

    login_user(user)
    return jsonify({
        "message": "Login successful.",
        "user": {
            "id":    user["id"],
            "name":  user["name"],
            "email": user["email"],
            "role":  user["role"],
        }
    })


@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    logout_user()
    return jsonify({"message": "Logged out."})


@app.route("/api/auth/me", methods=["GET"])
@login_required
def api_me():
    user = get_current_user()
    return jsonify({"user": user})


# ─────────────────────────────────────────────
# User Management (Admin Only)
# ─────────────────────────────────────────────
@app.route("/api/users", methods=["GET"])
@admin_required
def api_list_users():
    return jsonify({"users": list_users()})


@app.route("/api/users", methods=["POST"])
@admin_required
def api_create_user():
    data = request.get_json(force=True) or {}
    try:
        user = create_user(
            name     = data.get("name", ""),
            email    = data.get("email", ""),
            password = data.get("password", ""),
            role     = data.get("role", "hr"),
        )
        return jsonify({"user": user}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/users/<email>", methods=["PATCH"])
@admin_required
def api_update_user(email):
    data = request.get_json(force=True) or {}
    try:
        user = update_user(email, data)
        return jsonify({"user": user})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/users/<email>", methods=["DELETE"])
@admin_required
def api_delete_user(email):
    try:
        delete_user(email)
        return jsonify({"message": f"User '{email}' deleted."})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ─────────────────────────────────────────────
# Core Ranking Route
# ─────────────────────────────────────────────
@app.route("/api/rank", methods=["POST"])
@login_required
def rank():
    """
    POST /api/rank
    Accepts: multipart/form-data (job_description + resumes)
    Returns: full ranked + enriched candidate list with insights
    """
    user = get_current_user()

    # ── Validate job description ──────────────
    job_description = request.form.get("job_description", "").strip()
    if not job_description:
        return jsonify({"error": "Job description is required."}), 400
    if len(job_description) < 50:
        return jsonify({"error": "Job description is too short (min 50 characters)."}), 400

    # ── Validate uploaded files ───────────────
    uploaded_files = request.files.getlist("resumes")
    if not uploaded_files or all(f.filename == "" for f in uploaded_files):
        return jsonify({"error": "At least one resume PDF is required."}), 400
    if len(uploaded_files) > MAX_RESUMES:
        return jsonify({"error": f"Maximum {MAX_RESUMES} resumes allowed."}), 400

    # ── Process each resume ───────────────────
    resumes_data = []
    errors       = []
    seen_texts   = set()

    for file in uploaded_files:
        if file.filename == "":
            continue
        if not allowed_file(file.filename):
            errors.append(f"'{file.filename}' is not a PDF and was skipped.")
            continue

        original_name = secure_filename(file.filename)
        temp_name     = f"{uuid.uuid4().hex}_{original_name}"
        save_path     = os.path.join(app.config["UPLOAD_FOLDER"], temp_name)

        try:
            file.save(save_path)
            text = extract_text_from_pdf(save_path)
        except Exception as e:
            errors.append(f"Failed to parse '{original_name}': {str(e)}")
            continue
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

        is_valid, warning = validate_resume_text(text)
        if not is_valid:
            errors.append(f"'{original_name}' appears to be empty or unreadable.")
            continue

        text_hash = hash(text.strip().lower())
        if text_hash in seen_texts:
            errors.append(f"'{original_name}' is a duplicate and was skipped.")
            continue
        seen_texts.add(text_hash)

        candidate_name = original_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
        resumes_data.append({
            "name":     candidate_name,
            "filename": original_name,
            "text":     text,
            "warning":  warning,
        })

    if not resumes_data:
        return jsonify({"error": "No valid resumes could be processed.", "details": errors}), 422

    # ── AI Ranking ────────────────────────────
    try:
        ranked_candidates = rank_resumes(job_description, resumes_data)
    except Exception as e:
        return jsonify({"error": f"Ranking failed: {str(e)}"}), 500

    # ── Enrichment Pipeline ───────────────────
    # Temporarily attach raw text for enrichment
    for i, c in enumerate(ranked_candidates):
        c["_resume_text"] = resumes_data[i]["text"] if i < len(resumes_data) else ""

    jd_skills = extract_skills(job_description)

    try:
        ranked_candidates, insights = enrich_candidates(ranked_candidates, job_description, jd_skills)
    except Exception as e:
        insights = {}

    # Strip internal fields
    for c in ranked_candidates:
        c.pop("_resume_text", None)

    # ── Persist to history ────────────────────
    try:
        session_id = save_ranking_session(
            user_id           = user["id"],
            user_email        = user["email"],
            job_description   = job_description,
            ranked_candidates = ranked_candidates,
            total_processed   = len(resumes_data),
            skipped           = errors,
            insights          = insights,
        )
    except Exception:
        session_id = None

    return jsonify({
        "session_id":        session_id,
        "ranked_candidates": ranked_candidates,
        "total_processed":   len(resumes_data),
        "skipped":           errors,
        "insights":          insights,
    })


# ─────────────────────────────────────────────
# JD Improvement
# ─────────────────────────────────────────────
@app.route("/api/improve-jd", methods=["POST"])
@login_required
def api_improve_jd():
    data = request.get_json(force=True) or {}
    jd   = data.get("job_description", "").strip()
    if not jd:
        return jsonify({"error": "Job description is required."}), 400

    improved = improve_job_description(jd)
    return jsonify({"improved_jd": improved})


# ─────────────────────────────────────────────
# History Routes
# ─────────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
@login_required
def api_get_history():
    user = get_current_user()
    history = get_user_history(user["id"])
    return jsonify({"history": history})


@app.route("/api/history/<session_id>", methods=["GET"])
@login_required
def api_get_session(session_id):
    user = get_current_user()
    sess = get_session_by_id(user["id"], session_id)
    if not sess:
        return jsonify({"error": "Session not found."}), 404
    return jsonify({"session": sess})


@app.route("/api/history/<session_id>", methods=["DELETE"])
@login_required
def api_delete_session(session_id):
    user = get_current_user()
    deleted = delete_session(user["id"], session_id)
    if not deleted:
        return jsonify({"error": "Session not found."}), 404
    return jsonify({"message": "Session deleted."})


# ─────────────────────────────────────────────
# Export Route
# ─────────────────────────────────────────────
@app.route("/api/export/csv/<session_id>", methods=["GET"])
@login_required
def api_export_csv(session_id):
    user = get_current_user()
    sess = get_session_by_id(user["id"], session_id)
    if not sess:
        return jsonify({"error": "Session not found."}), 404

    csv_content = export_to_csv(sess["ranked_candidates"])
    response    = make_response(csv_content)
    response.headers["Content-Type"]        = "text/csv"
    response.headers["Content-Disposition"] = f"attachment; filename=resumerank_{session_id}.csv"
    return response


# ─────────────────────────────────────────────
# Comparison Route
# ─────────────────────────────────────────────
@app.route("/api/compare", methods=["POST"])
@login_required
def api_compare():
    user = get_current_user()
    data = request.get_json(force=True) or {}

    session_id     = data.get("session_id")
    selected_names = data.get("candidates", [])

    if not session_id:
        return jsonify({"error": "session_id is required."}), 400
    if len(selected_names) < 2:
        return jsonify({"error": "Select at least 2 candidates to compare."}), 400

    sess = get_session_by_id(user["id"], session_id)
    if not sess:
        return jsonify({"error": "Session not found."}), 404

    try:
        comparison = compare_candidates(sess["ranked_candidates"], selected_names)
        return jsonify({"comparison": comparison})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────────
# Email Shortlist Route
# ─────────────────────────────────────────────
@app.route("/api/email-shortlist", methods=["POST"])
@login_required
def api_email_shortlist():
    user = get_current_user()
    data = request.get_json(force=True) or {}

    recipient_email = data.get("recipient_email", "").strip()
    recipient_name  = data.get("recipient_name", "HR Team")
    session_id      = data.get("session_id")
    top_n           = int(data.get("top_n", 5))

    if not recipient_email:
        return jsonify({"error": "Recipient email is required."}), 400
    if not session_id:
        return jsonify({"error": "session_id is required."}), 400

    sess = get_session_by_id(user["id"], session_id)
    if not sess:
        return jsonify({"error": "Session not found."}), 404

    shortlisted = sess["ranked_candidates"][:top_n]
    result = send_shortlist_email(
        recipient_email       = recipient_email,
        recipient_name        = recipient_name,
        shortlisted           = shortlisted,
        job_description_preview = sess["jd_preview"],
        sender_name           = user["name"],
    )

    return jsonify(result)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
