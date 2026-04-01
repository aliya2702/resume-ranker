"""
history.py — Ranking History, Comparison & Export for ResumeRank.

Features:
  - Save ranking sessions per user
  - Load/list history
  - Candidate comparison (side-by-side)
  - CSV export
  - Email shortlist (SMTP)
"""

import os
import csv
import json
import uuid
import smtplib
import logging
from io import StringIO
from datetime import datetime
from typing import List, Dict, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")

# ─────────────────────────────────────────────
# History Store
# ─────────────────────────────────────────────

def _load_history() -> Dict[str, Any]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_history(data: Dict[str, Any]) -> None:
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error("Failed to save history: %s", e)


def save_ranking_session(
    user_id: str,
    user_email: str,
    job_description: str,
    ranked_candidates: List[Dict],
    total_processed: int,
    skipped: List[str],
    insights: Dict = None,
) -> str:
    """
    Persist a completed ranking session and return its session ID.
    """
    history = _load_history()

    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    jd_preview = job_description[:120].strip() + ("…" if len(job_description) > 120 else "")

    entry = {
        "session_id": session_id,
        "user_id": user_id,
        "user_email": user_email,
        "created_at": datetime.utcnow().isoformat(),
        "jd_preview": jd_preview,
        "job_description": job_description,
        "total_processed": total_processed,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "top_candidate": ranked_candidates[0]["name"] if ranked_candidates else None,
        "top_score": ranked_candidates[0]["score"] if ranked_candidates else 0,
        "avg_score": (
            round(sum(c["score"] for c in ranked_candidates) / len(ranked_candidates), 1)
            if ranked_candidates else 0
        ),
        "ranked_candidates": ranked_candidates,
        "insights": insights or {},
    }

    if user_id not in history:
        history[user_id] = []

    # Keep last 50 sessions per user
    history[user_id].insert(0, entry)
    history[user_id] = history[user_id][:50]

    _save_history(history)
    return session_id


def get_user_history(user_id: str) -> List[Dict]:
    """Return all sessions for a user (metadata only, no full candidate data)."""
    history = _load_history()
    sessions = history.get(user_id, [])

    return [
        {
            "session_id": s["session_id"],
            "created_at": s["created_at"],
            "jd_preview": s["jd_preview"],
            "total_processed": s["total_processed"],
            "skipped_count": s["skipped_count"],
            "top_candidate": s["top_candidate"],
            "top_score": s["top_score"],
            "avg_score": s["avg_score"],
        }
        for s in sessions
    ]


def get_session_by_id(user_id: str, session_id: str) -> Optional[Dict]:
    """Return full session data including ranked candidates."""
    history = _load_history()
    sessions = history.get(user_id, [])
    for s in sessions:
        if s["session_id"] == session_id:
            return s
    return None


def delete_session(user_id: str, session_id: str) -> bool:
    history = _load_history()
    sessions = history.get(user_id, [])
    original_len = len(sessions)
    history[user_id] = [s for s in sessions if s["session_id"] != session_id]
    _save_history(history)
    return len(history[user_id]) < original_len


# ─────────────────────────────────────────────
# Candidate Comparison
# ─────────────────────────────────────────────

def compare_candidates(
    candidates: List[Dict],
    selected_names: List[str],
) -> Dict[str, Any]:
    """
    Build a side-by-side comparison dict for the selected candidates.

    Args:
        candidates:     Full ranked_candidates list from a session.
        selected_names: Names of candidates to compare (2–4).

    Returns:
        Comparison dict with skill matrices and score breakdowns.
    """
    selected = [c for c in candidates if c["name"] in selected_names]

    if len(selected) < 2:
        raise ValueError("Select at least 2 candidates to compare.")

    # Gather all skill categories mentioned across all selected candidates
    all_categories = set()
    for c in selected:
        all_categories.update(c.get("matched_skills", {}).keys())
        all_categories.update(c.get("missing_skills", {}).keys())

    # Build skill matrix: category → {candidate_name: [skills]}
    skill_matrix = {}
    for cat in sorted(all_categories):
        skill_matrix[cat] = {}
        for c in selected:
            has = set(c.get("matched_skills", {}).get(cat, []))
            skill_matrix[cat][c["name"]] = {
                "matched": list(has),
                "missing": c.get("missing_skills", {}).get(cat, []),
            }

    return {
        "candidates": [
            {
                "name": c["name"],
                "score": c["score"],
                "tfidf_score": c.get("tfidf_score", 0),
                "skill_score": c.get("skill_score", 0),
                "explanation": c.get("explanation", ""),
                "warning": c.get("warning", ""),
                "risk_flags": c.get("risk_flags", []),
                "resume_strength": c.get("resume_strength", {}),
                "hidden_gem": c.get("hidden_gem", False),
            }
            for c in selected
        ],
        "skill_matrix": skill_matrix,
        "categories": sorted(all_categories),
    }


# ─────────────────────────────────────────────
# CSV Export
# ─────────────────────────────────────────────

def export_to_csv(ranked_candidates: List[Dict]) -> str:
    """
    Serialize ranked candidates to a CSV string.

    Returns:
        CSV content as a string (UTF-8).
    """
    output = StringIO()
    fieldnames = [
        "rank", "name", "score", "tfidf_score", "skill_score",
        "matched_skills", "missing_skills", "explanation", "warning",
        "hidden_gem", "risk_flags",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for c in ranked_candidates:
        # Flatten skill dicts to comma-separated strings
        matched_flat = ", ".join(
            s for skills in c.get("matched_skills", {}).values() for s in skills
        )
        missing_flat = ", ".join(
            s for skills in c.get("missing_skills", {}).values() for s in skills
        )
        risk_flat = "; ".join(c.get("risk_flags", []))

        writer.writerow({
            "rank": c.get("rank", ""),
            "name": c.get("name", ""),
            "score": c.get("score", ""),
            "tfidf_score": c.get("tfidf_score", ""),
            "skill_score": c.get("skill_score", ""),
            "matched_skills": matched_flat,
            "missing_skills": missing_flat,
            "explanation": c.get("explanation", ""),
            "warning": c.get("warning", ""),
            "hidden_gem": c.get("hidden_gem", False),
            "risk_flags": risk_flat,
        })

    return output.getvalue()


# ─────────────────────────────────────────────
# Email Shortlist
# ─────────────────────────────────────────────

def send_shortlist_email(
    recipient_email: str,
    recipient_name: str,
    shortlisted: List[Dict],
    job_description_preview: str,
    sender_name: str = "ResumeRank AI",
    smtp_host: str = None,
    smtp_port: int = 587,
    smtp_user: str = None,
    smtp_password: str = None,
) -> Dict[str, Any]:
    """
    Send a shortlist summary email.

    Falls back to returning the email content if SMTP is not configured.

    Returns:
        {"sent": bool, "preview": str (HTML email body)}
    """
    # Build HTML content
    candidate_rows = ""
    for c in shortlisted:
        matched_flat = ", ".join(
            s for skills in c.get("matched_skills", {}).values() for s in skills
        )[:80]
        badge_color = "#10b981" if c["score"] >= 70 else "#f59e0b" if c["score"] >= 50 else "#ef4444"
        gem_badge = "⭐ Hidden Gem  " if c.get("hidden_gem") else ""
        candidate_rows += f"""
        <tr style="border-bottom:1px solid #e5e7eb;">
          <td style="padding:12px 16px;font-weight:600;color:#111827;">#{c.get('rank','')} {c['name']}</td>
          <td style="padding:12px 16px;">
            <span style="background:{badge_color};color:white;padding:4px 10px;border-radius:20px;font-size:13px;font-weight:700;">
              {c['score']}%
            </span>
          </td>
          <td style="padding:12px 16px;color:#6b7280;font-size:13px;">{gem_badge}{matched_flat}</td>
          <td style="padding:12px 16px;color:#374151;font-size:13px;">{c.get('explanation','')[:120]}...</td>
        </tr>"""

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="font-family:'Segoe UI',sans-serif;background:#f9fafb;margin:0;padding:0;">
      <div style="max-width:800px;margin:40px auto;background:white;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.08);">
        <div style="background:linear-gradient(135deg,#1e293b,#334155);padding:32px 40px;">
          <h1 style="color:white;margin:0;font-size:24px;letter-spacing:-0.5px;">
            🏆 ResumeRank — Candidate Shortlist
          </h1>
          <p style="color:#94a3b8;margin:8px 0 0;">Prepared by {sender_name}</p>
        </div>
        <div style="padding:32px 40px;">
          <div style="background:#f1f5f9;border-radius:8px;padding:16px 20px;margin-bottom:28px;border-left:4px solid #6366f1;">
            <p style="margin:0;font-size:13px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">Job Description Preview</p>
            <p style="margin:8px 0 0;color:#1e293b;font-size:14px;">{job_description_preview}</p>
          </div>
          <h2 style="color:#1e293b;font-size:18px;margin:0 0 16px;">
            Top {len(shortlisted)} Candidates
          </h2>
          <table style="width:100%;border-collapse:collapse;font-size:14px;">
            <thead>
              <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
                <th style="padding:12px 16px;text-align:left;color:#64748b;font-weight:600;">Candidate</th>
                <th style="padding:12px 16px;text-align:left;color:#64748b;font-weight:600;">Score</th>
                <th style="padding:12px 16px;text-align:left;color:#64748b;font-weight:600;">Key Skills</th>
                <th style="padding:12px 16px;text-align:left;color:#64748b;font-weight:600;">Summary</th>
              </tr>
            </thead>
            <tbody>{candidate_rows}</tbody>
          </table>
          <p style="margin:32px 0 0;font-size:12px;color:#9ca3af;">
            Generated by ResumeRank AI · {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC
          </p>
        </div>
      </div>
    </body>
    </html>
    """

    # Try to send if SMTP is configured
    smtp_host = smtp_host or os.environ.get("SMTP_HOST")
    smtp_user = smtp_user or os.environ.get("SMTP_USER")
    smtp_password = smtp_password or os.environ.get("SMTP_PASSWORD")

    if smtp_host and smtp_user and smtp_password:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"ResumeRank — {len(shortlisted)} Shortlisted Candidates"
            msg["From"] = f"{sender_name} <{smtp_user}>"
            msg["To"] = f"{recipient_name} <{recipient_email}>"
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, recipient_email, msg.as_string())

            return {"sent": True, "preview": html_body}
        except Exception as e:
            logger.error("Email sending failed: %s", e)
            return {"sent": False, "preview": html_body, "error": str(e)}

    # No SMTP — return preview only
    return {"sent": False, "preview": html_body, "error": "SMTP not configured. Email preview generated."}
