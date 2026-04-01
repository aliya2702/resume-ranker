"""
auth.py — Authentication & Authorization for ResumeRank v2.
"""

import os
import json
import hashlib
import secrets
import logging
from datetime import datetime
from functools import wraps
from typing import Optional, Dict, Any

from flask import session, jsonify

logger = logging.getLogger(__name__)

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


def _hash_password(password: str) -> str:
    salted = f"resumerank_salt_{password}_2024"
    return hashlib.sha256(salted.encode()).hexdigest()


DEFAULT_USERS = {
    "admin@resumerank.ai": {
        "id": "usr_001",
        "name": "Admin User",
        "email": "admin@resumerank.ai",
        "password_hash": _hash_password("Admin@123"),
        "role": "admin",
        "created_at": "2024-01-01T00:00:00",
        "last_login": None,
        "login_count": 0,
        "active": True,
    },
    "hr@resumerank.ai": {
        "id": "usr_002",
        "name": "HR Manager",
        "email": "hr@resumerank.ai",
        "password_hash": _hash_password("HR@12345"),
        "role": "hr",
        "created_at": "2024-01-01T00:00:00",
        "last_login": None,
        "login_count": 0,
        "active": True,
    },
}


def _load_users() -> Dict[str, Any]:
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    _save_users(DEFAULT_USERS)
    return DEFAULT_USERS.copy()


def _save_users(users: Dict[str, Any]) -> None:
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        logger.error("Failed to save users: %s", e)


def authenticate(email: str, password: str) -> Optional[Dict[str, Any]]:
    users = _load_users()
    user  = users.get(email.lower().strip())
    if not user or not user.get("active", True):
        return None
    if user["password_hash"] != _hash_password(password):
        return None
    user["last_login"]  = datetime.utcnow().isoformat()
    user["login_count"] = user.get("login_count", 0) + 1
    users[email.lower().strip()] = user
    _save_users(users)
    return {k: v for k, v in user.items() if k != "password_hash"}


def get_current_user() -> Optional[Dict[str, Any]]:
    user_id = session.get("user_id")
    if not user_id:
        return None
    users = _load_users()
    for user in users.values():
        if user.get("id") == user_id:
            return {k: v for k, v in user.items() if k != "password_hash"}
    return None


def login_user(user: Dict[str, Any]) -> None:
    session.permanent = True
    session["user_id"]    = user["id"]
    session["user_email"] = user["email"]
    session["user_role"]  = user["role"]
    session["user_name"]  = user["name"]


def logout_user() -> None:
    session.clear()


def is_admin() -> bool:
    return session.get("user_role") == "admin"


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not get_current_user():
            return jsonify({"error": "Authentication required.", "redirect": "/login"}), 401
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not get_current_user():
            return jsonify({"error": "Authentication required.", "redirect": "/login"}), 401
        if not is_admin():
            return jsonify({"error": "Admin access required."}), 403
        return f(*args, **kwargs)
    return decorated


def list_users() -> list:
    users = _load_users()
    return [{k: v for k, v in u.items() if k != "password_hash"} for u in users.values()]


def create_user(name: str, email: str, password: str, role: str) -> Dict[str, Any]:
    users = _load_users()
    email = email.lower().strip()
    if email in users:
        raise ValueError(f"User '{email}' already exists.")
    if role not in ("hr", "admin"):
        raise ValueError("Role must be 'hr' or 'admin'.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    new_user = {
        "id": f"usr_{secrets.token_hex(4)}",
        "name": name,
        "email": email,
        "password_hash": _hash_password(password),
        "role": role,
        "created_at": datetime.utcnow().isoformat(),
        "last_login": None,
        "login_count": 0,
        "active": True,
    }
    users[email] = new_user
    _save_users(users)
    return {k: v for k, v in new_user.items() if k != "password_hash"}


def update_user(email: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    users = _load_users()
    email = email.lower().strip()
    if email not in users:
        raise ValueError(f"User '{email}' not found.")
    user = users[email]
    for field, value in updates.items():
        if field in {"name", "role", "active"}:
            user[field] = value
        elif field == "password":
            user["password_hash"] = _hash_password(value)
    users[email] = user
    _save_users(users)
    return {k: v for k, v in user.items() if k != "password_hash"}


def delete_user(email: str) -> None:
    users = _load_users()
    email = email.lower().strip()
    if email not in users:
        raise ValueError(f"User '{email}' not found.")
    del users[email]
    _save_users(users)
