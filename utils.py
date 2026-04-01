"""
utils.py — Helper utilities for the Resume Ranking System.

Responsibilities:
  - PDF text extraction (with pdfplumber primary, PyPDF2 fallback)
  - Text cleaning and normalisation
  - Resume validation
  - Skill keyword extraction
"""

import re
import logging
from typing import Tuple, List

# PDF parsing libraries (pdfplumber preferred for accuracy)
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

if not PDFPLUMBER_AVAILABLE and not PYPDF2_AVAILABLE:
    raise ImportError("Install at least one PDF library: pdfplumber or PyPDF2.")

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Skill Keyword Dictionary
# ─────────────────────────────────────────────
# These are grouped so we can later show matched/missing skills per category.
SKILL_GROUPS = {
    "Languages": [
        "javascript", "typescript", "python", "java", "c++", "c#", "ruby",
        "go", "rust", "php", "swift", "kotlin", "dart", "scala",
    ],
    "Frontend Frameworks": [
        "react", "reactjs", "react.js", "vue", "vuejs", "vue.js", "angular",
        "svelte", "nextjs", "next.js", "nuxt", "gatsby", "remix",
    ],
    "Styling": [
        "css", "sass", "scss", "less", "tailwind", "bootstrap", "material-ui",
        "styled-components", "emotion", "chakra", "ant design", "figma",
    ],
    "State Management": [
        "redux", "zustand", "mobx", "recoil", "context api", "pinia", "vuex",
    ],
    "Build Tools": [
        "webpack", "vite", "parcel", "rollup", "babel", "esbuild", "turbopack",
    ],
    "Testing": [
        "jest", "vitest", "cypress", "playwright", "testing library",
        "mocha", "chai", "enzyme",
    ],
    "Backend & APIs": [
        "node", "nodejs", "node.js", "express", "fastapi", "flask", "django",
        "rest", "graphql", "websocket", "grpc",
    ],
    "Version Control & DevOps": [
        "git", "github", "gitlab", "ci/cd", "docker", "kubernetes",
        "aws", "azure", "gcp", "vercel", "netlify",
    ],
    "Databases": [
        "mongodb", "postgresql", "mysql", "sqlite", "firebase", "supabase",
        "redis", "prisma",
    ],
    "Soft Skills & Experience": [
        "agile", "scrum", "team lead", "leadership", "communication",
        "problem solving", "mentoring", "code review",
    ],
}

# Flat list of all known skills (lowercase)
ALL_SKILLS: List[str] = [
    skill for skills in SKILL_GROUPS.values() for skill in skills
]


# ─────────────────────────────────────────────
# PDF Text Extraction
# ─────────────────────────────────────────────
def extract_text_from_pdf(filepath: str) -> str:
    """
    Extract raw text from a PDF file.

    Tries pdfplumber first (better table/layout handling),
    falls back to PyPDF2 if pdfplumber is unavailable or fails.

    Args:
        filepath: Absolute path to the PDF file.

    Returns:
        Extracted text as a single string.

    Raises:
        ValueError: If neither library can extract text.
    """
    text = ""

    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(filepath) as pdf:
                pages = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)
                text = "\n".join(pages)
        except Exception as e:
            logger.warning("pdfplumber failed (%s), trying PyPDF2 fallback.", e)

    # Fallback to PyPDF2 if pdfplumber gave nothing
    if not text.strip() and PYPDF2_AVAILABLE:
        try:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)
                text = "\n".join(pages)
        except Exception as e:
            logger.error("PyPDF2 also failed: %s", e)

    return clean_text(text)


# ─────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Normalise raw PDF text for NLP processing.

    Steps:
      1. Replace non-breaking spaces and special whitespace.
      2. Remove non-printable / control characters.
      3. Collapse excessive whitespace.
      4. Strip leading/trailing whitespace.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Replace common unicode whitespace variants
    text = text.replace("\xa0", " ").replace("\t", " ")

    # Remove control characters (keep newlines)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    # Collapse more than two consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ─────────────────────────────────────────────
# Resume Validation
# ─────────────────────────────────────────────
MIN_RESUME_CHARS = 100    # Fewer chars → likely empty/image-only PDF
SHORT_RESUME_CHARS = 300  # Warn but still process

def validate_resume_text(text: str) -> Tuple[bool, str]:
    """
    Validate extracted resume text.

    Args:
        text: Cleaned text from a resume.

    Returns:
        (is_valid, warning_message)
        is_valid      — False means skip this resume entirely.
        warning_message — Non-empty string if there is a soft warning.
    """
    char_count = len(text.strip())

    if char_count < MIN_RESUME_CHARS:
        return False, "Resume is empty or could not be read (possibly a scanned image)."

    if char_count < SHORT_RESUME_CHARS:
        return True, "Resume is very short — score may be less accurate."

    return True, ""


# ─────────────────────────────────────────────
# Skill Extraction
# ─────────────────────────────────────────────
def extract_skills(text: str) -> dict:
    """
    Scan text for known skill keywords, grouped by category.

    Args:
        text: Resume or job-description text (will be lowercased internally).

    Returns:
        Dict mapping category name → list of matched skills.
        Example: {"Frontend Frameworks": ["react", "nextjs"], ...}
    """
    lower = text.lower()
    matched: dict = {}

    for category, skills in SKILL_GROUPS.items():
        found = []
        for skill in skills:
            # Use word-boundary matching to avoid partial matches
            # e.g. "react" should not match "reactive"
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, lower):
                found.append(skill)
        if found:
            matched[category] = found

    return matched


def get_skill_gap(jd_skills: dict, resume_skills: dict) -> dict:
    """
    Compute skills present in the JD but missing from the resume.

    Args:
        jd_skills:     Skills extracted from the job description.
        resume_skills: Skills extracted from a candidate's resume.

    Returns:
        Dict mapping category → list of missing skills.
    """
    gap: dict = {}

    for category, required in jd_skills.items():
        candidate_has = set(resume_skills.get(category, []))
        missing = [s for s in required if s not in candidate_has]
        if missing:
            gap[category] = missing

    return gap
