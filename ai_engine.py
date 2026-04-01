"""
ai_engine.py — Core AI/NLP logic for the Resume Ranking System.

Pipeline:
  1. Preprocess job description and resume texts.
  2. Compute TF-IDF + Cosine Similarity scores (baseline).
  3. Compute skill-overlap bonus score.
  4. Blend scores into a final 0–100 ranking score.
  5. Generate a human-readable explanation per candidate using
     Claude (via Anthropic API) with a rule-based fallback.

Design Goals:
  - Deterministic, reproducible results.
  - Explainable scores (not a black box).
  - Graceful degradation: works without the Anthropic API key.
"""

import os
import re
import json
import logging
import requests
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import extract_skills, get_skill_gap, SKILL_GROUPS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-sonnet-4-20250514"

# Score blend weights (must sum to 1.0)
WEIGHT_TFIDF = 0.55   # TF-IDF cosine similarity
WEIGHT_SKILL = 0.45   # Skill keyword overlap


# ─────────────────────────────────────────────
# Text Preprocessing
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    Lowercase, remove punctuation, and collapse whitespace.
    Keeps the vocabulary clean for TF-IDF.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    return text.strip()


# ─────────────────────────────────────────────
# TF-IDF Scoring
# ─────────────────────────────────────────────
def compute_tfidf_scores(jd_text: str, resume_texts: List[str]) -> List[float]:
    """
    Vectorise the job description and all resumes with TF-IDF,
    then compute cosine similarity between the JD and each resume.

    Args:
        jd_text:      Preprocessed job description text.
        resume_texts: List of preprocessed resume texts.

    Returns:
        List of float similarity scores in [0, 1].
    """
    corpus = [jd_text] + resume_texts  # JD is index 0

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),       # unigrams + bigrams for richer features
        max_df=0.95,              # ignore near-universal terms
        min_df=1,                 # keep rare but potentially important terms
        sublinear_tf=True,        # dampen high-frequency terms
        stop_words="english",
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vector    = tfidf_matrix[0]           # shape: (1, vocab)
    resume_matrix = tfidf_matrix[1:]         # shape: (n_resumes, vocab)

    similarities = cosine_similarity(jd_vector, resume_matrix)[0]
    return similarities.tolist()


# ─────────────────────────────────────────────
# Skill Overlap Scoring
# ─────────────────────────────────────────────
def compute_skill_score(jd_skills: dict, resume_skills: dict) -> float:
    """
    Compute a skill overlap ratio between the JD and a resume.

    Score = (# skills in JD that candidate also has) / (# skills in JD)

    Returns:
        Float in [0, 1]. Returns 0 if the JD has no detectable skills.
    """
    jd_flat     = {s for skills in jd_skills.values() for s in skills}
    resume_flat = {s for skills in resume_skills.values() for s in skills}

    if not jd_flat:
        return 0.0

    matched_count = len(jd_flat & resume_flat)
    return matched_count / len(jd_flat)


# ─────────────────────────────────────────────
# Explanation Generation (LLM)
# ─────────────────────────────────────────────
def generate_explanation_llm(
    candidate_name: str,
    score: int,
    jd_text: str,
    resume_text: str,
    matched_skills: dict,
    missing_skills: dict,
) -> str:
    """
    Call the Anthropic Claude API to generate a contextual explanation.

    Falls back to rule-based explanation if the API call fails or no key is set.

    Args:
        candidate_name: Display name of the candidate.
        score:          Final ranking score (0–100).
        jd_text:        Original job description.
        resume_text:    Original resume text (truncated for prompt efficiency).
        matched_skills: Skills the candidate has that the JD requires.
        missing_skills: Skills the JD requires that the candidate lacks.

    Returns:
        Human-readable explanation string.
    """
    if not ANTHROPIC_API_KEY:
        logger.info("No ANTHROPIC_API_KEY set — using rule-based explanation.")
        return generate_explanation_rule_based(
            score, matched_skills, missing_skills
        )

    # Truncate long texts to keep the prompt concise (and cost-effective)
    jd_snippet     = jd_text[:1500]
    resume_snippet = resume_text[:2000]
    matched_str    = json.dumps(matched_skills, indent=2)
    missing_str    = json.dumps(missing_skills, indent=2)

    prompt = f"""You are an expert HR analyst evaluating a candidate for a software job.

Job Description (excerpt):
\"\"\"
{jd_snippet}
\"\"\"

Candidate Resume (excerpt):
\"\"\"
{resume_snippet}
\"\"\"

Matched Skills (candidate has these, JD requires them):
{matched_str}

Missing Skills (JD requires these, candidate lacks them):
{missing_str}

Overall Ranking Score: {score}/100

Write a concise 2–3 sentence explanation of why this candidate received this score.
Be specific: mention actual skills, experience, or gaps. Avoid generic phrases.
Do NOT start with "Based on" or "The candidate". Be direct and professional."""

    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        explanation = data["content"][0]["text"].strip()
        return explanation

    except Exception as e:
        logger.warning("LLM explanation failed (%s), using rule-based fallback.", e)
        return generate_explanation_rule_based(score, matched_skills, missing_skills)


def generate_explanation_rule_based(
    score: int,
    matched_skills: dict,
    missing_skills: dict,
) -> str:
    """
    Generate a deterministic, rule-based explanation when the LLM is unavailable.

    Args:
        score:          Final score (0–100).
        matched_skills: Skills the candidate matched.
        missing_skills: Skills the candidate is missing.

    Returns:
        Explanation string.
    """
    matched_flat = [s for skills in matched_skills.values() for s in skills]
    missing_flat = [s for skills in missing_skills.values() for s in skills]

    # ── Score band label ──────────────────────
    if score >= 85:
        band = "Excellent match."
    elif score >= 70:
        band = "Strong match."
    elif score >= 55:
        band = "Moderate match."
    elif score >= 40:
        band = "Partial match."
    else:
        band = "Weak match."

    # ── Matched skills sentence ───────────────
    if matched_flat:
        top_matched = matched_flat[:6]
        matched_str = ", ".join(top_matched)
        matched_sentence = f"Demonstrates proficiency in: {matched_str}."
    else:
        matched_sentence = "No specific required skills were detected in the resume."

    # ── Missing skills sentence ───────────────
    if missing_flat:
        top_missing = missing_flat[:4]
        missing_str = ", ".join(top_missing)
        missing_sentence = f"Gaps identified in: {missing_str}."
    else:
        missing_sentence = "Covers all detected skill requirements from the job description."

    return f"{band} {matched_sentence} {missing_sentence}"


# ─────────────────────────────────────────────
# Main Ranking Function
# ─────────────────────────────────────────────
def rank_resumes(
    job_description: str,
    resumes_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Rank a list of resumes against a job description.

    Args:
        job_description: Raw JD text (from form input).
        resumes_data:    List of dicts with keys: name, filename, text, warning.

    Returns:
        List of candidate dicts sorted by score descending, each containing:
          - name
          - score (int, 0–100)
          - tfidf_score (float, for transparency)
          - skill_score (float, for transparency)
          - matched_skills (dict)
          - missing_skills (dict)
          - explanation (str)
          - warning (str)
    """
    if not resumes_data:
        return []

    # ── Preprocess texts ──────────────────────
    jd_clean      = preprocess(job_description)
    resume_cleans = [preprocess(r["text"]) for r in resumes_data]

    # ── TF-IDF cosine similarity ──────────────
    tfidf_scores = compute_tfidf_scores(jd_clean, resume_cleans)

    # ── Skill extraction ──────────────────────
    jd_skills = extract_skills(job_description)

    # ── Build ranked results ──────────────────
    results = []
    for idx, resume in enumerate(resumes_data):
        resume_skills  = extract_skills(resume["text"])
        missing_skills = get_skill_gap(jd_skills, resume_skills)

        tfidf_score = tfidf_scores[idx]
        skill_score = compute_skill_score(jd_skills, resume_skills)

        # Blend into final score
        raw_score    = (WEIGHT_TFIDF * tfidf_score) + (WEIGHT_SKILL * skill_score)
        final_score  = min(100, int(raw_score * 100))

        # Generate explanation
        explanation = generate_explanation_llm(
            candidate_name  = resume["name"],
            score           = final_score,
            jd_text         = job_description,
            resume_text     = resume["text"],
            matched_skills  = resume_skills,
            missing_skills  = missing_skills,
        )

        results.append({
            "name":           resume["name"],
            "filename":       resume["filename"],
            "score":          final_score,
            "tfidf_score":    round(tfidf_score * 100, 1),
            "skill_score":    round(skill_score * 100, 1),
            "matched_skills": resume_skills,
            "missing_skills": missing_skills,
            "explanation":    explanation,
            "warning":        resume.get("warning", ""),
        })

    # Sort descending by score
    results.sort(key=lambda x: x["score"], reverse=True)

    # Assign ranks (1-indexed)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results
